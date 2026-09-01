//! Exercise the real signed model updater around a policy A/B test.
//!
//! `activate` installs an exact source rollback target and then an adapted policy.
//! `rollback` is a separate process invocation, proving the updater can recover from
//! its persisted journal/store rather than from in-memory test state.

use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use test_support::Publisher;
use updater::config::Config;
use updater::engine::{ApplyOptions, Engine};
use updater::faults::Faults;
use updater::proto::{ApplyResult, Target};
use updater::robot::{Health, RobotClient, SafeToRestart};
use updater::verify::KeyRing;

#[derive(Parser)]
#[command(about = "Signed updater proof for the EGGROLL Policy Patch Lab")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Activate {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        adapted: PathBuf,
        /// Updater component backing the patched runtime slot, e.g. model-stand.
        #[arg(long)]
        component: String,
    },
    Rollback {
        #[arg(long)]
        root: PathBuf,
    },
}

#[derive(Clone, Default)]
struct GateAudit {
    health_calls: Arc<AtomicUsize>,
    model_api_calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl RobotClient for GateAudit {
    async fn safe_to_restart(&self, _timeout: Duration) -> SafeToRestart {
        SafeToRestart::Yes
    }

    async fn health(&self, _timeout: Duration) -> Health {
        self.health_calls.fetch_add(1, Ordering::SeqCst);
        Health::Healthy
    }

    async fn model_api(&self, _timeout: Duration) -> Option<u32> {
        self.model_api_calls.fetch_add(1, Ordering::SeqCst);
        Some(1)
    }

    async fn remote_session_active(&self, _timeout: Duration) -> bool {
        false
    }
}

fn sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!("could not read {}: {error}", path.display());
    });
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn component_name(value: &str) -> &str {
    assert!(
        value.starts_with("model-")
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-'),
        "component must be one model-<slot> identifier"
    );
    value
}

fn config_text(root: &Path, component: &str) -> String {
    let component = component_name(component);
    format!(
        r#"trusted_keys_dir = "{keys}"
hw_rev = 1
state_dir = "{state}"

[component.{component}]
install_dir = "{install}"
keep_previous = 3
source = {{ type = "local_dir", path = "{releases}" }}
on_apply = {{ action = "none" }}
health = {{ probe = "socket", path = "{socket}", timeout = "2s" }}
"#,
        keys = root.join("keys").display(),
        state = root.join("state").display(),
        install = root.join("install").display(),
        releases = root.join("releases").display(),
        socket = root.join("audit-only-robotd.sock").display(),
        component = component,
    )
}

fn engine(root: &Path, audit: GateAudit) -> Engine {
    let text = std::fs::read_to_string(root.join("updater.toml"))
        .expect("persisted updater config must exist");
    let config = Config::from_toml(&text).expect("Policy Patch Lab config must be valid");
    let keys = KeyRing::load(&config.trusted_keys_dir, config.allow_dev_keys)
        .expect("Policy Patch Lab trust anchor must load");
    Engine::new(config, keys, Box::new(audit), Faults::none())
        .expect("updater engine must initialize")
        .without_deferred_restarts()
}

async fn apply_exact(
    engine: &mut Engine,
    component: &str,
    version: &str,
) -> (ApplyResult, Vec<String>) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let result = engine
        .apply(
            component,
            Target::Exact(semver::Version::parse(version).expect("fixed valid version")),
            ApplyOptions::default(),
            tx,
        )
        .await
        .unwrap_or_else(|error| panic!("applying {version} failed: {error}"));
    let mut phases = Vec::new();
    while let Ok(progress) = rx.try_recv() {
        phases.push(format!("{:?}", progress.phase));
    }
    (result, phases)
}

fn current_policy(root: &Path) -> PathBuf {
    root.join("install/current/policy.onnx")
}

fn write_json(path: &Path, payload: &Value) {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(payload).expect("audit JSON must serialize"),
    )
    .unwrap_or_else(|error| panic!("could not write {}: {error}", path.display()));
}

async fn activate(root: PathBuf, source: PathBuf, adapted: PathBuf, component: String) {
    let component = component_name(&component).to_owned();
    if root.exists()
        && std::fs::read_dir(&root)
            .expect("activation root must be readable")
            .next()
            .is_some()
    {
        panic!(
            "activation root must be absent or empty: {}",
            root.display()
        );
    }
    std::fs::create_dir_all(&root).expect("activation root must be creatable");
    let source_bytes = std::fs::read(&source).expect("source policy must be readable");
    let adapted_bytes = std::fs::read(&adapted).expect("adapted policy must be readable");
    let source_sha = sha256(&source);
    let adapted_sha = sha256(&adapted);

    let publisher = Publisher::new(root.join("keys"), root.join("releases"));
    for (version, bytes, revision) in [
        ("0.9.0", source_bytes.as_slice(), source_sha.as_str()),
        ("1.0.0", adapted_bytes.as_slice(), adapted_sha.as_str()),
    ] {
        publisher
            .release(version)
            .channel(&component)
            .file("policy.onnx", bytes, 0o644)
            .manifest(|manifest| {
                manifest["model_api"] = json!(1);
                manifest["source_revision"] = json!(revision);
            })
            .write();
    }
    std::fs::write(root.join("updater.toml"), config_text(&root, &component))
        .expect("updater config must be writable");
    write_json(
        &root.join("lab-metadata.json"),
        &json!({
            "schema": "eggroll-autopatch-updater-input-v2",
            "component": component,
            "source_sha256": source_sha,
            "adapted_sha256": adapted_sha,
        }),
    );

    let audit = GateAudit::default();
    let mut engine = engine(&root, audit.clone());
    let (source_result, source_phases) = apply_exact(&mut engine, &component, "0.9.0").await;
    let source_active_sha = sha256(&current_policy(&root));
    assert_eq!(
        source_active_sha, source_sha,
        "source activation changed bytes"
    );
    let (adapted_result, adapted_phases) = apply_exact(&mut engine, &component, "1.0.0").await;
    let adapted_active_sha = sha256(&current_policy(&root));
    assert_eq!(
        adapted_active_sha, adapted_sha,
        "adapted activation changed bytes"
    );

    let payload = json!({
        "schema": "eggroll-autopatch-updater-activation-v2",
        "component": component,
        "signature_and_artifact_verification": "passed_by_real_engine",
        "source": {
            "apply_result": source_result,
            "phases": source_phases,
            "active_sha256": source_active_sha,
        },
        "adapted": {
            "apply_result": adapted_result,
            "phases": adapted_phases,
            "active_sha256": adapted_active_sha,
        },
        "health_gate_calls": audit.health_calls.load(Ordering::SeqCst),
        "model_api_calls": audit.model_api_calls.load(Ordering::SeqCst),
        "current_policy": current_policy(&root),
    });
    write_json(&root.join("activation.json"), &payload);
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

async fn rollback(root: PathBuf) {
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(root.join("lab-metadata.json"))
            .expect("activation metadata must exist before rollback"),
    )
    .expect("activation metadata must be valid JSON");
    let expected = metadata["source_sha256"]
        .as_str()
        .expect("source SHA must be present");
    let component = component_name(
        metadata["component"]
            .as_str()
            .expect("component must be present"),
    );
    let before = sha256(&current_policy(&root));
    let audit = GateAudit::default();
    let mut engine = engine(&root, audit.clone());
    let result = engine
        .rollback(component)
        .await
        .unwrap_or_else(|error| panic!("rollback failed: {error}"));
    let after = sha256(&current_policy(&root));
    assert_eq!(
        after, expected,
        "rollback did not restore exact source bytes"
    );
    let payload = json!({
        "schema": "eggroll-autopatch-updater-rollback-v2",
        "component": component,
        "rollback_result": result,
        "before_sha256": before,
        "after_sha256": after,
        "expected_source_sha256": expected,
        "exact_source_restored": true,
        "health_gate_calls": audit.health_calls.load(Ordering::SeqCst),
        "current_policy": current_policy(&root),
    });
    write_json(&root.join("rollback.json"), &payload);
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

#[tokio::main]
async fn main() {
    match Cli::parse().command {
        Command::Activate {
            root,
            source,
            adapted,
            component,
        } => activate(root, source, adapted, component).await,
        Command::Rollback { root } => rollback(root).await,
    }
}
