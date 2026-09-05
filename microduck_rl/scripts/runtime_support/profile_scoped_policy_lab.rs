//! Prove profile-scoped routing through the real signed updater engine.

use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use clap::Parser;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use test_support::Publisher;
use updater::config::Config;
use updater::engine::{ApplyOptions, Engine};
use updater::faults::Faults;
use updater::profile_scoped_model_activation::{
    Decision, PRODUCTION_PATH, profile_scoped_model_activation,
};
use updater::proto::{ApplyResult, Target};
use updater::robot::{Health, RobotClient, SafeToRestart};
use updater::verify::KeyRing;

#[derive(Parser)]
#[command(about = "Profile-scoped signed updater proof for EGGROLL Autopatch")]
struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    adapted: PathBuf,
    #[arg(long)]
    component: String,
    #[arg(long)]
    artifact_id: String,
    #[arg(long)]
    activation_profile_sha256: String,
    #[arg(long)]
    release_scope_sha256: String,
    #[arg(long)]
    output: PathBuf,
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
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn require_sha256(name: &str, value: &str) {
    assert!(
        value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()),
        "{name} must be one SHA-256"
    );
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
    )
}

fn engine(root: &Path, audit: GateAudit) -> Engine {
    let text = std::fs::read_to_string(root.join("updater.toml"))
        .expect("persisted updater config must exist");
    let config = Config::from_toml(&text).expect("profile proof config must be valid");
    let keys = KeyRing::load(&config.trusted_keys_dir, config.allow_dev_keys)
        .expect("profile proof trust anchor must load");
    Engine::new(config, keys, Box::new(audit), Faults::none())
        .expect("updater engine must initialize")
        .without_deferred_restarts()
}

async fn apply_exact(engine: &mut Engine, component: &str, version: &str) -> ApplyResult {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    engine
        .apply(
            component,
            Target::Exact(semver::Version::parse(version).expect("fixed valid version")),
            ApplyOptions::default(),
            tx,
        )
        .await
        .unwrap_or_else(|error| panic!("applying {version} failed: {error}"))
}

fn prepare_root(
    root: &Path,
    source_bytes: &[u8],
    adapted_bytes: &[u8],
    source_sha: &str,
    adapted_sha: &str,
    component: &str,
) {
    std::fs::create_dir_all(root).expect("route root must be creatable");
    let publisher = Publisher::new(root.join("keys"), root.join("releases"));
    for (version, bytes, revision) in [
        ("0.9.0", source_bytes, source_sha),
        ("1.0.0", adapted_bytes, adapted_sha),
    ] {
        publisher
            .release(version)
            .channel(component)
            .file("policy.onnx", bytes, 0o644)
            .manifest(|manifest| {
                manifest["model_api"] = json!(1);
                manifest["source_revision"] = json!(revision);
            })
            .write();
    }
    std::fs::write(root.join("updater.toml"), config_text(root, component))
        .expect("updater config must be writable");
}

fn current_policy(root: &Path) -> PathBuf {
    root.join("install/current/policy.onnx")
}

async fn prove_route(
    root: &Path,
    observed_profile: &str,
    activation_profile: &str,
    source_bytes: &[u8],
    adapted_bytes: &[u8],
    source_sha: &str,
    adapted_sha: &str,
    component: &str,
) -> Value {
    prepare_root(
        root,
        source_bytes,
        adapted_bytes,
        source_sha,
        adapted_sha,
        component,
    );
    let audit = GateAudit::default();
    let mut engine = engine(root, audit.clone());
    let source_apply = apply_exact(&mut engine, component, "0.9.0").await;
    assert_eq!(sha256(&current_policy(root)), source_sha);
    let decision = profile_scoped_model_activation(observed_profile, activation_profile);
    let adapted_apply = if decision == Decision::ActivateAdapted {
        Some(apply_exact(&mut engine, component, "1.0.0").await)
    } else {
        None
    };
    let selected_sha = sha256(&current_policy(root));
    let expected = match decision {
        Decision::ActivateAdapted => adapted_sha,
        Decision::RetainSource => source_sha,
    };
    assert_eq!(selected_sha, expected);
    json!({
        "profile_sha256": observed_profile,
        "decision": decision.as_str(),
        "selected_policy_sha256": selected_sha,
        "source_apply_result": source_apply,
        "adapted_apply_result": adapted_apply,
        "signature_and_artifact_verification": "passed_by_real_engine",
        "health_gate_calls": audit.health_calls.load(Ordering::SeqCst),
        "model_api_calls": audit.model_api_calls.load(Ordering::SeqCst),
    })
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    require_sha256("activation profile", &args.activation_profile_sha256);
    require_sha256("release scope", &args.release_scope_sha256);
    let component = component_name(&args.component);
    assert!(
        !args.root.exists(),
        "proof root must not exist: {}",
        args.root.display()
    );
    let source_bytes = std::fs::read(&args.source).expect("source policy must be readable");
    let adapted_bytes = std::fs::read(&args.adapted).expect("adapted policy must be readable");
    let source_sha = sha256(&args.source);
    let adapted_sha = sha256(&args.adapted);
    let matched = prove_route(
        &args.root.join("matched"),
        &args.activation_profile_sha256,
        &args.activation_profile_sha256,
        &source_bytes,
        &adapted_bytes,
        &source_sha,
        &adapted_sha,
        component,
    )
    .await;
    let unknown = prove_route(
        &args.root.join("unknown"),
        "unknown",
        &args.activation_profile_sha256,
        &source_bytes,
        &adapted_bytes,
        &source_sha,
        &adapted_sha,
        component,
    )
    .await;
    let payload = json!({
        "schema": "eggroll-autopatch-routing-attestation-v1",
        "status": "pass",
        "artifact_id": args.artifact_id,
        "adapted_sha256": adapted_sha,
        "release_scope_sha256": args.release_scope_sha256,
        "source_fallback_sha256": source_sha,
        "unknown_profile_action": "retain_source",
        "production_path": PRODUCTION_PATH,
        "routes": [matched, unknown],
        "signed_updater_proof": {
            "status": "pass",
            "engine": "updater::engine::Engine::apply",
            "profile_router": "updater::profile_scoped_model_activation",
            "separate_fresh_roots": true,
        },
        "claim_scope": "signed updater integration proof; no physical robot",
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).expect("output parent must be creatable");
    }
    std::fs::write(
        &args.output,
        serde_json::to_vec_pretty(&payload).expect("routing proof must serialize"),
    )
    .expect("routing proof must be writable");
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}
