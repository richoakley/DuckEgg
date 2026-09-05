//! Fail-closed policy selection used by the profile-scoped updater proof.

/// Evidence names this path so an attestation cannot silently substitute a
/// campaign-side Python decision for the updater's routing decision.
pub const PRODUCTION_PATH: &str = "updaterd::profile_scoped_model_activation";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Decision {
    ActivateAdapted,
    RetainSource,
}

impl Decision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ActivateAdapted => "activate_adapted",
            Self::RetainSource => "retain_source",
        }
    }
}

/// Select adapted bytes only for the exact attested profile hash.
///
/// Unknown and unattested profiles always retain the declared source.  The
/// caller still passes the selected version through the ordinary signed,
/// health-gated updater engine.
pub fn profile_scoped_model_activation(
    observed_profile_sha256: &str,
    activation_profile_sha256: &str,
) -> Decision {
    if observed_profile_sha256 == activation_profile_sha256 {
        Decision::ActivateAdapted
    } else {
        Decision::RetainSource
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, profile_scoped_model_activation};

    #[test]
    fn exact_profile_activates_and_every_other_value_retains_source() {
        let profile = "a".repeat(64);
        assert_eq!(
            profile_scoped_model_activation(&profile, &profile),
            Decision::ActivateAdapted
        );
        assert_eq!(
            profile_scoped_model_activation(&"b".repeat(64), &profile),
            Decision::RetainSource
        );
        assert_eq!(
            profile_scoped_model_activation("unknown", &profile),
            Decision::RetainSource
        );
    }
}
