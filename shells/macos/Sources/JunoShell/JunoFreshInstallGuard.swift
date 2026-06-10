// JunoFreshInstallGuard.swift
//
// Self-healing detection for "TCC was reset / fresh install / re-install"
// scenarios where the user's preference plist still contains
// ``JunoOnboardingCompleted = true`` but the system-level permission
// rows that the onboarding flow walked them through have been wiped.
//
// Without this check, a user who:
//   1. completed onboarding once (mic + accessibility granted),
//   2. then ran a local reset/install flow with the
//      ``--keep-user-data`` flag (or otherwise had TCC reset by the OS,
//      Migration Assistant, ``tccutil reset All``, etc.),
// would land on **Home** with the "Permissions needed" pill and
// inline "Allow microphone / Open Accessibility" cards — bypassing
// the proper onboarding flow that explains *why* each grant matters.
// That's a confusing half-state we want to avoid.
//
// **Heuristic**: if onboarding is marked completed but the
// **Microphone** authorization status is ``notDetermined`` or Accessibility is
// not trusted, we've been re-installed or had TCC wiped/mismatched. AX has no
// reliable "notDetermined vs revoked" API, and Juno cannot paste without it,
// so we send the user back through setup instead of showing Home as ready while
// the actual paste path is blocked.
//
// On detection we:
//   - clear ``JunoOnboardingCompleted`` so the welcome flow runs,
//   - clear the first-run brand-delight flag so the user gets the full
//     welcome pop again,
//   - clear the Voice Actions cold-nudge cooldown so the Reminders ask
//     re-surfaces after onboarding,
//   - turn the Voice Actions master toggle OFF (it was opt-in; that
//     opt-in was tied to the prior install's TCC state).
//
// **Hard rule**: this runs at app launch *before* anything else reads
// ``onboardingCompleted``. Updates with intact TCC are a no-op — the
// fingerprint check passes immediately.

import ApplicationServices
import AVFoundation
import Foundation

enum JunoFreshInstallGuard {

    /// Run once during ``JunoShell.init`` (before legacy-defaults
    /// migration so resets on this path don't clobber freshly migrated
    /// values). Idempotent and side-effect-free in the steady state.
    static func runOnce() {
        let ud = UserDefaults.standard
        let onboarded = ud.bool(forKey: JunoUserDefaults.onboardingCompletedKey)
        guard onboarded else {
            // Already in onboarding; nothing to recover from.
            return
        }
        let micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        let axTrusted = AXIsProcessTrusted()
        guard micStatus == .notDetermined || !axTrusted else {
            // Core grants intact, normal launch.
            return
        }

        NSLog(
            "Juno: fresh-install/TCC-reset detected — onboardingCompleted=true but mic=%@ ax=%@. Resetting onboarding state so the user sees the welcome flow.",
            String(describing: micStatus),
            axTrusted ? "trusted" : "missing"
        )
        // Reset every flag that says "user already saw the onboarding
        // story." TCC itself is *not* touched here — the OS owns those
        // rows and the install script (or System Settings) is the
        // canonical place to reset them.
        JunoUserDefaults.resetOnboardingForRetest()

        // Voice Actions: opt-in was tied to the previous install's TCC.
        // Reset to defaults so the user opts in again with the new
        // grants. Per-kind nudge cooldowns and the legacy single-key
        // cooldown are both cleared so the Home priority card re-appears
        // for every action after onboarding completes.
        // Default ON: with the silent-failure path replaced by a toast that
        // explains exactly why an action did not run, leaving Voice Actions
        // off by default just produces dead-silent dictation when the user
        // says "Juno, take a note…". The blocked path still triggers the
        // first-launch permission flow; nothing bypasses macOS prompts.
        ud.set(true, forKey: JunoUserDefaults.actionsEnabledKey)
        ud.removeObject(forKey: "JunoRemindersNudgeColdDismissedAt")
        JunoUserDefaults.clearAllActionsNudgeDismissals()
        ud.synchronize()
    }
}
