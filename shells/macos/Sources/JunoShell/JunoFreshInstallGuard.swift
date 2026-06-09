// JunoFreshInstallGuard.swift
//
// App-launch repair for onboarding state.
//
// Juno used to rerun the full welcome flow whenever onboarding was completed
// but core TCC permissions were missing. That is too aggressive for updates:
// local ad-hoc builds and some macOS upgrade paths can invalidate TCC rows
// without making the user a new customer again.
//
// The guard now only resets onboarding when Juno explicitly increments
// ``currentOnboardingRequirementsVersion``. Missing permissions after a normal
// update are repaired by the Home and Settings permission surfaces.
//
// **Hard rule**: this runs at app launch before anything else reads
// ``onboardingCompleted``.

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

        let currentRequirements = JunoUserDefaults.currentOnboardingRequirementsVersion
        let completedRequirements = JunoUserDefaults.onboardingRequirementsVersion
        if completedRequirements <= 0 {
            // Existing installs predate the requirements marker. Preserve their
            // completed onboarding and mark them current; future requirement
            // bumps can still intentionally rerun setup.
            JunoUserDefaults.onboardingRequirementsVersion = currentRequirements
        } else if completedRequirements < currentRequirements {
            NSLog(
                "Juno: onboarding requirements changed from %d to %d. Resetting onboarding state.",
                completedRequirements,
                currentRequirements
            )
            JunoUserDefaults.resetOnboardingForRetest()

            // Voice Actions: opt-in was tied to the previous onboarding
            // requirements. Reset to defaults so the user makes a fresh choice
            // during the new required setup.
            ud.removeObject(forKey: JunoUserDefaults.actionsEnabledKey)
            ud.removeObject(forKey: "JunoRemindersNudgeColdDismissedAt")
            JunoUserDefaults.clearAllActionsNudgeDismissals()
            ud.synchronize()
            return
        }

        let micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        let axTrusted = AXIsProcessTrusted()
        if micStatus == .notDetermined || !axTrusted {
            NSLog(
                "Juno: onboarding remains completed but permissions need repair after launch/update: mic=%@ ax=%@.",
                String(describing: micStatus),
                axTrusted ? "trusted" : "missing"
            )
        }
    }
}
