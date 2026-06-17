import CoreGraphics
import Foundation

/// Single gate for visible-screen text context.
///
/// The rule is intentionally strict: dictation paths may consume screen terms
/// only when the user has opted in and macOS access is already granted.
enum JunoScreenContextAccess {
    private static let settingsOpenCooldown: TimeInterval = 1.25
    private static var lastSettingsOpenAt: Date?

    static var isEnabled: Bool {
        JunoUserDefaults.screenContextEnabled
    }

    static var permissionGranted: Bool {
        CGPreflightScreenCaptureAccess()
    }

    static var isEnabledAndGranted: Bool {
        isEnabled && permissionGranted
    }

    static func requestFromExplicitUserAction(completion: @escaping (Bool) -> Void) {
        let request = {
            JunoUserDefaults.screenContextEnabled = true

            if permissionGranted {
                completion(true)
                return
            }

            // CGRequestScreenCaptureAccess() is the macOS-supported way to
            // REGISTER Juno in System Settings → Privacy → Screen Recording
            // (preflight only checks existing approval; it never creates the
            // row). It is idempotent and must be called on EVERY explicit
            // request — NOT gated behind a persisted flag.
            //
            // Why this matters for reinstalls AND upgrades: the old code gated
            // the request behind a sticky Bool in UserDefaults, which survives
            // both. A freshly-installed/upgraded build is a (potentially) new
            // code identity macOS hasn't registered, so the stale "already
            // requested" flag made us skip the request entirely — the user then
            // opened System Settings and Juno wasn't in the list at all (the
            // reported bug). We now (1) always call the request so each build
            // re-registers itself, and (2) tie "already prompted" to the
            // current BUILD, so an upgrade re-shows the one-time consent dialog
            // cleanly (when the grant didn't carry over) rather than jumping to
            // a Settings list the new build isn't in yet.
            let currentBuild = (Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String) ?? ""
            let alreadyPromptedThisBuild = !currentBuild.isEmpty
                && JunoUserDefaults.screenRecordingPromptRequestedBuild == currentBuild
            JunoUserDefaults.screenRecordingPromptRequestedBuild = currentBuild
            // Keep the legacy Bool updated too (some older support tooling reads it).
            JunoUserDefaults.screenRecordingPromptRequested = true
            _ = CGRequestScreenCaptureAccess()

            if permissionGranted {
                completion(true)
                return
            }

            // First request for THIS build shows the one-time consent dialog,
            // which carries its own "Open System Settings" button — don't stack
            // a second navigation on top of it. A repeat request for the same
            // build is a silent no-op, so take the user to Settings, where Juno
            // is now listed because the call above registered it.
            if alreadyPromptedThisBuild {
                openSystemSettings()
            }
            completion(permissionGranted)
        }
        if Thread.isMainThread {
            request()
        } else {
            DispatchQueue.main.async(execute: request)
        }
    }

    static func openSystemSettings() {
        let open = {
            let now = Date()
            if let lastSettingsOpenAt, now.timeIntervalSince(lastSettingsOpenAt) < settingsOpenCooldown {
                return
            }
            lastSettingsOpenAt = now
            JunoSystemSettingsLinks.openScreenRecordingPrivacy()
        }
        if Thread.isMainThread {
            open()
        } else {
            DispatchQueue.main.async(execute: open)
        }
    }
}
