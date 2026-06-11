import CoreGraphics
import Foundation

/// Single gate for visible-screen text context.
///
/// The rule is intentionally strict: dictation paths may consume screen terms
/// only when the user has opted in and macOS access is already granted.
enum JunoScreenContextAccess {
    private static let settingsOpenCooldown: TimeInterval = 1.25
    private static var lastSettingsOpenAt: Date?
    private static var requestIssuedThisSession = false

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

            // Preflight only checks existing approval; it does not create the
            // row in System Settings. The request call is the macOS-supported
            // way to register Juno for Screen Recording, so keep it behind one
            // explicit user action and never call it from dictation/runtime paths.
            if !requestIssuedThisSession {
                requestIssuedThisSession = true
                _ = CGRequestScreenCaptureAccess()
            }

            let granted = permissionGranted
            if !granted {
                openSystemSettings()
            }
            completion(granted)
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
