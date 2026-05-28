import AppKit
import AVFoundation
import Foundation
import Speech

/// Opens Privacy & Security sub-panes in **System Settings**.
///
/// Strategy:
/// 1. Prefer legacy ``com.apple.preference.security?Privacy_*`` URLs first — they still
///    route most reliably on many macOS 14/15 builds.
/// 2. Then try the newer ``com.apple.settings.PrivacySecurity`` (+ ``.extension``) forms.
/// 3. Fall back to ``/usr/bin/open`` with the same strings (sometimes behaves differently
///    from ``NSWorkspace``).
/// 4. Finally launch the System Settings app so the user can open **Privacy & Security** manually.
///
/// For **Microphone** / **Speech recognition**, when access is denied we call the matching
/// ``request*`` API once before opening Settings so this **bundle path** is registered with
/// TCC — otherwise the Microphone list can look empty until a prompt has been attempted.
enum JunoSystemSettingsLinks {
    private static func openFirstWorking(_ urls: [URL]) -> URL? {
        assert(Thread.isMainThread)
        for url in urls where NSWorkspace.shared.open(url) {
            return url
        }
        return nil
    }

    private static func shellOpen(_ url: URL) {
        let t = Process()
        t.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        t.arguments = [url.absoluteString]
        try? t.run()
    }

    private static func urlsForPrivacyAnchor(_ anchor: String) -> [URL] {
        let specs: [String] = [
            "x-apple.systempreferences:com.apple.preference.security?\(anchor)",
            "x-apple.systempreferences:com.apple.settings.PrivacySecurity?\(anchor)",
            "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?\(anchor)",
        ]
        return specs.compactMap { URL(string: $0) }
    }

    private static func trailingPrivacyFallbacks() -> [URL] {
        [
            "x-apple.systempreferences:com.apple.preference.security?Privacy",
            "x-apple.systempreferences:com.apple.preference.security",
        ].compactMap { URL(string: $0) }
    }

    private static func openSystemSettingsApplication() {
        let paths = [
            "/System/Applications/System Settings.app",
            "/System/Applications/System Preferences.app",
        ]
        for path in paths where FileManager.default.fileExists(atPath: path) {
            _ = NSWorkspace.shared.open(URL(fileURLWithPath: path, isDirectory: true))
            return
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        task.arguments = ["-a", "System Settings"]
        try? task.run()
    }

    private static func openPrivacyPane(anchor: String) {
        var urls = urlsForPrivacyAnchor(anchor)
        urls.append(contentsOf: trailingPrivacyFallbacks())

        let work = {
            if openFirstWorking(urls) != nil {
                return
            }
            // `open(1)` from Terminal sometimes succeeds when `NSWorkspace` does not.
            if let first = urls.first {
                shellOpen(first)
            }
            openSystemSettingsApplication()
        }
        if Thread.isMainThread {
            work()
        } else {
            DispatchQueue.main.async(execute: work)
        }
    }

    static func openMicrophonePrivacy() {
        let run = { openPrivacyPane(anchor: "Privacy_Microphone") }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .denied, .restricted:
            AVCaptureDevice.requestAccess(for: .audio) { _ in
                DispatchQueue.main.async(execute: run)
            }
        default:
            if Thread.isMainThread {
                run()
            } else {
                DispatchQueue.main.async(execute: run)
            }
        }
    }

    static func openAccessibilityPrivacy() {
        let run = { openPrivacyPane(anchor: "Privacy_Accessibility") }
        if Thread.isMainThread {
            run()
        } else {
            DispatchQueue.main.async(execute: run)
        }
    }

    static func openAutomationPrivacy() {
        let run = { openPrivacyPane(anchor: "Privacy_Automation") }
        if Thread.isMainThread {
            run()
        } else {
            DispatchQueue.main.async(execute: run)
        }
    }

    static func openRemindersPrivacy() {
        let run = { openPrivacyPane(anchor: "Privacy_Reminders") }
        if Thread.isMainThread {
            run()
        } else {
            DispatchQueue.main.async(execute: run)
        }
    }

    static func openCalendarsPrivacy() {
        let run = { openPrivacyPane(anchor: "Privacy_Calendars") }
        if Thread.isMainThread {
            run()
        } else {
            DispatchQueue.main.async(execute: run)
        }
    }

    static func openSpeechRecognitionPrivacy() {
        let run = { openPrivacyPane(anchor: "Privacy_SpeechRecognition") }
        switch SFSpeechRecognizer.authorizationStatus() {
        case .denied, .restricted:
            SFSpeechRecognizer.requestAuthorization { _ in
                DispatchQueue.main.async(execute: run)
            }
        default:
            if Thread.isMainThread {
                run()
            } else {
                DispatchQueue.main.async(execute: run)
            }
        }
    }
}
