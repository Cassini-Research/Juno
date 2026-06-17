import AppKit
import Foundation

// MARK: - App location / translocation guard
//
// When a user opens Juno.app directly from the mounted DMG or any quarantined
// folder (instead of dragging it to /Applications first), macOS runs it under
// App Translocation: the bundle executes from a randomized, read-only path
// like `/private/var/folders/…/AppTranslocation/<UUID>/d/Juno.app`. That path
// changes on every launch, so any TCC grant the user makes (Microphone,
// Accessibility, Input Monitoring) is keyed to a location that no longer exists
// next time — permissions appear to "reset" and the dictation hotkey silently
// stops receiving key events. This is a leading cause of fresh-install
// "permissions keep resetting / I have to grant twice / press twice" reports.
//
// We detect the condition and tell the user to move Juno to Applications. We do
// NOT auto-move the bundle (moving a running, translocated app from itself is
// error-prone); the DMG ships an /Applications symlink for drag-install and
// this guidance closes the loop.
enum JunoAppLocation {
    /// True when the running bundle is executing from an App Translocation
    /// (Gatekeeper randomized) path rather than its real on-disk location.
    static var isTranslocated: Bool {
        Bundle.main.bundlePath.contains("/AppTranslocation/")
    }

    /// True when the bundle lives in a system or user Applications folder,
    /// where its path is stable across launches (so TCC grants persist).
    static var isInApplicationsFolder: Bool {
        let path = Bundle.main.bundlePath
        return path.hasPrefix("/Applications/")
            || path.hasPrefix("\(NSHomeDirectory())/Applications/")
    }

    /// Log the launch location once at startup so a "permissions reset" report
    /// can be diagnosed from the logs alone.
    static func logLaunchLocation() {
        NSLog("Juno: launch location bundlePath=%@ translocated=%@ inApplications=%@",
              Bundle.main.bundlePath,
              isTranslocated ? "true" : "false",
              isInApplicationsFolder ? "true" : "false")
    }

    /// If we're running translocated, show a blocking alert guiding the user to
    /// move Juno to Applications. Returns true if the warning was shown.
    @discardableResult
    @MainActor
    static func warnIfTranslocated() -> Bool {
        guard isTranslocated else { return false }
        NSLog("Juno: WARNING running translocated — TCC permissions will not persist; prompting move to Applications")
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Move Juno to your Applications folder"
        alert.informativeText = """
        Juno is running from a temporary location, so macOS resets its \
        permissions every time you open it — that's why the microphone or \
        dictation hotkey may stop working.

        Quit Juno, drag it into your Applications folder, then open it from \
        there once. You'll only need to do this once.
        """
        alert.addButton(withTitle: "Quit Juno")
        alert.addButton(withTitle: "Continue Anyway")
        NSApplication.shared.activate(ignoringOtherApps: true)
        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            NSLog("Juno: user chose Quit from translocation warning")
            NSApplication.shared.terminate(nil)
        } else {
            NSLog("Juno: user chose Continue Anyway from translocation warning")
        }
        return true
    }
}
