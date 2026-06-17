import AppKit
import Foundation

// MARK: - App location / first-run install guard
//
// When a user opens Juno.app directly from the mounted DMG (instead of first
// dragging it to /Applications), macOS runs it under App Translocation: the
// bundle executes from a randomized, read-only path like
// `/private/var/folders/…/AppTranslocation/<UUID>/d/Juno.app`. That path
// changes on every launch, so any TCC grant the user makes (Microphone,
// Accessibility, Input Monitoring) is keyed to a location that no longer exists
// next time — permissions appear to "reset" and the dictation hotkey silently
// stops receiving key events. This is a leading cause of fresh-install
// "permissions keep resetting / I have to grant twice / press twice" reports.
//
// To make first run "just work", when we detect a translocated launch we offer
// to install Juno into /Applications and relaunch from there — so the very
// first open from the DMG ends as a proper, stable, auto-started install.
enum JunoAppLocation {
    /// True when the running bundle is executing from an App Translocation
    /// randomized translocation path rather than its real on-disk location.
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

    /// If running translocated (opened from the DMG / a quarantined folder),
    /// offer to install into /Applications and relaunch from there. Returns
    /// true if the prompt was shown.
    @discardableResult
    @MainActor
    static func offerInstallToApplicationsIfNeeded() -> Bool {
        guard isTranslocated else { return false }
        NSLog("Juno: running translocated — offering install to /Applications")
        let alert = NSAlert()
        alert.alertStyle = .informational
        alert.messageText = "Move Juno to your Applications folder?"
        alert.informativeText = """
        Juno is running from a temporary location. Installing it into your \
        Applications folder lets it keep its microphone and dictation \
        permissions and start up properly. Juno will move itself and reopen — \
        you only need to do this once.
        """
        alert.addButton(withTitle: "Move to Applications & Relaunch")
        alert.addButton(withTitle: "Not Now")
        NSApplication.shared.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn {
            NSLog("Juno: user accepted move-to-Applications")
            performMoveToApplicationsAndRelaunch()
        } else {
            NSLog("Juno: user declined move-to-Applications (continuing translocated)")
        }
        return true
    }

    // MARK: - Move + relaunch

    @MainActor
    private static func performMoveToApplicationsAndRelaunch() {
        let src = Bundle.main.bundleURL
        let appName = src.lastPathComponent  // "Juno.app"
        // The copy of a large bundle blocks; run it off the main thread, then
        // relaunch on the main thread.
        DispatchQueue.global(qos: .userInitiated).async {
            let dest = installDestination(appName: appName)
            let fm = FileManager.default
            do {
                if fm.fileExists(atPath: dest.path) {
                    // Replacing an older install. We are NOT running from this
                    // path (we're translocated), so removing it is safe.
                    try fm.removeItem(at: dest)
                }
                try fm.createDirectory(at: dest.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                try fm.copyItem(at: src, to: dest)
                stripQuarantine(at: dest)
                NSLog("Juno: installed to %@ — relaunching", dest.path)
                DispatchQueue.main.async { relaunch(at: dest) }
            } catch {
                NSLog("Juno: move-to-Applications failed: %@", error.localizedDescription)
                DispatchQueue.main.async {
                    let a = NSAlert()
                    a.alertStyle = .warning
                    a.messageText = "Couldn't move Juno automatically"
                    a.informativeText = "Please quit Juno and drag it into your "
                        + "Applications folder manually, then open it from there."
                    a.addButton(withTitle: "OK")
                    a.runModal()
                }
            }
        }
    }

    /// Prefer /Applications; fall back to the user's ~/Applications when
    /// /Applications isn't writable (avoids an admin-auth prompt while still
    /// landing in a stable, non-translocated path).
    private static func installDestination(appName: String) -> URL {
        let systemApps = URL(fileURLWithPath: "/Applications", isDirectory: true)
        if FileManager.default.isWritableFile(atPath: systemApps.path) {
            return systemApps.appendingPathComponent(appName)
        }
        let userApps = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Applications", isDirectory: true)
        return userApps.appendingPathComponent(appName)
    }

    /// Remove the quarantine flag from the installed copy so macOS does not
    /// translocate it again on next launch.
    private static func stripQuarantine(at url: URL) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/xattr")
        p.arguments = ["-dr", "com.apple.quarantine", url.path]
        try? p.run()
        p.waitUntilExit()
    }

    @MainActor
    private static func relaunch(at url: URL) {
        let config = NSWorkspace.OpenConfiguration()
        config.createsNewApplicationInstance = true
        config.activates = true
        NSWorkspace.shared.openApplication(at: url, configuration: config) { _, err in
            DispatchQueue.main.async {
                if let err {
                    NSLog("Juno: relaunch from %@ failed: %@", url.path, err.localizedDescription)
                }
                NSApplication.shared.terminate(nil)
            }
        }
    }
}
