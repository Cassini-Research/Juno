import Foundation

/// One-time copy of preferences from the previous bundle id (`com.juno.shell`) for upgrades (shortcut, name, etc.).
/// Onboarding completion and brand-delight flags are **not** imported: ``JunoOnboardingCompleted`` is set only
/// when the user finishes the in-app flow (``JunoOnboarding``), not from Juno-era defaults.
enum JunoLegacyDefaultsMigration {
    private static let markerKey = "JunoMigratedPreferencesFromComJunoShell_v1"

    static func runOnce() {
        guard !UserDefaults.standard.bool(forKey: markerKey) else { return }
        guard let legacy = UserDefaults(suiteName: "com.juno.shell") else {
            UserDefaults.standard.set(true, forKey: markerKey)
            return
        }
        let ud = UserDefaults.standard
        func copyIfMissing(_ legacyKey: String, _ newKey: String) {
            guard ud.object(forKey: newKey) == nil else { return }
            guard let v = legacy.object(forKey: legacyKey) else { return }
            ud.set(v, forKey: newKey)
        }
        copyIfMissing("JunoPreferredDisplayName", JunoUserDefaults.preferredDisplayNameKey)
        copyIfMissing("JunoShortcutKey", "JunoShortcutKey")
        copyIfMissing("JunoShellLifetimeWordCount", "JunoShellLifetimeWordCount")
        copyIfMissing("JunoCustomHotkeyPlistData", "JunoCustomHotkeyPlistData")
        UserDefaults.standard.set(true, forKey: markerKey)
        ud.synchronize()
    }
}
