import Foundation

/// User-facing app identity strings.
enum JunoProductIdentity {
    /// Name shown in Finder / menu bar (from bundle).
    static var shortName: String {
        if let s = Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String,
           !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return s
        }
        if let s = Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String,
           !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return s
        }
        return "Juno"
    }

    /// Used in copy for System Settings / `tccutil` (must match app bundle id).
    static var bundleIdentifier: String {
        Bundle.main.bundleIdentifier ?? "com.juno.shell"
    }

    /// Marketing version only — shown in the sidebar footer, Privacy page,
    /// About panel. Build number is hidden because the default value (`1`)
    /// reads as a bug to users.
    static var versionSummary: String {
        let v = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.0.0"
        return v
    }

    /// Verbose form with build number — for diagnostics (support bundle, a
    /// future debug About row). Not for user-facing chrome.
    static var versionDetail: String {
        let v = versionSummary
        let build = (Bundle.main.infoDictionary?["CFBundleVersion"] as? String) ?? ""
        let trimmed = build.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return v }
        return "\(v) (\(trimmed))"
    }
}
