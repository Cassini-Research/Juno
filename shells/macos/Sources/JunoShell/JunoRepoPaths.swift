import Foundation

/// Best-effort discovery of the repository root from a running binary.
/// Only used by developer-facing surfaces (model inventory, setup helper,
/// app log paths). The end-user help window does NOT consume this — see
/// fix #6.
enum JunoRepoPaths {
    /// Walks up from the running executable looking for the project root marker.
    static func guessRepoRoot() -> String? {
        let exe = Bundle.main.executablePath ?? CommandLine.arguments[0]
        var url = URL(fileURLWithPath: exe)
        for _ in 0..<16 {
            url.deleteLastPathComponent()
            let marker = url.appendingPathComponent("pyproject.toml")
            if FileManager.default.fileExists(atPath: marker.path) {
                return url.path
            }
        }
        return nil
    }
}
