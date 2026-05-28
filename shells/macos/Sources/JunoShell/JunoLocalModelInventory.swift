import Foundation

enum JunoLocalModelInventory {
    struct Inventory {
        let previewModelOnDisk: Bool
        let finalModelOnDisk: Bool
    }

    static func snapshot() -> Inventory {
        let fm = FileManager.default
        let roots = candidateRoots()
        var preview = false
        var final = false
        for root in roots {
            if !preview, hasContent(at: root.appendingPathComponent("models/preview"), fm: fm) {
                preview = true
            }
            if !final, hasContent(at: root.appendingPathComponent("models/final"), fm: fm) {
                final = true
            }
            if preview && final { break }
        }
        return Inventory(previewModelOnDisk: preview, finalModelOnDisk: final)
    }

    private static func candidateRoots() -> [URL] {
        var roots: [URL] = []
        if let repo = JunoRepoPaths.guessRepoRoot() {
            roots.append(URL(fileURLWithPath: repo).appendingPathComponent(".juno_v2_demo"))
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        roots.append(cwd.appendingPathComponent(".juno_v2_demo"))
        if let home = ProcessInfo.processInfo.environment["HOME"] {
            roots.append(URL(fileURLWithPath: home).appendingPathComponent(".juno_v2_demo"))
        }
        return roots
    }

    private static func hasContent(at url: URL, fm: FileManager) -> Bool {
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { return false }
        if !isDir.boolValue { return true }
        let contents = (try? fm.contentsOfDirectory(atPath: url.path)) ?? []
        return contents.contains { !$0.hasPrefix(".") }
    }
}
