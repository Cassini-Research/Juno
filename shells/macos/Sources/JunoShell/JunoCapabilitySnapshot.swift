import Foundation

/// Runs ``juno-capability`` and returns the JSON dictionary (empty on failure).
enum JunoCapabilitySnapshot {
    static func capture() -> [String: Any] {
        guard let bin = HelperBinary.path("juno-capability") else {
            return JunoLocalCapability.snapshot()
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: bin)
        let out = Pipe()
        task.standardOutput = out
        task.standardError = Pipe()
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            return JunoLocalCapability.snapshot()
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return JunoLocalCapability.snapshot()
        }
        if (obj["has_ax_trust"] as? Bool) == false {
            let local = JunoLocalCapability.snapshot()
            if (local["has_ax_trust"] as? Bool) == true {
                return local
            }
        }
        return obj
    }
}
