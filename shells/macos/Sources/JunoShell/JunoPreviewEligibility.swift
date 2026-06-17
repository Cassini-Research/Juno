import Darwin
import Foundation

/// Hardware gate for live HUD transcription preview.
///
/// Dictation itself stays available everywhere Juno supports. This gate only
/// controls the optional live-preview ASR path, which can compete with final
/// transcription on smaller Apple Silicon Macs.
enum JunoPreviewEligibility {
    struct Snapshot: Equatable {
        let chipName: String
        let chipGeneration: Int?
        let memoryGB: Int

        var isEligible: Bool {
            guard let chipGeneration else { return false }
            return chipGeneration >= 2 && memoryGB >= 32
        }

        var warningMessage: String? {
            guard isEligible, memoryGB <= 64 else { return nil }
            return "Live preview can hamper transcription performance on \(memoryGB) GB Macs."
        }

        var unavailableMessage: String? {
            if chipGeneration == nil {
                return "Live preview requires Apple Silicon M2 or newer."
            }
            if let chipGeneration, chipGeneration < 2 {
                return "Live preview requires Apple Silicon M2 or newer. This Mac reports \(chipName)."
            }
            if memoryGB < 32 {
                return "Live preview requires at least 32 GB memory. This Mac has \(memoryGB) GB."
            }
            return nil
        }

    }

    static var current: Snapshot {
        cachedSnapshot
    }

    private static let cachedSnapshot: Snapshot = {
        if let forced = forcedSnapshotFromEnvironment() {
            return forced
        }

        let brand = sysctlString("machdep.cpu.brand_string") ?? ""
        let generation = appleChipGeneration(from: brand)
        let chipName = normalizedChipName(brand: brand, generation: generation)
        let memoryGB = normalizedMemoryGB(bytes: ProcessInfo.processInfo.physicalMemory)
        return Snapshot(chipName: chipName, chipGeneration: generation, memoryGB: memoryGB)
    }()

    static func appleChipGeneration(from raw: String) -> Int? {
        let pattern = #"(?i)\bApple\s+M(\d+)\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        guard let match = regex.firstMatch(in: raw, range: range),
              match.numberOfRanges > 1,
              let capture = Range(match.range(at: 1), in: raw)
        else { return nil }
        return Int(raw[capture])
    }

    private static func forcedSnapshotFromEnvironment() -> Snapshot? {
        let env = ProcessInfo.processInfo.environment
        let rawChip = env["JUNO_PREVIEW_ELIGIBILITY_CHIP"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let rawMemory = env["JUNO_PREVIEW_ELIGIBILITY_MEMORY_GB"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard rawChip?.isEmpty == false || rawMemory?.isEmpty == false else { return nil }

        let chipName = rawChip?.isEmpty == false ? rawChip! : "Unknown"
        let generation = rawChip.flatMap { appleChipGeneration(from: $0) ?? generationFromShortChipName($0) }
        let memoryGB = rawMemory.flatMap(Int.init) ?? normalizedMemoryGB(bytes: ProcessInfo.processInfo.physicalMemory)
        return Snapshot(chipName: chipName, chipGeneration: generation, memoryGB: memoryGB)
    }

    private static func generationFromShortChipName(_ raw: String) -> Int? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard trimmed.hasPrefix("M") else { return nil }
        return Int(trimmed.dropFirst())
    }

    private static func normalizedChipName(brand: String, generation: Int?) -> String {
        let trimmed = brand.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        if let generation { return "Apple M\(generation)" }
        return sysctlString("hw.model") ?? "Unknown Mac"
    }

    private static func normalizedMemoryGB(bytes: UInt64) -> Int {
        let gib = Double(bytes) / 1_073_741_824.0
        return max(1, Int(gib.rounded()))
    }

    private static func sysctlString(_ name: String) -> String? {
        var size: size_t = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
