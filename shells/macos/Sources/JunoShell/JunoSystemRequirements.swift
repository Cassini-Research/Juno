import AppKit
import Darwin
import Foundation

/// Canonical hardware / OS floor for Juno.
///
/// Hard launch requirements:
///
///   * **macOS version.** Juno refuses to launch below
///     ``minimumMacOSMajorVersion`` (Sequoia).
///   * **Memory.** Juno refuses to launch below ``minimumMemoryGB``.
///
/// Live HUD preview support follows the hard OS floor (see
/// ``JunoPreviewEligibility``). The app-level launch gate runs before
/// onboarding or engine startup.
enum JunoSystemRequirements {
    /// macOS 15 Sequoia. Hard floor — Juno will not run below this.
    static let minimumMacOSMajorVersion = 15
    static let minimumMacOSName = "Sequoia"
    /// 24 GB. Hard floor — Juno will not run below this.
    static let minimumMemoryGB = 24

    struct Snapshot: Equatable {
        let osMajorVersion: Int
        let memoryGB: Int
        let chipName: String

        var meetsMinimumOS: Bool {
            osMajorVersion >= JunoSystemRequirements.minimumMacOSMajorVersion
        }
        var meetsMinimumMemory: Bool {
            memoryGB >= JunoSystemRequirements.minimumMemoryGB
        }
        var meetsAllRequirements: Bool { meetsMinimumOS && meetsMinimumMemory }

        /// Blocking copy shown when the OS is too old to run Juno at all.
        var unsupportedOSMessage: String? {
            guard !meetsMinimumOS else { return nil }
            return "Juno requires macOS \(JunoSystemRequirements.minimumMacOSName) "
                + "(\(JunoSystemRequirements.minimumMacOSMajorVersion)) or newer. "
                + "This Mac is running macOS \(osMajorVersion)."
        }

        /// Blocking copy shown when the host has too little RAM to run Juno.
        var unsupportedMemoryMessage: String? {
            guard meetsMinimumOS, !meetsMinimumMemory else { return nil }
            return "Juno requires at least "
                + "\(JunoSystemRequirements.minimumMemoryGB) GB of memory. This Mac "
                + "has \(memoryGB) GB."
        }

        var unsupportedRequirementsMessage: String? {
            unsupportedOSMessage ?? unsupportedMemoryMessage
        }

        var unsupportedRequirementsTitle: String? {
            if !meetsMinimumOS {
                return "macOS \(JunoSystemRequirements.minimumMacOSName) or newer required"
            }
            if !meetsMinimumMemory {
                return "\(JunoSystemRequirements.minimumMemoryGB) GB memory required"
            }
            return nil
        }

        /// Retained for onboarding UI compatibility. Hardware requirements are
        /// hard launch gates, so no non-blocking warning is surfaced here.
        var onboardingWarningMessage: String? {
            nil
        }
    }

    static var current: Snapshot { cachedSnapshot }

    private static let cachedSnapshot: Snapshot = {
        let env = ProcessInfo.processInfo.environment
        let osMajor = env["JUNO_REQUIREMENTS_OS_MAJOR"].flatMap(Int.init)
            ?? ProcessInfo.processInfo.operatingSystemVersion.majorVersion
        // Accept the legacy preview-eligibility memory override as a fallback so
        // existing QA recipes keep working.
        let memoryGB = (env["JUNO_REQUIREMENTS_MEMORY_GB"] ?? env["JUNO_PREVIEW_ELIGIBILITY_MEMORY_GB"])
            .flatMap(Int.init)
            ?? normalizedMemoryGB(bytes: ProcessInfo.processInfo.physicalMemory)
        let chipName = sysctlString("machdep.cpu.brand_string")
            ?? sysctlString("hw.model")
            ?? "Unknown Mac"
        return Snapshot(osMajorVersion: osMajor, memoryGB: memoryGB, chipName: chipName)
    }()

    /// Present a blocking alert and terminate when this Mac does not meet
    /// Juno's launch requirements. Call once, early, on the main thread.
    @MainActor
    static func enforceMinimumRequirementsOrTerminate() {
        let snapshot = current
        guard !snapshot.meetsAllRequirements else { return }

        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = snapshot.unsupportedRequirementsTitle ?? "Juno cannot run on this Mac"
        alert.informativeText = snapshot.unsupportedRequirementsMessage
            ?? "Juno requires macOS \(minimumMacOSName) (\(minimumMacOSMajorVersion)) or newer "
            + "and at least \(minimumMemoryGB) GB of memory."
        alert.addButton(withTitle: "Quit")
        alert.runModal()

        NSApp.terminate(nil)
        // `terminate` can return on some code paths before AppKit tears down;
        // hard-exit guarantees we never proceed into engine / onboarding
        // bring-up on an unsupported OS.
        exit(0)
    }

    // MARK: - Hardware probing helpers

    static func normalizedMemoryGB(bytes: UInt64) -> Int {
        let gib = Double(bytes) / 1_073_741_824.0
        return max(1, Int(gib.rounded()))
    }

    /// Parses the Apple Silicon generation ("M3" → 3) from a CPU brand string.
    /// Returns `nil` for Intel / unparseable strings. Kept for display + future
    /// gating; chip generation no longer gates any feature on its own.
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

    static func sysctlString(_ name: String) -> String? {
        var size: size_t = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
