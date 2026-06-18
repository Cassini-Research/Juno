import AppKit
import Darwin
import Foundation

/// Canonical hardware / OS floor for Juno.
///
/// Two tiers, deliberately different in strictness:
///
///   * **macOS version is a hard requirement.** Juno refuses to launch below
///     ``minimumMacOSMajorVersion`` (Sequoia) — enforced once at startup by
///     ``enforceMinimumOSOrTerminate()``, which shows a blocking alert and
///     quits.
///   * **Memory is a soft requirement.** Below ``minimumMemoryGB`` onboarding
///     surfaces a warning (``Snapshot/onboardingWarningMessage``) but the user
///     can still continue.
///
/// Live HUD preview support is *tied to* meeting both floors (see
/// ``JunoPreviewEligibility``), so any Mac that meets Juno's minimum gets live
/// preview on by default rather than gating it behind a higher hardware bar.
enum JunoSystemRequirements {
    /// macOS 15 Sequoia. Hard floor — Juno will not run below this.
    static let minimumMacOSMajorVersion = 15
    static let minimumMacOSName = "Sequoia"
    /// 16 GB. Soft floor — onboarding warns below this but still allows use.
    static let minimumMemoryGB = 16

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

        /// Non-blocking onboarding warning. The OS floor is handled by the hard
        /// block, so this only ever covers the memory floor.
        var onboardingWarningMessage: String? {
            guard meetsMinimumOS, !meetsMinimumMemory else { return nil }
            return "Juno runs best on Macs with at least "
                + "\(JunoSystemRequirements.minimumMemoryGB) GB of memory. This Mac "
                + "has \(memoryGB) GB, so dictation and live preview may be slow."
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

    /// Present a blocking alert and terminate when the host macOS is older than
    /// ``minimumMacOSMajorVersion``. Call once, early, on the main thread.
    @MainActor
    static func enforceMinimumOSOrTerminate() {
        let snapshot = current
        guard !snapshot.meetsMinimumOS else { return }

        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "macOS \(minimumMacOSName) or newer required"
        alert.informativeText = snapshot.unsupportedOSMessage
            ?? "Juno requires macOS \(minimumMacOSName) (\(minimumMacOSMajorVersion)) or newer."
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
