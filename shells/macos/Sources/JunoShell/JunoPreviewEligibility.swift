import Foundation

/// Hardware gate for the live HUD transcription preview.
///
/// Preview support is **tied to Juno's system requirements** (see
/// ``JunoSystemRequirements``): any Mac that meets the minimum macOS version and
/// memory floor is eligible, and preview is on by default there. Dictation
/// itself works on every Mac Juno runs on — this gate only governs the optional
/// live-preview ASR path, which competes with final transcription for
/// resources.
enum JunoPreviewEligibility {
    struct Snapshot: Equatable {
        let osMajorVersion: Int
        let memoryGB: Int
        let chipName: String

        /// Eligible exactly when the Mac meets Juno's minimum requirements.
        var isEligible: Bool {
            osMajorVersion >= JunoSystemRequirements.minimumMacOSMajorVersion
                && memoryGB >= JunoSystemRequirements.minimumMemoryGB
        }

        /// Perf caveat shown for eligible Macs near the memory floor. Above
        /// 32 GB preview comfortably co-exists with final transcription.
        var warningMessage: String? {
            guard isEligible, memoryGB < 32 else { return nil }
            return "Live preview can slow final transcription on \(memoryGB) GB Macs."
        }

        /// Why preview is unavailable (the OS floor is enforced as a hard block
        /// elsewhere, so an ineligible *running* Mac is always memory-limited).
        var unavailableMessage: String? {
            guard !isEligible else { return nil }
            if memoryGB < JunoSystemRequirements.minimumMemoryGB {
                return "Live preview needs at least "
                    + "\(JunoSystemRequirements.minimumMemoryGB) GB of memory. "
                    + "This Mac has \(memoryGB) GB."
            }
            return "Live preview isn’t available on this Mac."
        }
    }

    static var current: Snapshot {
        let req = JunoSystemRequirements.current
        return Snapshot(
            osMajorVersion: req.osMajorVersion,
            memoryGB: req.memoryGB,
            chipName: req.chipName
        )
    }
}
