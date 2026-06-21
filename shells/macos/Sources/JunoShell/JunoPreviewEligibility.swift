import Foundation

/// Hardware gate for the live HUD transcription preview.
///
/// Preview support follows Juno's hard launch requirements (see
/// ``JunoSystemRequirements``): any Mac that can run Juno gets preview on by
/// default.
enum JunoPreviewEligibility {
    struct Snapshot: Equatable {
        let osMajorVersion: Int
        let memoryGB: Int
        let chipName: String

        /// Eligible exactly when the Mac meets Juno's hard launch requirements.
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

        /// Why preview is unavailable. Requirements are enforced as a hard block
        /// elsewhere, so this is only expected in tests or unusual launch paths.
        var unavailableMessage: String? {
            guard !isEligible else { return nil }
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
