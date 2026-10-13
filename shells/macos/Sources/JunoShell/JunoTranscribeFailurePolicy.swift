import Foundation

/// Pure decisions for surfacing an `ok: false` final-transcription response.
///
/// The engine answers `ok: false` in two very different situations. Either it
/// broke (`transcribe_failed`, an unreachable or incompatible engine, a
/// transport error), or it worked and one of its own guards judged that the
/// take held no usable speech. Recording both as `transcribe_failed` tells the
/// user the product failed after a silent take and buries real engine errors
/// in History. This type maps the engine's `error_code` to what the shell
/// records and shows, so the mapping lives in one testable place instead of
/// being re-derived from string comparisons at each call site.
enum JunoTranscribeFailurePolicy {
    enum Kind: Equatable {
        /// The engine found nothing it could call speech in the take.
        case noSpeechDetected
        /// The engine decoded text and its hallucination guard discarded it.
        /// The user may well have spoken, so this must never be presented as
        /// "no speech".
        case transcriptRejected
        /// A genuine engine or transport failure.
        case engineFailure
    }

    struct Disposition: Equatable {
        let kind: Kind
        /// `failure_reason` persisted on the history row. The engine's own
        /// code for a guard rejection; `transcribe_failed` otherwise.
        let failureReason: String
        /// Terminal HUD wire state when the take leaves nothing to offer.
        let hudStateWire: String
        /// True when the outcome ends the session before the live-caption
        /// draft is offered for copy. Only `low_signal_audio`, the engine's
        /// audio-level verdict, does this, as it always has. Every other
        /// verdict keeps the draft recoverable: the text-level guards judge
        /// the decoded words and can misfire on quiet real speech.
        let skipsLiveDraftFallback: Bool

        /// Inline HUD hint for the outcome, if it has one.
        var speechHint: String? {
            kind == .noSpeechDetected ? JunoTranscribeFailurePolicy.noSpeechHint : nil
        }
    }

    struct HUDErrorCopy: Equatable {
        let title: String
        let message: String
    }

    static let genericFailureReason = "transcribe_failed"
    static let noSpeechHint = "No clear speech detected."
    /// The sentence History already shows for `empty_audio`, reused for every
    /// no-speech verdict so one outcome reads the same on the HUD and in
    /// History.
    static let noSpeechDetail = "Juno did not hear enough audio to transcribe."

    /// `userInfo` key under which `JunoBroker.parseTranscribeResponse` carries
    /// the engine's `error_code` on a failed transcribe result.
    private static let engineErrorCodeUserInfoKey = "JunoErrorCode"

    private static let lowSignalAudioCode = "low_signal_audio"

    /// Engine verdicts that mean "nothing was said".
    private static let noSpeechCodes: Set<String> = [
        lowSignalAudioCode,
        "silence_hallucination",
        "empty_audio",
    ]

    /// Engine verdicts that mean "something was decoded and thrown away".
    private static let rejectedTranscriptCodes: Set<String> = [
        "hallucinated_transcript",
    ]

    static func disposition(forEngineErrorCode code: String?) -> Disposition {
        let normalized = code?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if noSpeechCodes.contains(normalized) {
            return Disposition(
                kind: .noSpeechDetected,
                failureReason: normalized,
                hudStateWire: HUDState.error(reason: .lowSignalAudio).wireString,
                skipsLiveDraftFallback: normalized == lowSignalAudioCode
            )
        }
        if rejectedTranscriptCodes.contains(normalized) {
            return Disposition(
                kind: .transcriptRejected,
                failureReason: normalized,
                hudStateWire: HUDState.error(reason: .transcribeFailed).wireString,
                skipsLiveDraftFallback: false
            )
        }
        return Disposition(
            kind: .engineFailure,
            failureReason: genericFailureReason,
            hudStateWire: HUDState.error(reason: .transcribeFailed).wireString,
            skipsLiveDraftFallback: false
        )
    }

    /// Disposition for the error `JunoBroker.transcribeWav` completes with.
    /// Transport errors carry no engine code and stay engine failures.
    static func disposition(forBrokerError error: Error) -> Disposition {
        let code = (error as NSError).userInfo[engineErrorCodeUserInfoKey] as? String
        return disposition(forEngineErrorCode: code)
    }

    /// Error-island copy for the terminal states this policy produces, or nil
    /// when the island should keep its own copy for the state.
    static func hudErrorCopy(for state: HUDState) -> HUDErrorCopy? {
        guard case .error(reason: .lowSignalAudio) = state else { return nil }
        return HUDErrorCopy(
            title: "No clear speech detected",
            message: noSpeechDetail
        )
    }

    /// History label for a persisted guard-rejection reason, or nil when the
    /// reason is not one this policy classifies.
    static func historyLabel(forFailureReason reason: String) -> String? {
        switch disposition(forEngineErrorCode: reason).kind {
        case .noSpeechDetected:
            return noSpeechDetail
        case .transcriptRejected:
            return "Juno discarded an unreliable transcript instead of inserting it."
        case .engineFailure:
            return nil
        }
    }
}
