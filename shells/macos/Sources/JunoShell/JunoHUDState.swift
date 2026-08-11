import Foundation

// MARK: - HUDState — typed representation of the floating-HUD state machine
//
// The HUD state machine used to be a free-form `String` written by
// `DictationController` and read by 40+ comparison sites across
// `JunoShellApp.swift`, `OverlayWindow.swift`, and `JunoBrandIslandStack.swift`.
// New states added to the producer routinely failed to update the visibility
// predicate or animation triggers — the audit identifies this as the root
// cause of "loads of HUD issues we keep running into."
//
// This file introduces a typed `HUDState` enum plus a wire-string codec. The
// producer continues to store/emit the wire string for backwards compatibility
// with logging, support bundles, and the broker (whose wire protocol is out of
// scope for this fix). Read sites consume `controller.hudState` instead of
// peeking at the raw string, so adding or renaming a state lights up every
// switch as a compile-time error.
//
// Design notes:
//   - `unknown(String)` is the catch-all for forward-compatibility with
//     wire strings the engine adds before the shell has caught up. We
//     deliberately preserve the raw string so support bundles stay
//     greppable.
//   - `BlockReason` and `ErrorReason` use `String` raw values matching the
//     wire suffix (after `blocked:` / `error:`), with an `unknown(String)`
//     fallback for the same reason as above.
//   - `isInDictationFlow` is the canonical replacement for the duplicated
//     predicate at `OverlayWindow.swift:243-251`.

/// Canonical states the floating HUD can occupy. Mirrors the contract in
/// `MAC_RUNTIME_TRUTH.md`.
public enum HUDState: Equatable {
    case idle
    case checkingCapability
    case checkingMic
    case waitingSpeech
    case listening
    case partialCommit
    case refining
    case blocked(reason: BlockReason)
    case error(reason: ErrorReason)
    /// Wire string the shell does not (yet) recognise. Preserved verbatim so
    /// logs and support bundles still grep against the raw wire format. The
    /// HUD treats this as "not in dictation flow" so we never lock the
    /// overlay open on a typo.
    case unknown(String)

    /// Wire-format string the broker / logger / support bundle uses. Every
    /// `case` here must produce a stable, lossless encoding because
    /// `HUDState.from(wireString:)` is the inverse.
    public var wireString: String {
        switch self {
        case .idle:                return "idle"
        case .checkingCapability:  return "checking_capability"
        case .checkingMic:         return "checking_mic"
        case .waitingSpeech:       return "waiting_speech"
        case .listening:           return "listening"
        case .partialCommit:       return "partial_commit"
        case .refining:            return "refining"
        case .blocked(let reason): return "blocked:\(reason.rawValue)"
        case .error(let reason):   return "error:\(reason.rawValue)"
        case .unknown(let raw):    return raw
        }
    }

    /// Decode a wire string back to the typed enum. Always returns a value —
    /// unrecognised inputs fall back to `.unknown(raw)` so callers never have
    /// to handle an `Optional` for a string the engine emitted.
    public static func from(wireString raw: String) -> HUDState {
        switch raw {
        case "idle":                return .idle
        case "checking_capability": return .checkingCapability
        case "checking_mic":        return .checkingMic
        case "waiting_speech":      return .waitingSpeech
        case "listening":           return .listening
        case "partial_commit":      return .partialCommit
        case "refining":            return .refining
        default:
            if raw.hasPrefix("blocked:") {
                let suffix = String(raw.dropFirst("blocked:".count))
                return .blocked(reason: BlockReason(rawValue: suffix))
            }
            if raw.hasPrefix("error:") {
                let suffix = String(raw.dropFirst("error:".count))
                return .error(reason: ErrorReason(rawValue: suffix))
            }
            return .unknown(raw)
        }
    }

    /// Replaces the duplicated predicate at `OverlayWindow.swift:243-251`.
    /// True iff the floating HUD should be visible solely because of the
    /// state — the actual `updateVisibility` site OR-combines this with
    /// transient flags (done count, action result, copy-ready, etc.).
    public var isInDictationFlow: Bool {
        switch self {
        case .checkingCapability, .checkingMic, .waitingSpeech,
             .listening, .partialCommit, .refining:
            return true
        case .blocked, .error:
            return true
        case .idle, .unknown:
            return false
        }
    }

    /// True iff the HUD should render in the danger / error palette.
    public var isErrorOrBlocked: Bool {
        switch self {
        case .blocked, .error: return true
        default:               return false
        }
    }

    /// Startup states where an impatient repeat tap should be treated as a
    /// duplicate of the opening request, not as the user's stop intent.
    public var isOpeningTransition: Bool {
        switch self {
        case .checkingCapability, .checkingMic, .waitingSpeech:
            return true
        default:
            return false
        }
    }
}

enum JunoDictationHotkeyAction: Equatable {
    case begin
    case stop
    case cancelOpening
    case resetTerminal
    case ignore
    case ignoreStartupRepeat
}

enum JunoDictationHotkeyPolicy {
    static let startupRepeatDebounceSeconds: TimeInterval = 1.0

    static func startupDebounceUntil(startedAt: TimeInterval) -> TimeInterval {
        startedAt + startupRepeatDebounceSeconds
    }

    static func action(
        for state: HUDState,
        now: TimeInterval,
        startupDebounceUntil: TimeInterval
    ) -> JunoDictationHotkeyAction {
        if state.isOpeningTransition, now < startupDebounceUntil {
            return .ignoreStartupRepeat
        }

        switch state {
        case .idle:
            return .begin
        case .checkingCapability:
            return .cancelOpening
        case .checkingMic, .waitingSpeech, .listening, .partialCommit:
            return .stop
        case .blocked, .error:
            return .resetTerminal
        case .refining, .unknown:
            return .ignore
        }
    }
}

// MARK: - Sub-state reasons

/// Reasons the HUD is blocked from running dictation. Wire format is the
/// suffix after `blocked:` (e.g. `blocked:ax_permission_missing` decodes to
/// `.axPermissionMissing`). Unrecognised reasons survive as `.unknown(raw)`
/// so newer engine builds don't crash an older shell.
public enum BlockReason: Equatable {
    case axPermissionMissing
    case unknown(String)

    public init(rawValue: String) {
        switch rawValue {
        case "ax_permission_missing": self = .axPermissionMissing
        default:                       self = .unknown(rawValue)
        }
    }

    public var rawValue: String {
        switch self {
        case .axPermissionMissing: return "ax_permission_missing"
        case .unknown(let raw):    return raw
        }
    }
}

/// Reasons the HUD is in an error state. Wire format is the suffix after
/// `error:` (e.g. `error:mic_no_audio` decodes to `.micNoAudio`).
public enum ErrorReason: Equatable {
    case micNoAudio
    case transcribeFailed
    /// `error:<message>` — the producer at `JunoShellApp.swift:3523` wraps
    /// arbitrary `error.localizedDescription` strings into this case.
    case unknown(String)

    public init(rawValue: String) {
        switch rawValue {
        case "mic_no_audio":      self = .micNoAudio
        case "transcribe_failed": self = .transcribeFailed
        default:                  self = .unknown(rawValue)
        }
    }

    public var rawValue: String {
        switch self {
        case .micNoAudio:       return "mic_no_audio"
        case .transcribeFailed: return "transcribe_failed"
        case .unknown(let raw): return raw
        }
    }
}
