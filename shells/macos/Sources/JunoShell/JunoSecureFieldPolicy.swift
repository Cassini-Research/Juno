import Foundation

/// Centralised privacy decision for AX-detected secure-text inputs.
///
/// When the focused field is a secure text input (AX subrole
/// `AXSecureTextField`), Juno must force five things off for the
/// in-flight utterance:
///
///   1. Context tape entries (selected/focused/clipboard text).
///   2. Learn-from-corrections (post-utterance text-mon observer).
///   3. History append (broker `insertion/committed` write).
///   4. Audio save / upload (broker `ingest_wav` upload of session WAV).
///   5. Paste at cursor (the hotkey-driven Cmd+V into the focused app).
///
/// The audit at HEAD found only one and a half of these gates wired in
/// the shell (context tape strip + a partial paste clamp). The remaining
/// gates were silently relying on engine-side enforcement, leaving zero
/// defence-in-depth at the shell.
///
/// `SecureFieldPolicy` makes the contract a single source of truth.
/// Every callsite asks the policy; the policy is a pure function of the
/// "secure" flag captured at dictation-start (TOCTOU pin: re-reading at
/// paste/learn time would let the user switch fields mid-utterance and
/// bypass the gate). When the engine ALSO gates these surfaces, the
/// shell-side gate is belt-and-braces — two doors are better than one.
enum SecureFieldPolicy {
    /// Resolved decision. All five booleans are `false` when `secure ==
    /// true`; downstream callers still respect their own per-feature
    /// toggles when they are `true` (this gate is permissive only).
    struct Decision: Equatable {
        let allowContext: Bool
        let allowLearning: Bool
        let allowHistory: Bool
        let allowAudioSave: Bool
        let allowPaste: Bool
    }

    /// Resolve all five gates from the secure flag.
    static func decision(secure: Bool) -> Decision {
        let allow = !secure
        return Decision(
            allowContext: allow,
            allowLearning: allow,
            allowHistory: allow,
            allowAudioSave: allow,
            allowPaste: allow
        )
    }

    // Gate-named convenience helpers. Each callsite may import the one
    // it needs without dragging the full ``Decision`` struct around.
    // Behaviourally identical to ``decision(secure:).allowX``; kept as
    // separate methods so the per-callsite intent is grep-friendly and
    // each gate has a one-liner unit test that names its surface.

    /// Gate 1: context tape entries (selected_text, focused_text,
    /// focused_text_before, focused_text_after, clipboard_text).
    static func allowContext(secure: Bool) -> Bool { decision(secure: secure).allowContext }

    /// Gate 2: learn-from-corrections post-utterance observer.
    static func allowLearning(secure: Bool) -> Bool { decision(secure: secure).allowLearning }

    /// Gate 3: history append (broker `insertion/committed`).
    static func allowHistory(secure: Bool) -> Bool { decision(secure: secure).allowHistory }

    /// Gate 4: audio upload (broker `ingest_wav`) and any local save.
    static func allowAudioSave(secure: Bool) -> Bool { decision(secure: secure).allowAudioSave }

    /// Gate 5: paste at cursor (`juno-paste`, in-process Cmd+V, etc.).
    static func allowPaste(secure: Bool) -> Bool { decision(secure: secure).allowPaste }
}
