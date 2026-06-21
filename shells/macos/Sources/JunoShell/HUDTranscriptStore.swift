import CryptoKit
import Foundation

// MARK: - Final-reconciliation patch types (Qwen, stage="final")

struct TranscriptPatchEnvelope: Decodable {
    let schemaVersion: String
    let utteranceId: String
    let stage: String
    let baseVisibleRevision: Int?
    let baseTextHash: String?
    let baseVisibleText: String?
    let stablePrefixChars: Int?
    let correctedText: String
    let ops: [TranscriptPatchOpDTO]
    let confidence: Double

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case utteranceId = "utterance_id"
        case stage
        case baseVisibleRevision = "base_visible_revision"
        case baseTextHash = "base_text_hash"
        case baseVisibleText = "base_visible_text"
        case stablePrefixChars = "stable_prefix_chars"
        case correctedText = "corrected_text"
        case ops
        case confidence
    }
}

struct TranscriptPatchOpDTO: Decodable {
    let op: String
    let startChar: Int
    let endChar: Int
    let text: String
    let reason: String
    let confidence: Double
    let sourceText: String?

    enum CodingKeys: String, CodingKey {
        case op
        case startChar = "start_char"
        case endChar = "end_char"
        case text
        case reason
        case confidence
        case sourceText = "source_text"
    }
}

// MARK: - HUDTranscriptStore (two-zone: LocalAgreement-2)

/// Renders the live transcript HUD with two zones:
///
/// - **Committed**: words that two consecutive Whisper passes agreed on. Append-only
///   by algorithm guarantee — never shrinks, never reorders. Rendered at full opacity.
/// - **Tail**: legacy/debug support for the unstable hypothesis past the agreement
///   boundary. The production engine preview path does not feed this into the HUD;
///   user-visible live text is committed text only.
///
/// The store does not implement an additional acceptance policy. The engine's
/// LocalAgreement-2 state machine (`juno_v2.preview.live_agreement.HypothesisBuffer`)
/// is the single source of truth for what's committed.
///
/// Final reconciliation (Qwen-corrected paste-text) still goes through the patch
/// envelope path so unchanged words keep their span identity for animation.
final class HUDTranscriptStore: ObservableObject {
    /// Combined `committedText + " " + tailText`. Kept for legacy bindings; new
    /// views should observe ``committedText`` and ``tailText`` directly so they
    /// can render the zones with separate opacity / animation.
    @Published private(set) var text: String = ""
    @Published private(set) var committedText: String = ""
    @Published private(set) var tailText: String = ""
    @Published private(set) var spans: [HUDTranscriptSpan] = []
    private(set) var revision: Int = 0
    private(set) var lastHash: String = ""
    var rawText: String {
        Self.compose(committed: committedText, tail: tailText)
    }

    /// Test-only instrumentation — true after the most recent public mutator's body
    /// executed on the main thread. Always true in production after a successful
    /// mutation; HUDTranscriptStoreTests asserts on this.
    private(set) var _debugLastMutationOnMainThread: Bool = false

    // Issue #14: SwiftUI `@Published` mutations from off-main threads produce
    // undefined behaviour. The recorder/network callbacks return off-main, so we
    // hop here. `DispatchQueue.main.sync` is correct as long as no caller already
    // holds the main thread blocked — that's true for all current call sites.
    private func _onMain(_ body: () -> Void) {
        if Thread.isMainThread {
            _debugLastMutationOnMainThread = true
            body()
        } else {
            DispatchQueue.main.sync {
                _debugLastMutationOnMainThread = true
                body()
            }
        }
    }

    func reset() {
        _onMain {
            committedText = ""
            tailText = ""
            text = ""
            spans = []
            revision = 0
            lastHash = Self.visibleHash("")
        }
    }

    // MARK: Live path — engine emits committedText + tailText per chunk

    /// Apply the latest committed prefix from the engine. The engine's
    /// LocalAgreement-2 invariant is "committed is append-only"; this method
    /// asserts that invariant and never shrinks the displayed committed zone.
    ///
    /// Returns true if the committed zone changed.
    @discardableResult
    func applyCommittedPrefix(_ incoming: String) -> Bool {
        var changed = false
        _onMain { changed = _applyCommittedPrefix(incoming) }
        return changed
    }

    @discardableResult
    private func _applyCommittedPrefix(_ incoming: String) -> Bool {
        let trimmed = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == committedText { return false }

        if !committedText.isEmpty && !trimmed.hasPrefix(committedText) {
            // Engine bug: LocalAgreement-2 guarantees committed never shrinks
            // or reorders. If we see it, refuse the update — better a slightly
            // stale HUD than a visibly-shrinking one. The engine should fix
            // its state machine.
            NSLog(
                "HUD: refused committed-prefix shrink — current=%d new=%d",
                committedText.count,
                trimmed.count
            )
            return false
        }

        committedText = trimmed
        _rebuild(origin: .committed)
        return true
    }

    /// Apply the latest unstable tail from the engine. Tail can change
    /// wholesale; production callers keep this empty and use a non-text
    /// activity indicator instead.
    func applyUnstableTail(_ incoming: String) {
        _onMain { _applyUnstableTail(incoming) }
    }

    private func _applyUnstableTail(_ incoming: String) {
        let trimmed = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == tailText { return }
        tailText = trimmed
        _rebuild(origin: .tail)
    }

    /// Convenience: apply both zones in one main-thread hop. This keeps the
    /// strict append-only committed-prefix contract for callers that are fed by
    /// raw LocalAgreement commits.
    @discardableResult
    func applyPreviewChunk(committed: String, tail: String) -> Bool {
        var changed = false
        _onMain {
            let commChanged = _applyCommittedPrefix(committed)
            let trimmedTail = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            let tailChanged = trimmedTail != tailText
            if tailChanged {
                tailText = trimmedTail
                _rebuild(origin: .tail)
            }
            changed = commChanged || tailChanged
        }
        return changed
    }

    /// Apply a root-level preview update that may include a bounded suffix
    /// correction. The engine remains append-only per transport segment, but the
    /// shell stitches multiple pause-bounded segments into one root utterance and
    /// mid-utterance audio checkpoints can return a better root snapshot. Those
    /// root snapshots are allowed to revise the suffix when they still share a
    /// stable leading phrase with what is already visible.
    @discardableResult
    func applyPreviewRevision(committed: String, tail: String) -> Bool {
        var changed = false
        _onMain {
            let trimmedCommitted = committed.trimmingCharacters(in: .whitespacesAndNewlines)
            let trimmedTail = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            var committedChanged = false

            if trimmedCommitted == committedText {
                committedChanged = false
            } else if committedText.isEmpty || trimmedCommitted.hasPrefix(committedText) {
                committedText = trimmedCommitted
                committedChanged = true
            } else if Self.acceptsPreviewSuffixRevision(current: committedText, incoming: trimmedCommitted) {
                committedText = trimmedCommitted
                committedChanged = true
            } else {
                NSLog(
                    "HUD: refused preview-root revision current=%d new=%d",
                    committedText.count,
                    trimmedCommitted.count
                )
            }

            let tailChanged = trimmedTail != tailText
            if tailChanged {
                tailText = trimmedTail
            }
            if committedChanged || tailChanged {
                _rebuild(origin: tailChanged && !committedChanged ? .tail : .committed)
            }
            changed = committedChanged || tailChanged
        }
        return changed
    }

    // MARK: Final path — Qwen-adjudicated text or its patch envelope

    /// Apply a fully-settled final transcript. Used when the broker returns
    /// the final paste-quality string and we want to replace the live preview
    /// with it. The tail zone is cleared.
    func applyFinalText(_ finalText: String) {
        _onMain { _applyFinalText(finalText) }
    }

    private func _applyFinalText(_ finalText: String) {
        let settled = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !settled.isEmpty else { return }
        committedText = settled
        tailText = ""
        _rebuild(origin: .corrected)
    }

    /// Apply a transcript patch envelope. Final patches settle the whole HUD.
    /// Live patches are accepted only when they are anchored to the exact
    /// visible snapshot the engine corrected, or to an append-only prefix of the
    /// current HUD; this lets memory/screen corrections reach the HUD without
    /// handing live correction permission to wholesale-replace unstable text.
    @discardableResult
    func applyPatchEnvelope(_ envelope: TranscriptPatchEnvelope) -> Bool {
        var result = false
        _onMain { result = _applyPatchEnvelope(envelope) }
        return result
    }

    private func _applyPatchEnvelope(_ envelope: TranscriptPatchEnvelope) -> Bool {
        guard envelope.schemaVersion == "transcript_patch_v1" else { return false }
        if envelope.stage == "final" {
            return _applyFinalReconciliationEnvelope(envelope)
        }
        if envelope.stage == "live" {
            return _applyLiveCorrectionEnvelope(envelope)
        }
        NSLog("HUD: ignored unknown patch envelope stage=%@", envelope.stage)
        return false
    }

    private func _applyLiveCorrectionEnvelope(_ envelope: TranscriptPatchEnvelope) -> Bool {
        let snapshot = (envelope.baseVisibleText ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let current = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let corrected = envelope.correctedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !snapshot.isEmpty, !current.isEmpty, !corrected.isEmpty else { return false }
        guard envelope.confidence >= 0.60 else { return false }
        guard Self.livePatchOpsAreBounded(envelope.ops, snapshotChars: snapshot.count) else {
            NSLog("HUD: refused live patch with unbounded ops")
            return false
        }
        guard Self.livePatchTokenDeltaIsBounded(snapshot: snapshot, corrected: corrected) else {
            NSLog("HUD: refused live patch token delta current=%d corrected=%d", snapshot.count, corrected.count)
            return false
        }

        let suffix: String
        if current == snapshot {
            suffix = ""
        } else {
            guard current.hasPrefix(snapshot) else {
                NSLog("HUD: refused live patch snapshot drift current=%d snapshot=%d", current.count, snapshot.count)
                return false
            }
            let suffixStart = current.index(current.startIndex, offsetBy: snapshot.count)
            let rawSuffix = String(current[suffixStart...])
            guard rawSuffix.first?.isWhitespace == true else {
                NSLog("HUD: refused live patch mid-token drift current=%d snapshot=%d", current.count, snapshot.count)
                return false
            }
            suffix = rawSuffix
        }

        if suffix.isEmpty, corrected == current {
            return true
        }
        let changedRanges = applyOpsToCurrentText(envelope.ops)
        committedText = Self.liveCorrectedText(corrected: corrected, suffix: suffix)
        tailText = ""
        _rebuild(origin: .committed, changedRanges: changedRanges)
        return true
    }

    private static func livePatchOpsAreBounded(_ ops: [TranscriptPatchOpDTO], snapshotChars: Int) -> Bool {
        guard ops.count <= 8 else { return false }
        for op in ops {
            guard op.confidence >= 0.60 else { return false }
            guard op.startChar >= 0, op.endChar >= op.startChar, op.endChar <= snapshotChars else {
                return false
            }
            switch op.op {
            case "replace", "insert", "delete", "punctuate", "case":
                continue
            default:
                return false
            }
        }
        return true
    }

    private static func livePatchTokenDeltaIsBounded(snapshot: String, corrected: String) -> Bool {
        let before = previewRevisionTokens(snapshot).count
        let after = previewRevisionTokens(corrected).count
        guard before > 0, after > 0 else { return false }
        if after > before + 4 { return false }
        if after < max(1, before - 5) { return false }
        return true
    }

    private static func liveCorrectedText(corrected: String, suffix: String) -> String {
        let base = corrected.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !suffix.isEmpty else { return base }
        return base + suffix
    }

    private static func acceptsPreviewSuffixRevision(current: String, incoming: String) -> Bool {
        if acceptsSpokenPunctuationCueRevision(current: current, incoming: incoming) {
            return true
        }
        let currentTokens = previewRevisionTokens(current)
        let incomingTokens = previewRevisionTokens(incoming)
        let minCount = min(currentTokens.count, incomingTokens.count)
        guard minCount >= 4 else { return false }
        var common = 0
        while common < minCount, currentTokens[common] == incomingTokens[common] {
            common += 1
        }
        let required = min(8, max(3, minCount / 2))
        return common >= required
    }

    /// Determiners that keep a following "new line"/"new paragraph"/"full stop"
    /// etc. as literal prose ("the new line is short") rather than a spoken
    /// cue. Shared by ``smoothForDisplay``, ``transcriptWithSpokenBreakCuesResolved``
    /// and the committed-shrink gate so the three cannot drift apart.
    private static let spokenBreakCueDeterminers: Set<String> = [
        "the", "a", "an", "this", "that", "each", "every", "my", "your", "our",
    ]

    private static func acceptsSpokenPunctuationCueRevision(current: String, incoming: String) -> Bool {
        let currentTokens = previewRevisionTokens(current)
        let incomingTokens = previewRevisionTokens(incoming)
        guard currentTokens.count >= incomingTokens.count else { return false }
        guard currentTokens.prefix(incomingTokens.count).elementsEqual(incomingTokens) else { return false }
        let dropped = Array(currentTokens.dropFirst(incomingTokens.count))
        guard !dropped.isEmpty, dropped.count <= 2 else { return false }
        // Determiner guard (mirrors smoothForDisplay / transcriptWithSpokenBreakCuesResolved):
        // "...the new line", "...a full stop" is real dictated prose, not a
        // spoken punctuation cue — never shrink those words off the committed
        // HUD. The cue must not be immediately preceded by a determiner.
        if let preceding = incomingTokens.last,
           spokenBreakCueDeterminers.contains(preceding) {
            return false
        }
        return isSpokenPunctuationCueSuffix(dropped)
    }

    private static func isSpokenPunctuationCueSuffix(_ tokens: [String]) -> Bool {
        let joined = tokens.joined(separator: " ")
        switch joined {
        // Only accept shrinking committed HUD text when the dropped suffix is
        // an unambiguous punctuation cue. The bare words "new", "full",
        // "question", "exclamation" and "period" are excluded: they are also
        // ordinary trailing words ("a grace period", "the full report"), and
        // accepting them would silently drop a real dictated word from the HUD.
        case "comma", "colon", "semicolon",
             "question mark",
             "exclamation point", "exclamation mark",
             "full stop",
             "new line", "newline", "line break", "new paragraph":
            return true
        default:
            return false
        }
    }

    private static func previewRevisionTokens(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        let apostrophe = UnicodeScalar("'")
        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || scalar == apostrophe {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                tokens.append(current.lowercased())
                current = ""
            }
        }
        if !current.isEmpty {
            tokens.append(current.lowercased())
        }
        return tokens
    }

    /// Apply a final reconciliation envelope via the ops path so unchanged
    /// prefix words keep their span identity. Falls back to a hard replace if
    /// the engine's snapshot has diverged from what's on screen.
    private func _applyFinalReconciliationEnvelope(_ envelope: TranscriptPatchEnvelope) -> Bool {
        let snapshot = (envelope.baseVisibleText ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let current = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let corrected = envelope.correctedText.trimmingCharacters(in: .whitespacesAndNewlines)

        if envelope.ops.isEmpty {
            if corrected == current { return true }
            NSLog("HUD: final patch with empty ops, hard-replacing")
            _applyFinalText(envelope.correctedText)
            return true
        }

        if !snapshot.isEmpty, current != snapshot {
            NSLog(
                "HUD: final patch snapshot diverged (current=%d snapshot=%d) — hard-replacing",
                current.count, snapshot.count
            )
            _applyFinalText(envelope.correctedText)
            return true
        }

        let ranges = applyOpsToCurrentText(envelope.ops)
        guard !ranges.isEmpty else { return true }
        // After ops, the visible text becomes the corrected paste-final form.
        // Clear tail; committed = corrected.
        committedText = corrected
        tailText = ""
        _rebuild(origin: .corrected, changedRanges: ranges)
        return true
    }

    private func applyOpsToCurrentText(_ ops: [TranscriptPatchOpDTO]) -> [Range<Int>] {
        var current = text
        var changed: [Range<Int>] = []
        for op in ops.sorted(by: { $0.startChar > $1.startChar }) {
            let start = max(0, min(op.startChar, current.count))
            let end = max(start, min(op.endChar, current.count))
            guard let s = current.index(current.startIndex, offsetBy: start, limitedBy: current.endIndex),
                  let e = current.index(current.startIndex, offsetBy: end, limitedBy: current.endIndex)
            else { continue }
            current.replaceSubrange(s..<e, with: op.text)
            let insertedEnd = start + op.text.count
            if insertedEnd > start {
                changed.append(start..<insertedEnd)
            }
        }
        return changed
    }

    // MARK: Span rebuild



    // MARK: Display smoothing (render-only; never touches stored text)

    static func smoothForDisplay(_ committed: String) -> String {
        guard !committed.isEmpty else { return committed }
        var out = committed
        // Drop a single seam period followed by a lowercase continuation
        // ("how we think. ask," → "how we think ask,"). Ellipses and
        // initialisms are preserved.
        if let regex = try? NSRegularExpression(pattern: "(?<![.A-Z])\\.( +)(?=[a-z])") {
            out = regex.stringByReplacingMatches(
                in: out, range: NSRange(out.startIndex..., in: out), withTemplate: "$1"
            )
        }
        // Spoken newline cues stay visible and also create the intended
        // display break unless preceded by a determiner ("the new line is
        // short" stays literal). The cue text remains on the HUD so users can
        // see that Juno heard it before final paste converts it to structure.
        if let cue = try? NSRegularExpression(
            pattern: "(?:^|(?<= ))[Nn]ew +([Ll]ine|[Pp]aragraph)\\b[ .,]*",
            options: []
        ) {
            let ns = out as NSString
            var result = ""
            var cursor = 0
            for match in cue.matches(in: out, range: NSRange(location: 0, length: ns.length)) {
                let before = ns.substring(to: match.range.location)
                let prevWord = before.split(separator: " ").last.map {
                    $0.trimmingCharacters(in: CharacterSet(charactersIn: ",.;:")).lowercased()
                } ?? ""
                result += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
                if spokenBreakCueDeterminers.contains(prevWord) {
                    result += ns.substring(with: match.range)
                } else {
                    let cueText = ns.substring(with: match.range)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        .trimmingCharacters(in: CharacterSet(charactersIn: ".,;:!?"))
                    result += cueText
                    let cueWord = cueText.lowercased()
                    result += cueWord.contains("paragraph") ? "\n\n" : "\n"
                }
                cursor = match.range.location + match.range.length
            }
            result += ns.substring(from: cursor)
            out = result
        }
        return out
    }

    /// Spoken "new line"/"new paragraph" cues → real breaks, with the literal
    /// cue words REMOVED. `smoothForDisplay` deliberately keeps the cue text
    /// visible on the live HUD; this variant is for copy/paste surfaces (e.g.
    /// the broker-failure fallback transcript) where the literal words must
    /// never reach the user's clipboard/document. The same determiner guard
    /// ("the new line is short") keeps genuine prose literal. Pass RAW text
    /// (e.g. `rawText`), not already-smoothed `text`, to avoid double breaks.
    static func transcriptWithSpokenBreakCuesResolved(_ text: String) -> String {
        guard !text.isEmpty,
              let cue = try? NSRegularExpression(
                pattern: "(?:^|(?<= ))[Nn]ew +([Ll]ine|[Pp]aragraph)\\b[ .,]*",
                options: []
              )
        else { return text }
        let ns = text as NSString
        var result = ""
        var cursor = 0
        for match in cue.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let before = ns.substring(to: match.range.location)
            let prevWord = before.split(separator: " ").last.map {
                $0.trimmingCharacters(in: CharacterSet(charactersIn: ",.;:")).lowercased()
            } ?? ""
            result += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            if spokenBreakCueDeterminers.contains(prevWord) {
                result += ns.substring(with: match.range)
            } else {
                let cueText = ns.substring(with: match.range)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: CharacterSet(charactersIn: ".,;:!?"))
                result += cueText.lowercased().contains("paragraph") ? "\n\n" : "\n"
            }
            cursor = match.range.location + match.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    private func _rebuild(origin: HUDTranscriptSpan.Origin, changedRanges: [Range<Int>] = []) {
        revision += 1
        text = Self.compose(committed: Self.smoothForDisplay(committedText), tail: tailText)
        lastHash = Self.visibleHash(text)

        var newSpans: [HUDTranscriptSpan] = []

        // Display-only smoothing of rolling-window seams. The raw
        // committedText keeps the engine's append-only contract (all
        // invariant checks above run on it); only what the user SEES is
        // cleaned: stray window-join periods before lowercase continuations
        // are dropped and spoken newline cues render as actual breaks.
        let displayCommitted = Self.smoothForDisplay(committedText)

        for (i, item) in Self.wordSpans(for: displayCommitted).enumerated() {
            newSpans.append(HUDTranscriptSpan(
                id: "c-\(i)",
                text: item.word,
                origin: .committed,
                revision: revision,
                changed: changedRanges.contains { $0.overlaps(item.range) }
            ))
        }

        let commLen = committedText.isEmpty ? 0 : committedText.count + 1
        for (i, item) in Self.wordSpans(for: tailText).enumerated() {
            // Spans for tail words use a different prefix so they animate
            // independently from committed spans across updates.
            let absoluteRange = (item.range.lowerBound + commLen)..<(item.range.upperBound + commLen)
            newSpans.append(HUDTranscriptSpan(
                id: "t-\(i)",
                text: item.word,
                origin: .tail,
                revision: revision,
                changed: changedRanges.contains { $0.overlaps(absoluteRange) }
            ))
        }

        // If this rebuild came from a final/corrected origin, mark all spans
        // with that origin instead of committed/tail. Final paste-display has
        // no tail.
        if origin == .corrected {
            newSpans = newSpans.map { span in
                HUDTranscriptSpan(
                    id: span.id,
                    text: span.text,
                    origin: .corrected,
                    revision: span.revision,
                    changed: span.changed
                )
            }
        }

        spans = newSpans
    }

    private static func compose(committed: String, tail: String) -> String {
        if committed.isEmpty { return tail }
        if tail.isEmpty { return committed }
        return committed + " " + tail
    }

    // MARK: Word-span helper (unchanged from prior implementation)

    private static func wordSpans(for value: String) -> [(word: String, range: Range<Int>)] {
        var out: [(String, Range<Int>)] = []
        var start: String.Index?
        var offset = 0
        var startOffset = 0
        for idx in value.indices {
            let ch = value[idx]
            if ch.isWhitespace || ch.isNewline {
                if let s = start {
                    out.append((String(value[s..<idx]), startOffset..<offset))
                    start = nil
                }
            } else if start == nil {
                start = idx
                startOffset = offset
            }
            offset += 1
        }
        if let s = start {
            out.append((String(value[s..<value.endIndex]), startOffset..<offset))
        }
        return out
    }

    private static func visibleHash(_ value: String) -> String {
        let normalized = value.split { $0.isWhitespace || $0.isNewline }.joined(separator: " ")
        let digest = SHA256.hash(data: Data(normalized.utf8))
        return digest.compactMap { String(format: "%02x", $0) }.joined().prefix(16).description
    }
}
