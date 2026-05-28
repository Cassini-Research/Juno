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
    /// wholesale; the visual treatment (dimmed rendering) makes this an
    /// honest "thinking" indicator instead of a glitch.
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

    /// Apply a stage="final" patch envelope so unchanged words keep their span
    /// identity. Live-stage envelopes are NOT accepted — the live HUD is fed
    /// exclusively by ``applyPreviewChunk`` now.
    @discardableResult
    func applyPatchEnvelope(_ envelope: TranscriptPatchEnvelope) -> Bool {
        var result = false
        _onMain { result = _applyPatchEnvelope(envelope) }
        return result
    }

    private func _applyPatchEnvelope(_ envelope: TranscriptPatchEnvelope) -> Bool {
        guard envelope.schemaVersion == "transcript_patch_v1" else { return false }
        guard envelope.stage == "final" else {
            // Live-stage patches are no longer accepted; they used to drive the
            // HUD-freeze bug (eight gates returning previous on drift).
            NSLog("HUD: ignored non-final patch envelope stage=%@", envelope.stage)
            return false
        }
        return _applyFinalReconciliationEnvelope(envelope)
    }

    private static func acceptsPreviewSuffixRevision(current: String, incoming: String) -> Bool {
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

    private func _rebuild(origin: HUDTranscriptSpan.Origin, changedRanges: [Range<Int>] = []) {
        revision += 1
        text = Self.compose(committed: committedText, tail: tailText)
        lastHash = Self.visibleHash(text)

        var newSpans: [HUDTranscriptSpan] = []

        for (i, item) in Self.wordSpans(for: committedText).enumerated() {
            let token = Self.idToken(item.word)
            newSpans.append(HUDTranscriptSpan(
                id: "c-\(i)-\(token)",
                text: item.word,
                origin: .committed,
                revision: revision,
                changed: changedRanges.contains { $0.overlaps(item.range) }
            ))
        }

        let commLen = committedText.isEmpty ? 0 : committedText.count + 1
        for (i, item) in Self.wordSpans(for: tailText).enumerated() {
            let token = Self.idToken(item.word)
            // Spans for tail words use a different prefix so they animate
            // independently from committed spans across updates.
            let absoluteRange = (item.range.lowerBound + commLen)..<(item.range.upperBound + commLen)
            newSpans.append(HUDTranscriptSpan(
                id: "t-\(i)-\(token)",
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

    private static func idToken(_ word: String) -> String {
        word.lowercased().filter { $0.isLetter || $0.isNumber }
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
