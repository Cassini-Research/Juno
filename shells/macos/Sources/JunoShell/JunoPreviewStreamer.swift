// JunoPreviewStreamer.swift
//
// Drives the engine-side live preview during dictation. While the user is
// speaking, the shell pushes ~120 ms PCM chunks here and this class POSTs
// them to ``/api/broker/dictation/preview/chunk`` on the broker. Each
// response carries the cumulative-utterance partial text the engine's
// preview backend produced for that chunk; the shell renders it directly
// in the HUD with no Apple Speech intermediate.
//
// Why a dedicated class (and not just a closure inside JunoShellApp):
//   - The streamer owns the per-utterance ``decode_seq`` counter; spreading
//     that across the recorder callback path was leak-prone.
//   - Encapsulating the network/decode lifecycle here makes preview failure
//     explicit without introducing a second transcription engine.
//   - One TURN_OFF point at ``cancel()`` so unrelated cleanup doesn't have
//     to know about partial-decode bookkeeping.

import Foundation

/// Lives on the main thread by convention — `DictationController` (the
/// only owner) is itself main-thread-bound and broker completions return
/// on the main queue. Audio callbacks can arrive off-main, so `enqueue`
/// bounces those frames to main before touching streamer state. We avoid
/// `@MainActor` to keep call-site noise low; if Swift 6 isolation is
/// ever turned on, the right answer is to also annotate `DictationController`.
final class JunoPreviewStreamer {

    /// The currently streaming utterance. ``nil`` between sessions.
    private(set) var utteranceId: String?
    /// Transport id for the active preview segment. The public
    /// ``utteranceId`` remains the root dictation id; this id is what the
    /// preview backend sees so pause-bounded segments get independent decoder
    /// state.
    private var activeSegmentId: String?
    private var segmentIndex: Int = 0
    private var segmentStartedAt: Date?
    /// Sequential decode index sent on each chunk; the broker uses this to
    /// reset its decoder state on ``decode_seq=0`` and to discard out-of-order
    /// chunks. Resets for each preview segment.
    private var decodeSeq: Int = 0
    /// Wall-clock when the first chunk was sent for the current utterance.
    /// Used by callers to measure first visible local preview latency.
    private(set) var firstChunkSentAt: Date?
    /// Wall-clock when the first non-empty engine partial arrived. ``nil``
    /// while we are still waiting (or never if the budget expired and we
    /// never got a partial).
    private(set) var firstPartialAt: Date?
    /// Set when the streamer has decided to give up on engine output for
    /// the current utterance (network error, timeout, backend unavailable).
    /// Subsequent partials are dropped so a late engine answer doesn't fight
    /// corrected text already on screen.
    private(set) var didGiveUp: Bool = false
    /// Consecutive broker errors on this utterance. Reset on every successful
    /// response. We only give up after several in a row so one slow decode
    /// does not kill the live HUD path mid-utterance.
    private var consecutiveErrorCount: Int = 0
    private var activeSegmentText: String = ""
    private var rollingSegmentId: String?
    private var rollTimeoutWork: DispatchWorkItem?

    /// Called on the main queue with each preview chunk's two-zone result.
    ///
    /// - `committed`: words the engine's LocalAgreement state machine has
    ///   confirmed. Append-only for a root utterance.
    /// - `tail`: unstable hypothesis past the committed boundary. May change
    ///   wholesale between decodes.
    /// - `isFinal`: true only on the explicit ``finish()`` chunk at stop time.
    var onPartial: ((_ uid: String, _ committed: String, _ tail: String, _ isFinal: Bool) -> Void)?
    /// Called on the main queue when the engine path is unusable for this
    /// utterance (decode failure, network error).
    var onGiveUp: ((_ uid: String, _ reason: String) -> Void)?
    /// Called when preview freshness is at risk. Live Qwen correction is
    /// opportunistic; the owner uses this signal to pause correction rather
    /// than letting it compete with caption freshness.
    var onBackpressure: ((_ uid: String, _ reason: String) -> Void)?
    /// Current HUD text from a fallback or corrected path. The broker uses
    /// this only to decide whether a late preview fragment overlaps text
    /// already visible to the user.
    var visibleTextHint: (() -> String)?
    /// Context terms captured at dictation start. The preview backend uses
    /// them only for evidence-gated repair, not as an ASR prompt.
    var candidateEntities: (() -> [String])?
    /// Frozen context snapshots for the active dictation. Kept separate from
    /// ``visibleTextHint`` so HUD misspellings do not feed back as context.
    var sessionContextTape: (() -> [String: Any])?
    var postJSON: (String, [String: Any], @escaping ([String: Any]) -> Void) -> Void = {
        path, payload, completion in
        JunoBroker.postJSON(path: path, payload: payload, completion: completion)
    }

    /// PCM chunks accumulate in a small ring so we don't fire a request
    /// per recorder callback (which deliver buffers smaller than the chunk
    /// budget). When ``pendingPCM.count`` crosses ``minChunkBytes`` we drain
    /// it into a single POST.
    private var pendingPCM = Data()
    /// 120 ms @ 16 kHz mono int16 = 16000 * 0.12 * 2 = 3840 bytes.
    private let minChunkBytes = 3_840
    /// Hard cap so pathologically slow decode never lets pending balloon.
    private let maxQueuedBytes = 64_000
    private var inFlightCount = 0
    // Stateful preview decoders must see chunks in strict sequence. Keeping
    // one request in flight avoids request-thread reorder and prevents the
    // preview lane from building work faster than the single decode owner can
    // consume it.
    private let maxInFlight = 1
    private struct PendingDecode {
        var rootUid: String
        var segmentUid: String
        var seq: Int
        var data: Data
        var isFinal: Bool
        var createdAt: Date
        var coalescedChunks: Int
    }
    private var pendingDecode: PendingDecode?
    private var isFinishing = false
    private var finishCompletion: ((_ deliveredFinal: Bool) -> Void)?
    private var finishTimeoutWork: DispatchWorkItem?

    /// Begin streaming a fresh utterance. Resets counters and clears any
    /// stale state from the previous turn.
    func start(utteranceId: String) {
        self.utteranceId = utteranceId
        activeSegmentId = Self.segmentId(root: utteranceId, index: 0)
        segmentIndex = 0
        segmentStartedAt = Date()
        decodeSeq = 0
        firstChunkSentAt = nil
        firstPartialAt = nil
        didGiveUp = false
        consecutiveErrorCount = 0
        activeSegmentText = ""
        rollingSegmentId = nil
        rollTimeoutWork?.cancel()
        rollTimeoutWork = nil
        isFinishing = false
        finishCompletion = nil
        finishTimeoutWork?.cancel()
        finishTimeoutWork = nil
        inFlightCount = 0
        pendingDecode = nil
        pendingPCM.removeAll(keepingCapacity: true)
    }

    /// Push a freshly recorded PCM buffer into the streamer. Buffers smaller
    /// than the chunk budget accumulate; once the budget is reached we send
    /// the whole accumulated chunk as one decode request.
    func enqueue(pcm: Data) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in
                self?.enqueue(pcm: pcm)
            }
            return
        }
        guard utteranceId != nil, !didGiveUp, !isFinishing else { return }
        pendingPCM.append(pcm)
        if rollingSegmentId != nil {
            trimPendingIfNeeded()
            return
        }
        if pendingPCM.count >= minChunkBytes {
            flushPending(isFinal: false)
        } else if pendingPCM.count >= maxQueuedBytes {
            // Keep bounded memory if callbacks arrive faster than the
            // broker can accept queued chunks. The final WAV is kept in
            // DictationController.sessionPCMData; this buffer is preview-only.
            trimPendingIfNeeded()
        }
    }

    /// Close the current preview transport segment and start a fresh one while
    /// keeping the root dictation session alive. The broker maps segment-local
    /// ids and sequence numbers back onto one cumulative preview state, so the
    /// HUD keeps stable text across long dictations.
    func rollSegment(reason: String, timeout: TimeInterval = 0.45) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in
                self?.rollSegment(reason: reason, timeout: timeout)
            }
            return
        }
        guard let root = utteranceId,
              let segment = activeSegmentId,
              !didGiveUp,
              !isFinishing,
              rollingSegmentId == nil
        else { return }
        guard !activeSegmentText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingPCM.isEmpty else {
            return
        }
        rollingSegmentId = segment
        NSLog("Juno preview-streamer roll root=\(root) segment=\(segment) reason=\(reason)")
        flushPending(isFinal: true)
        if inFlightCount == 0 && pendingDecode == nil {
            completeSegmentRoll(reason: reason)
            return
        }
        if timeout > 0 {
            rollTimeoutWork?.cancel()
            let work = DispatchWorkItem { [weak self] in
                self?.completeSegmentRoll(reason: "\(reason)_timeout")
            }
            rollTimeoutWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + timeout, execute: work)
        }
    }

    var activeSegmentAge: TimeInterval? {
        guard let segmentStartedAt else { return nil }
        return Date().timeIntervalSince(segmentStartedAt)
    }

    /// Send the final chunk for this utterance. After this call the
    /// streamer goes back to idle; further ``enqueue`` calls no-op.
    func finish(timeout: TimeInterval = 0.9, completion: ((_ deliveredFinal: Bool) -> Void)? = nil) {
        guard utteranceId != nil, !didGiveUp else {
            utteranceId = nil
            completion?(false)
            return
        }
        isFinishing = true
        finishCompletion = completion
        // Drain whatever is left, even if under the chunk budget.
        flushPending(isFinal: true)
        if inFlightCount == 0 && pendingDecode == nil {
            completeFinish(deliveredFinal: false)
            return
        }
        if timeout > 0 {
            finishTimeoutWork?.cancel()
            let work = DispatchWorkItem { [weak self] in
                self?.completeFinish(deliveredFinal: false)
            }
            finishTimeoutWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + timeout, execute: work)
        }
    }

    /// Abort the in-flight utterance silently — used on cancel/Esc. We do NOT
    /// POST a final chunk: the shell will rerun the full utterance through
    /// ``ingest_wav`` on cancel when needed.
    func cancel(reason: String) {
        let uid = utteranceId
        utteranceId = nil
        activeSegmentId = nil
        didGiveUp = true
        consecutiveErrorCount = 0
        activeSegmentText = ""
        rollingSegmentId = nil
        rollTimeoutWork?.cancel()
        rollTimeoutWork = nil
        isFinishing = false
        finishCompletion = nil
        finishTimeoutWork?.cancel()
        finishTimeoutWork = nil
        inFlightCount = 0
        pendingDecode = nil
        pendingPCM.removeAll(keepingCapacity: true)
        if let uid {
            NSLog("Juno preview-streamer cancel uid=\(uid) reason=\(reason)")
        }
    }

    /// Mark the engine path as unusable for this utterance. Callers use
    /// this when the first-word budget expires before any partial arrived.
    func giveUpForFallback(reason: String) {
        guard let uid = utteranceId, !didGiveUp else { return }
        didGiveUp = true
        onGiveUp?(uid, reason)
    }

    // MARK: - HTTP

    private func flushPending(isFinal: Bool) {
        guard let root = utteranceId, let segment = activeSegmentId, !didGiveUp else { return }
        guard !pendingPCM.isEmpty || isFinal else { return }
        let chunk = pendingPCM
        pendingPCM.removeAll(keepingCapacity: true)
        sendOrCoalesce(rootUid: root, segmentUid: segment, data: chunk, isFinal: isFinal)
    }

    private func nextDecodeSeq() -> Int {
        let seq = decodeSeq
        decodeSeq += 1
        return seq
    }

    private func sendOrCoalesce(rootUid: String, segmentUid: String, data: Data, isFinal: Bool) {
        if inFlightCount >= maxInFlight {
            if var pending = pendingDecode, pending.rootUid == rootUid, pending.segmentUid == segmentUid {
                if !data.isEmpty {
                    pending.data.append(data)
                    pending.coalescedChunks += 1
                }
                pending.isFinal = pending.isFinal || isFinal
                pendingDecode = pending
            } else {
                pendingDecode = PendingDecode(
                    rootUid: rootUid,
                    segmentUid: segmentUid,
                    seq: nextDecodeSeq(),
                    data: data,
                    isFinal: isFinal,
                    createdAt: Date(),
                    coalescedChunks: data.isEmpty ? 0 : 1
                )
            }
            if let pending = pendingDecode {
                onBackpressure?(
                    rootUid,
                    "preview_coalesced_chunks_\(pending.coalescedChunks)_bytes_\(pending.data.count)"
                )
            }
            return
        }
        sendChunk(
            rootUid: rootUid,
            segmentUid: segmentUid,
            seq: nextDecodeSeq(),
            data: data,
            isFinal: isFinal,
            coalescedChunks: 0
        )
    }

    private func drainQueuedIfPossible() {
        guard !didGiveUp else { return }
        guard rollingSegmentId == nil || pendingDecode?.isFinal == true else { return }
        guard inFlightCount < maxInFlight, let next = pendingDecode else { return }
        pendingDecode = nil
        guard utteranceId == next.rootUid else { return }
        sendChunk(
            rootUid: next.rootUid,
            segmentUid: next.segmentUid,
            seq: next.seq,
            data: next.data,
            isFinal: next.isFinal,
            coalescedChunks: next.coalescedChunks
        )
    }

    private func sendChunk(
        rootUid: String,
        segmentUid: String,
        seq: Int,
        data chunk: Data,
        isFinal: Bool,
        coalescedChunks: Int
    ) {
        if firstChunkSentAt == nil { firstChunkSentAt = Date() }
        inFlightCount += 1
        let audioMs = (Double(chunk.count) / 2.0 / 16_000.0) * 1000.0
        var payload: [String: Any] = [
            "utterance_id": segmentUid,
            "root_utterance_id": rootUid,
            "root_final": isFinal && isFinishing,
            "audio_b64": chunk.base64EncodedString(),
            "sample_rate_hz": 16000,
            "decode_seq": seq,
            "is_final": isFinal,
            "coalesced_chunk_count": coalescedChunks,
            "coalesced_audio_ms": audioMs,
        ]
        let hint = (visibleTextHint?() ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !hint.isEmpty {
            payload["visible_text_hint"] = hint
        }
        let candidates = candidateEntities?() ?? []
        if !candidates.isEmpty {
            payload["candidate_entities"] = Array(candidates.prefix(24))
        }
        if let tape = sessionContextTape?(), !tape.isEmpty {
            payload["session_context_tape"] = tape
        }
        postJSON("api/broker/dictation/preview/chunk", payload) { [weak self] obj in
            guard let self else { return }
            self.inFlightCount = max(0, self.inFlightCount - 1)
            defer { self.drainQueuedIfPossible() }
            // Late responses for an utterance we already abandoned are
            // dropped silently — they'd otherwise fight whatever source
            // the HUD has now committed to.
            guard self.utteranceId == rootUid, !self.didGiveUp else { return }
            guard self.activeSegmentId == segmentUid
                    || self.rollingSegmentId == segmentUid
                    || self.isFinishing
            else { return }
            if let ok = obj["ok"] as? Bool, ok {
                self.consecutiveErrorCount = 0
                let committed = (obj["committed_text"] as? String) ?? ""
                let tail = (obj["tail_text"] as? String) ?? ""
                let combined = (obj["text"] as? String) ?? ""
                let previewMetadata = obj["preview_metadata"] as? [String: Any]
                let tailDisplaySuppressReason = ((obj["tail_display_suppress_reason"] as? String)
                    ?? (previewMetadata?["tail_display_suppress_reason"] as? String)
                    ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                // Do not render unstable ASR tail text in production. The
                // backend may keep tail internally for final-word promotion,
                // but user-visible HUD text is committed text only.
                let visibleTail = ""
                let effectiveCommitted = committed.isEmpty && tail.isEmpty
                    ? combined
                    : committed
                let text = Self.joinedText(effectiveCommitted, visibleTail)
                let final = (obj["is_final"] as? Bool) ?? false
                let decodeMs = (obj["decode_ms"] as? Double)
                    ?? (obj["decode_ms"] as? NSNumber)?.doubleValue
                    ?? 0
                let skipReason = (obj["filtered_reason"] as? String) ?? ""
                NSLog("Juno preview-streamer send.ok seq=\(seq) decode_ms=\(Int(decodeMs)) committed_chars=\(effectiveCommitted.count) tail_chars=\(tail.count) tail_hidden=\(!tail.isEmpty) tail_engine_suppressed=\(!tailDisplaySuppressReason.isEmpty) skip=\(skipReason)")
                if !final, decodeMs > 450 {
                    self.onBackpressure?(rootUid, String(format: "preview_decode_slow_%.0fms", decodeMs))
                }
                if !text.isEmpty, self.firstPartialAt == nil {
                    self.firstPartialAt = Date()
                }
                if !text.isEmpty {
                    self.activeSegmentText = text
                }
                self.onPartial?(rootUid, effectiveCommitted, visibleTail, final && self.isFinishing)
                if final && self.rollingSegmentId == segmentUid && !self.isFinishing {
                    self.completeSegmentRoll(reason: "final_response")
                } else if final {
                    self.completeFinish(deliveredFinal: true)
                }
            } else {
                let reason = (obj["error"] as? String) ?? "unknown"
                self.consecutiveErrorCount += 1
                NSLog("Juno preview-streamer send.error seq=\(seq) reason=\(reason) consecutive=\(self.consecutiveErrorCount)")
                if self.consecutiveErrorCount >= 4 {
                    self.giveUpForFallback(reason: "decode_error_x\(self.consecutiveErrorCount): \(reason)")
                }
                if isFinal {
                    self.completeFinish(deliveredFinal: false)
                }
            }
        }
    }

    private func completeFinish(deliveredFinal: Bool) {
        guard isFinishing || finishCompletion != nil else { return }
        finishTimeoutWork?.cancel()
        finishTimeoutWork = nil
        rollTimeoutWork?.cancel()
        rollTimeoutWork = nil
        let completion = finishCompletion
        finishCompletion = nil
        utteranceId = nil
        activeSegmentId = nil
        isFinishing = false
        rollingSegmentId = nil
        consecutiveErrorCount = 0
        activeSegmentText = ""
        pendingDecode = nil
        pendingPCM.removeAll(keepingCapacity: true)
        completion?(deliveredFinal)
    }

    private func completeSegmentRoll(reason: String) {
        guard let root = utteranceId, rollingSegmentId != nil, !isFinishing else { return }
        rollTimeoutWork?.cancel()
        rollTimeoutWork = nil
        activeSegmentText = ""
        segmentIndex += 1
        activeSegmentId = Self.segmentId(root: root, index: segmentIndex)
        segmentStartedAt = Date()
        decodeSeq = 0
        rollingSegmentId = nil
        pendingDecode = nil
        NSLog("Juno preview-streamer new-segment root=\(root) segment=\(activeSegmentId ?? "") reason=\(reason)")
        if pendingPCM.count >= minChunkBytes {
            flushPending(isFinal: false)
        } else {
            trimPendingIfNeeded()
        }
    }

    private func trimPendingIfNeeded() {
        if pendingPCM.count > maxQueuedBytes {
            pendingPCM.removeSubrange(0..<(pendingPCM.count - maxQueuedBytes))
        }
    }

    private static func segmentId(root: String, index: Int) -> String {
        "\(root)__preview\(index)"
    }

    private static func joinedText(_ prefix: String, _ current: String) -> String {
        let a = prefix.trimmingCharacters(in: .whitespacesAndNewlines)
        let b = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if a.isEmpty { return b }
        if b.isEmpty { return a }
        if b.hasPrefix(a) { return b }
        return "\(a) \(b)"
    }

}
