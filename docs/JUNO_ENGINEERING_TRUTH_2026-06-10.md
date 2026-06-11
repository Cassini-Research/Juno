# Juno engineering truth — session forensics, 2026-06-10

Source of evidence: the full broker trace of the post-install session
(`service_live_20260610_092412_4f715fdf.jsonl`, 3,061 events, 6 dictation
turns, 09:24–09:35), the surviving `product_history.sqlite` (18 utterances
incl. the scripted e2e batch from 00:31–00:41), and the production code as of
this working tree. Every claim below is traced to one of those; nothing is
inferred from docs or plans. The raw trace and audio were destroyed at 09:51
by a fresh install — key excerpts are reproduced inline here.

The six real turns analyzed (trace IDs abbreviated):

| # | id | what was said | outcome |
|---|----|----|----|
| U1 | `0C081332` | "Hey Juno, take a note titled 'What is Juno…'" (~28s) | note executed, body corrupted, HUD froze at `Hey Juno: note`, 16.9s to action |
| U2 | `D50D8809` | (3.5s, near-silence) | correctly rejected (`oneshot_silence_hallucination_rejected`, raw="!") |
| U3 | `1B00E830` | "Hey Juno, …" test phrase (~10s) | hallucinated **note** action created from non-note speech, paste suppressed, 26.5s |
| U4 | `8DE4DF07` | 71-word dictation | pasted after **26.0s** |
| U5 | `DFBC4F66` | 121-word bug report | pasted after **40.5s**; planner JSON invalid; "God" survived |
| U6 | `505EF20F` | 171-word bug report | pasted after **53.9s**; "Scratch that" detected but not applied |

---

## 1. Architecture as actually built (traced from entry points)

```
mic (Swift shell)
 └─ audio chunks → broker (juno_v2.workbench.server)
     ├─ PREVIEW LANE (resident MLX Whisper large-v3-turbo, separate process)
     │   rolling ~8–10s windows (preview0..N per utterance)
     │   LocalAgreement-2 commit + tail  (juno_v2/preview/live_agreement.py,
     │   streaming_core.py) → personalization repair → orthography
     │   → committed_text/tail_text → HUD
     ├─ LIVE CHECKPOINT LANE (per window rollover)
     │   "oneshot" transcription (live_preview_hint = reuse preview text)
     │   → Qwen3-0.6B live adjudication (max 160 tok)
     │   → live transcript_decision (feeds action hint, not the paste)
     └─ FINAL LANE (on hotkey stop, fully serialized on the paste path):
         1. full-audio Whisper decode            (0.4–5.8s observed)
         2. ITN spoken-punctuation               (ms)
         3. Qwen3-0.6B FINAL adjudication        (up to 19.3s, usually discarded)
         4. Qwen3-4B turn planner (turn_plan_v1) (7.9–28.0s, + repair retry 10–13s)
         5. writer resolution / actions / format (ms–s)
         6. paste or action dispatch
```

The Qwen backend work replaced the old final/writer arrangement with two new
components: the **turn planner** (`juno_v2/turn_plan/`, Qwen3-4B-Instruct-2507-4bit
via mlx_lm, `on_demand` residency, idle TTL was 30s) and the **live corrector**
(Qwen3-0.6B-4bit, resident) which also serves as the final adjudicator.
`juno_v2/final/backends/qwen_asr.py` was deleted; ASR is Whisper end-to-end.

The turn planner re-emits the *entire transcript inside JSON* (corrected
transcript object + content units + actions, observed 5,158 output chars for a
976-char utterance), so its decode time scales ~2× with utterance length. It
ran on **every** utterance, wake or not, on the paste critical path.

---

## 2. Symptom → root cause (all verified)

### 2.1 The colon + the HUD "stops working" — FIXED
`juno_v2/workbench/server.py` `_action_preview_display_text()` builds
`f"Hey Juno: {', '.join(kinds)}"` whenever the preview text matches the wake
regex, then **replaced the committed HUD text with it and blanked the tail**
for the rest of the utterance. Trace U1, decode 13: raw Whisper =
`"Hey Juno, take a note titled, what is Juno?"`, committed →
`"Hey Juno: note"` (14 chars), then `preview_committed_chars` stayed 14 for
the remaining ~25s while you spoke. Same in U3 (frozen at `"Hey Juno"`).
The synthetic string also leaked into the live transcript hint
(`oneshot_final_transcription_result backend=live_preview_hint
raw_text="Hey Juno: note"`), corrupting the live action-source lane.

**Fix applied**: the takeover is removed; wake utterances stream as normal
dictation; detected kinds remain as display metadata only
(`display_override.display_only=true`). The native Apple app icon on the
action result chip was separately verified healthy — `com.apple.Notes`,
`com.apple.reminders`, `com.apple.iCal` all resolve via LaunchServices on
this machine, so `JunoActionNativeIcon` renders the real icons.

### 2.2 "I don't know" committed next to "Hey Juno" — FIXED (same day, PM round)
Fix: commit draft-horizon guard in `juno_v2/preview/streaming_core.py` —
agreed words ending inside the last 600ms of buffered audio
(`JUNO_V2_PREVIEW_COMMIT_DRAFT_HORIZON_MS`, enabled in `run_engine.sh`) are
demoted to the tail until more audio confirms or corrects them. Scripted
reproduction of this exact failure in
`tests/test_preview_commit_draft_horizon.py`. Original analysis below.
During U5 you said "…the moment it listens to Hey Juno, it stops working."
At the preview2→preview3 window boundary (t≈332.6s) the rolling window ended
right after "hey Juno it sto—". Whisper, decoding a truncated window,
completed it as `"hey Juno I don't know."` and the **window-final tail
promotion** committed it. Every subsequent live decision still contained
`"…listens to hey Juno I don't know. it stops working…"` — committed text is
immutable in the live lane, so the hallucination never healed on screen. The
final full-audio decode produced the correct text, so it did not reach the
paste — this class is HUD-only, but it is exactly what you watched happen.

Mechanism amplified by this round's changes in
`juno_v2/preview/streaming_core.py` / `live_agreement.py`:
`_MAX_FINAL_TAIL_PROMOTION_WORDS = 12`,
`_MAX_FINAL_EMPTY_COMMITTED_TAIL_PROMOTION_WORDS = 32` (new), and the new
`confirmed_budget_tail` path that promotes a quarantined single-word tail at
window finals. Up to 12 *unconfirmed* words can be committed at every window
rollover with no second agreeing decode.

### 2.3 "God" in the final text — analyzed, not fixed
U5 final Whisper decode transcribed your "…'I don't know' **got** committed"
as `"I don't know God committed"` — a plain acoustic substitution, capital-G.
It survived to the paste because the repair stage that should have caught it
never ran: the turn planner returned `invalid_json` after 10.6s, the repair
retry also returned `invalid_json`, and the writer fell back to
`pass_through_commit` (raw Whisper text + deterministic formatting only).
Long dictation **never** gets model repair in this architecture — the legacy
writer rewrite also declines on length. So: ASR substitution + no repair lane
for exactly the utterances most likely to contain errors.

### 2.4 Latency (15–54s stop→paste) — FIXED (first cut)
Measured breakdown, U6 (171 words): stop → Whisper 5.8s → final 0.6B
adjudication **19.3s, output then discarded**
(`unsupported_output_phrase:full stops` → fallback) → 4B turn plan **28.0s**
(plan validated, then *ignored*: `turn_plan_text_commit_policy_ignored`,
pasted text was pass-through anyway) → paste at **53.9s**. U5: 40.5s
(planner produced nothing usable). U4: 26.0s. Even the scripted e2e batch at
00:31–00:41 averaged 15–31s per utterance. Across **all six** real turns the
4B planner contributed zero pasted characters; its only outputs were two note
actions (one corrupted, one hallucinated).

Contributing: writer `idle_unload_ttl_s` was 30s — real pauses between
utterances exceed that, so nearly every turn also paid the 3–5s Qwen3-4B
cold start.

**Fixes applied** (all env-overridable):
- `juno_v2/writer/service.py` + `config.py`: model turn planner now skips
  **long plain dictation** (>16 words, no wake, no selection, no explicit
  spoken structure request). Deterministic structural plans (lists/checklists)
  still render at zero model cost; short commands, selection transforms,
  memory teaches, and explicit "note down N points…" requests keep the model.
  (`JUNO_V2_TURN_PLAN_DICTATION=1` restores the old behavior;
  `JUNO_V2_TURN_PLAN_MAX_DICTATION_WORDS` tunes the cutoff.)
- `juno_v2/transcript/adjudicator.py`: final-stage adjudication input capped
  at 60 words (`JUNO_V2_FINAL_ADJUDICATION_MAX_WORDS`, 0 disables). Past the
  cap the final keeps the Whisper text instead of stalling 14–19s behind a
  decode that was being discarded.
- `juno_v2/writer/config.py`: `idle_unload_ttl_s` 30s → 300s.

Expected effect on the observed session: U4 26.0s → ~3–4s, U5 40.5s → ~4–5s,
U6 53.9s → ~7s (Whisper decode dominates). Wake/action turns still pay the
planner (see §3.2). All 176 tests pass.

### 2.5 Actions flow "breaking horribly" — FIXED (same day, PM round)
Fixes: (1) token-boundary span grounding + `_snap_span_to_source` +
note-instruction stripping kill the "e titled," class; (2) planner actions
require a deterministic action verb in the spoken text
(`turn_plan_actions_ignored_without_verb`) so wake alone can no longer
reroute dictation into Notes; (3) the writer's `turn_plan_action_only`
suppressing NOOP is gone — text delivery is guaranteed when no action
dispatches; (4) rejected action attempts keep the transcript in History
(`failure_reason=action_rejected` with recoverable text). Regression tests in
`tests/test_qwen_turn_planner.py`. Original analysis below.
Four distinct, confirmed defects:
1. **Corrupted note body**: the executed Apple Note for U1 begins
   `"e titled, what is Juno and why does it exist? …"` — the planner's
   evidence-span repair (`action_body_repaired_from_evidence`) sliced
   "take a not|e titled" mid-word. Span math, not ASR.
2. **Hallucinated action**: U3 ("Hey Juno, I did not reinstall…") contains no
   note verb, but wake ⇒ the 4B was pushed to produce an action and invented
   `kind=note` with the whole sentence as body. Your dictation was then
   **paste-suppressed** — the text went to Notes instead of your editor.
3. **Silent data loss**: history shows two utterances
   (`macshell-3F27833F`, `codex_e2e_formal_mail_cleanup`) with
   `failure_reason=turn_plan_action_only`, `actions=None`,
   `transcript=None` — the plan said "action only", action coercion then
   produced zero actions, and nothing was pasted and nothing executed.
   18–31s of processing, total output: nothing.
4. **Latency**: 16.9–26.5s from stop to the action firing, during which the
   old HUD showed only the frozen status string.

### 2.6 Punctuation — largely FIXED (same day, PM round)
Fixes: newline/terminal cues consume ASR-attached punctuation (no more
"\n\n, text"); determiner guard stops "the new paragraph is short" from
converting; spoken quote pairs convert ("quote fix colon … quote" →
'"fix: …"'); spoken punctuation now runs in TERMINAL/CODE profiles;
terminal double-operator ordering fixed ("double ampersand" → "&&").
"Scratch that" retakes now apply deterministically
(`juno_core_v3/dictation/self_corrections.py`). Remaining: baseline comma
quality of raw Whisper on long dictation (needs the repair lane, §3.3).
Tests: `tests/test_itn_spoken_punctuation.py`,
`tests/test_self_correction_retakes.py`. Original analysis below.
- Reproduced deterministically: ITN newline cue does not consume
  Whisper-attached punctuation —
  `"…exist? New paragraph, text is still…"` → `"…exist?\n\n, text is still…"`
  (rule `(r"new\s+paragraph", "\n\n", "newline")` in `juno_v2/itn/rules.py`;
  the comma Whisper glued to the cue survives as a paragraph-leading ", ").
  This corrupted U1's note body and U6's paste.
- "quote … quote" is not converted at all
  (`codex_e2e_terminal_code_exactness` pasted the literal word "quote").
- Baseline comma/period quality is raw Whisper-turbo punctuation, because the
  repair lane (planner/writer) never actually runs on real dictation (§2.3).

### 2.7 "Silent S" / trailing-word junk — commit path FIXED (same day, PM round)
The draft-horizon guard (§2.2 fix) covers this class on the commit path:
trailing words in the truncated-decode zone can no longer commit until more
audio confirms them. Acoustic verification of specific words still needs
replay fixtures with retained audio (docs/JUNO_TEST_DESIGN.md §3). Original
analysis below.
Same class as §2.2: end-of-window and end-of-utterance promotions of
low-evidence tail words. Observed artifacts in this session: committed live
text ending `"don'"` (U4), `"at UD. UI Interactions."` (U3 final),
`"it's listens"` (U5). When a word ends in an unvoiced consonant cluster and
the window cuts there, Whisper completes the truncated audio with a plausible
continuation, and the relaxed tail-promotion rules commit it. The audio files
that would let us verify specific words were deleted by the 09:51 reinstall —
re-verify with retained audio after the next session (audio is kept under
`logs/service/workbench/audio` until reinstall).

---

## 3. What has to change for Juno to work properly (exact guidance)

### 3.1 Make the paste path model-free; move meaning off-path
Target: stop→paste ≤3s at p95.
- Paste path = full-audio Whisper + ITN + deterministic formatting only.
  (After today's gates this is true for plain dictation >16 words; make it
  true universally once 3.3 lands.)
- Anything model-driven (action extraction, transforms, punctuation repair)
  runs **async after paste-or-suppress is decided**, surfacing in the HUD
  chip when done. Actions do not need the text paste to wait for them.

### 3.2 Replace the "one 4B JSON mega-plan per utterance" design
`turn_plan_v1` makes the model restate the entire transcript inside JSON.
That is the direct cause of: invalid_json on long inputs, 8–28s decodes,
"e titled," span corruption, and grounding failures. Replace with:
1. **Intent gate** (deterministic wake + verb grammar, plus the resident
   0.6B as tie-breaker, ≤300ms) — decides dictation vs action vs transform.
2. **Action argument extraction on the post-wake span only**, with a
   *minimal* schema (kind, title, body_span_offsets, when) and ≤256-token
   constrained output. Spans are **character offsets into the source text**,
   never re-typed text — the engine slices the body itself, which makes the
   "e titled," class impossible.
3. **Never drop text**: if action coercion yields zero actions, paste the
   transcript (kill the `turn_plan_action_only` data-loss path in
   `juno_v2/writer/service.py` by requiring `action_count > 0` before
   returning the suppressing NOOP outcome).
4. Keep `JUNO_ACTIONS_*` validation, but validation failure must demote to
   dictation, not to nothing.

### 3.3 Model strategy (the bold part)
- **ASR**: keep `mlx-community/whisper-large-v3-turbo` for preview + final;
  5.8s for 166s of audio is acceptable and it is not the latency problem.
  The "God/got" class wants a *contextual rescoring/repair* step, not a
  bigger ASR.
- **Planner/extractor**: keep Qwen3-4B-Instruct-2507-4bit **only off the
  paste path**, resident (`residency_policy=resident` in `run_engine.sh`,
  2.6GB is within the 12GB budget) with mlx_lm **prompt caching** — the
  system prompt is multi-thousand tokens and is currently re-prefilled every
  call. With a cached prefix and §3.2's small outputs, action extraction is
  1–3s.
- **Do not** put a reasoning model on the paste path — thinking tokens are
  pure added latency there. If reasoning-class quality is wanted for complex
  compound actions, run it async behind the instant chip ("Working…" →
  result), where 5–10s is acceptable.
- **Repair lane for dictation** (fixes "God", commas): resident Qwen3-0.6B
  (already loaded) doing *windowed diff-repair*: feed ≤200-char windows
  around low-confidence ASR words (avg_logprob is already in the trace),
  constrained to emit either `KEEP` or a ≤8-token replacement. Bounded cost
  (~hundreds of ms), no transcript re-emission, no "full stops" blow-ups —
  the current adjudicator fails precisely because it re-emits everything.

### 3.4 Live lane (HUD trust)
- Window-boundary commits require cross-window agreement: the first decode of
  window N+1 must confirm the promoted tail of window N before it is
  display-committed; otherwise hold it in the tail style.
- Cap final tail promotion to ≤4 words and only those with word-level
  timestamps fully inside voiced audio (VAD), reverting this round's
  32-word empty-committed promotion.
- Wake-adjacency quarantine: after a committed wake phrase, the next tail
  needs two agreeing decodes (this is where "I don't know" landed).

### 3.5 Punctuation
- ITN: newline/punct cue replacement must consume adjacent punctuation the
  ASR attached to the cue (fixes `"\n\n, text"` — one-line rule change plus
  cases for `. New paragraph` / `New paragraph,` / `new line.`).
- Convert "quote … unquote/quote" pairs deterministically.
- Sentence-boundary cleanup (orphan commas before terminal punctuation,
  double punctuation) as a deterministic final-formatter pass — these are
  form, not meaning, and need no model.

### 3.6 Process/telemetry
- Fresh-install must archive, not delete, `logs/service/workbench` (this
  session's audio evidence was destroyed mid-analysis).
- `insertion_committed.shell_timeline` already carries stage timestamps —
  add planner/adjudication spans and alert in dev builds when stop→paste
  exceeds budget.
- The HUD-takeover behavior had dedicated unit tests *asserting the takeover*
  — tests encoded the wrong contract. Contract-level tests should assert the
  user-visible invariant instead: committed HUD text is always a prefix-stable
  transcript of speech, never a synthesized status string.

---

## 4. Changes applied in this session (scope: action icon/colon + loading)

| file | change |
|---|---|
| `juno_v2/workbench/server.py` | Wake-status takeover of HUD committed/tail text removed; status is metadata-only now |
| `juno_v2/writer/service.py` | Model turn planner gated for long plain dictation; deterministic structural plan substitutes; trace event `turn_plan_skipped` |
| `juno_v2/writer/config.py` | `turn_plan_dictation_enabled` + `turn_plan_max_dictation_words` (env-overridable); writer idle TTL 30s→300s |
| `juno_v2/transcript/adjudicator.py` | Final-stage adjudication word cap (default 60, env `JUNO_V2_FINAL_ADJUDICATION_MAX_WORDS`) |
| `juno_v2/turn_plan/planner.py` | `structural_instruction_present()` exposed for the gate |

Verification: full test suite `tests/` — 176 passed at the time of the AM
round. The installed app still runs the pre-fix bundle; rebuild + reinstall
via `./scripts/fresh_juno_macos_environment.sh` to pick these up.

## 5. PM round (same day) — defect fixes on top of the analysis

| area | change | tests |
|---|---|---|
| Span grounding | `span_present` / `_span_contains` token-boundary; `_snap_span_to_source` recovers truncated retyped spans; note-instruction stripping consumes "titled," | `test_qwen_turn_planner.py` span section |
| Action safety | verb-gated planner actions; writer text-delivery guarantee (no `turn_plan_action_only` black hole); rejected attempts keep transcript in History | `test_qwen_turn_planner.py` action-lane section |
| Correction cues | `_SELF_CORRECTION_CUE_RE` disambiguated (bare "actually"/"make it" are prose); deterministic retake application wired into the pipeline (`juno_core_v3/dictation/self_corrections.py`) | `test_self_correction_retakes.py` |
| ITN | cue-adjacent punctuation consumption; determiner literal-mention guard; spoken quote pairs; terminal/code profiles run spoken punctuation; terminal op ordering | `test_itn_spoken_punctuation.py` |
| Live lane | commit draft-horizon guard (600ms, env-gated, on in prod launcher) + telemetry counter | `test_preview_commit_draft_horizon.py` |
| Residency | writer back to `resident` in `run_engine.sh` (May decision restored) | — |
| Scenario coverage | weird-case suite: count mismatches, 14-action compound, mid-sentence wake, filler policy, profile routing | `test_weird_case_scenarios.py` |

Test design for the full use-case space: `docs/JUNO_TEST_DESIGN.md`.
