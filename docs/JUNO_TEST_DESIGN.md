# Juno test design — covering the real use-case space

Goal: every class of silent failure observed in production (2026-06-10
forensics) and every "weird but normal" dictation behavior has a test that
fails before the bug ships. Juno is layered, so the tests are layered: each
behavior is pinned at the cheapest layer that can express it
deterministically, and the classes that genuinely need audio or model
judgment get a replay harness and telemetry watchdogs instead of wishful
unit tests.

## 1. Harness layers

| Layer | Harness | Determinism | What belongs here |
|---|---|---|---|
| L1 ITN / deterministic form | direct function calls (`ITNEngine.run`, `strip_fillers`, `normalize_plain_dictation`) | full | spoken punctuation, quotes, terminal ops, numbers, fillers |
| L2 Live preview lane | `StreamingPreviewSessionManager` + scripted decoders (`_ScriptedDecoder`, `_SequenceDecoder`) | full — synthetic hypotheses with controlled word timestamps | LocalAgreement commits, draft-horizon, tail promotion, replay/BoH guards |
| L3 Turn plan / actions | plan dicts through `normalize_turn_plan` → `validate_turn_plan` → `actions_from_turn_plan` → `render_turn_plan` | full — the model is simulated by handing in plans, including malformed ones | span grounding, body repair, count mismatches, compound batches, schedule parsing |
| L4 Writer service | `WriterService.process_transcript` with `_TurnPlanBackend` fakes | full | turn-plan outcome routing, text-delivery guarantee, memory mutations, transforms |
| L5 Pipeline e2e (no audio model) | `OneShotDictationPipeline` with `FakeTranscriber` | full | wake gating, action dispatch vs paste decisions, history persistence, retake application |
| L6 Replay (real audio, real models) | `juno_v2/engine/replay.py` + retained WAVs from `logs/service/workbench/audio` | semi — real ASR, fixed audio | ASR substitutions ("God/got"), silent-S truncations, boundary hallucinations with real acoustics |
| L7 Telemetry watchdogs | trace assertions over `service_live_*.jsonl` in dev builds | production-shaped | latency budgets, fallback-rate regressions, data-loss invariants |

## 2. Use-case matrix

| Use case (user-named) | Layer | Status | Test |
|---|---|---|---|
| Mid-utterance corrections ("scratch that", "no wait", numeric retakes) | L1/L5 | covered | `test_self_correction_retakes.py`; pipeline wiring via `oneshot_self_corrections_applied` |
| Corrections inside action bodies ("3pm scratch that 4.15pm") | L3 | covered | `test_qwen_turn_planner.py` (cue tests); `_latest_self_correction_tail` tightening |
| Correction-cue words as content ("the scratch that feature", "delete that file") | L1/L3 | covered | `test_self_correction_retakes.py` literal cases; cue regex guards |
| Announced count ≠ spoken count ("four points", three spoken) | L3 | covered | `test_weird_case_scenarios.py` — renders spoken items only, never invents |
| Unannounced structure (bare ordinals) | L3/model | contract pinned | deterministic stays out; ≤16-word utterances reach the model planner |
| 10–20 actions in one utterance | L3 | covered | 14-action compound coercion test (notes + reminder + alarm, schedules parsed) |
| Hallucinated actions (wake + no action verb) | L5 | covered | `test_pipeline_ignores_planner_actions_without_action_verb` |
| Action span corruption (model-retyped "e titled,") | L3 | covered | token-boundary `span_present`, `_snap_span_to_source`, body-repair tests |
| Action failure must never lose text | L4/L5 | covered | writer text-delivery guarantee + rejected-action history persistence tests |
| Wake word quoted mid-sentence | L5 | covered | `test_wake_word_quoted_mid_sentence_stays_dictation` |
| Random words committed mid-utterance ("I don't know" next to wake) | L2 | covered | `test_preview_commit_draft_horizon.py` — scripted truncated-zone hallucination |
| Trailing junk on cut-off words ("silent S") | L2 + L6 | partial | draft horizon covers the commit path; acoustic verification needs L6 replay fixtures (audio was destroyed by reinstall — record new fixtures) |
| Punctuation: cue-adjacent commas, "New paragraph, text" | L1 | covered | `test_itn_spoken_punctuation.py` |
| Spoken quotes, terminal colon/dash/ops exactness | L1 | covered | quote-pair + terminal-ops tests; terminal/code profiles now run spoken punctuation |
| Filler retention by app (keep "um" except where mode strips) | L1 | covered | conservative `strip_fillers` contract; mode-driven stripping stays in writer-mode tests |
| App-specific writing (modes, tone) | L4 | existing | mode tests in `test_qwen_turn_planner.py` / writer suites |
| Memory / screen-context term preservation | L2/L4 + L6 | partial | preview repair term tests exist; end-to-end "bias term survives to paste" needs L6 replay with seeded memory |
| ASR substitutions ("God"/"got") | L6 + repair lane | gap (architectural) | needs the windowed diff-repair lane (engineering-truth doc §3.3) + replay fixtures asserting repair |
| Paste latency budget | L7 | designed | assert `insertion_committed.shell_timeline` stop→paste ≤ 7s (long), ≤ 4s (short) in dev-build watchdog; alert on `turn_plan decode_ms` > 4s |
| Silent data loss (anything in, nothing out) | L5 + L7 | covered/designed | pipeline guarantees paste OR ≥1 dispatched action OR history transcript; L7 watchdog: any utterance with words>0, paste none, actions none, history transcript empty ⇒ red |

## 3. The L6 replay harness (the part that needs building next)

Deterministic tests cannot express acoustics. The replay harness closes that:

1. **Fixture capture**: dictate a curated script through the real app with
   audio retention on; copy WAVs + the trace out of
   `logs/service/workbench/` *before* any reinstall (the install script
   currently destroys them — change it to archive).
2. **Curated scripts** (one fixture each): trailing unvoiced consonants
   ("tests", "stops", "sixths"), wake phrase mid-flow, "got/God" minimal
   pair, fast numbers and times, the full bug-report-style long ramble,
   spoken quotes in terminal context, 14-action compound command.
3. **Assertions**: run `juno_v2.engine.replay` against the fixture set per
   PR; diff committed-lane text and final paste against golden transcripts
   with a tolerance list (allowed variants), and assert *invariants* rather
   than exact strings where ASR wobbles: no hallucination-blocklist phrase
   in committed text, no commit of words whose timestamps end inside the
   draft horizon, paste non-empty, actions grounded.
4. **Budget**: the fixture set must run < 5 min on an M-series laptop so it
   gates merges.

## 4. Invariants that must hold everywhere (encoded, not aspirational)

1. **No silent loss**: every utterance with speech ends in a paste, a
   dispatched action, or a recoverable History transcript. (L5 tests +
   L7 watchdog.)
2. **Wake is not consent**: planner actions require a deterministic action
   signal in the spoken text. (L5.)
3. **Spans are offsets, not retypes**: any model-provided span must ground
   at token boundaries in the source, or be snapped/repaired/rejected. (L3.)
4. **Committed HUD text is speech**: no synthesized status strings in the
   committed lane; no commits from the truncated-decode zone. (L2.)
5. **Markers are content until proven cues**: correction/quote/punctuation
   words convert only with unambiguous evidence. (L1.)
6. **The paste path stays fast**: no model decode on the paste path for
   long plain dictation; budget watchdogs on every stage. (L7.)

## 5. Known gaps, named honestly

- "God/got"-class ASR substitutions have no repair lane yet — that is the
  windowed diff-repair design in `JUNO_ENGINEERING_TRUTH_2026-06-10.md` §3.3.
  Until it lands, long-dictation pastes carry raw Whisper word choices.
- Mixed utterances ("take a note X and also write Y") currently suppress
  the paste entirely when actions dispatch (`_turn_plan_allows_mixed_paste`
  is hard-false); the dictation half is recoverable from History only.
- Operations on existing actions ("mark it done", "move it to 5pm") via the
  turn-plan lane reject as unsupported; the legacy followup lane covers
  only the most recent reminder within 30s.
- L6 fixtures do not exist yet because the 09:51 reinstall destroyed the
  session audio; capture a fresh fixture set before the next bundle build.
