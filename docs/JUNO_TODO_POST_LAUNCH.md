# Juno post-launch TODO (scoped out of 2026-06-10 go-live by decision)

1. **Action operations + followups** (`JUNO_ACTIONS_OPERATIONS=0`,
   `JUNO_ACTIONS_FOLLOWUP=0` in run_engine.sh). Launch scope is creation
   only (note/reminder/alarm + compound). Re-enabling needs: extractor-lane
   routing for planner-emitted operations (wired, env-gated), reference
   resolution UX for ambiguous targets, and tests against a populated
   actions index.
2. **Long-text tone rewriting** (formal-email mode over 100+ word
   dictation). The editor lane fixes/structures but does not wholesale
   re-tone long text (decode cost). Needs streaming rewrite or a dedicated
   async surface.
3. **Speculative prefill during speech** — pre-fill the editor KV on the
   stable transcript prefix mid-utterance; cuts stop→paste by the suffix
   prefill time on long rambles.
4. **Rolling final transcript** — final-quality decode of windows during
   speech so stop-time Whisper cost is ~1s regardless of length (today:
   up to ~6s on 3-minute audio).
5. **Editor shadow telemetry review** — first week: tally
   `dictation_edit_generated` vs `dictation_edit_floor` reasons and edit
   acceptance quality from History; tighten the system prompt or model
   choice (verify current-gen small instruct models first).
6. **KV cache quantization** for the editor prefix if memory pressure
   appears on 16GB machines.
7. **L6 replay fixture set** — record curated fixtures (silent-S endings,
   got/God minimal pairs, 20-action batch, letter-lists) with retention on.
