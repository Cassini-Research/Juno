# LiveJuno Demo Script

This is a high-confidence demo script based on the current unit tests and in-app
action examples. Say each numbered line as a separate dictation unless the row
explicitly says it is setup text.

For the most reliable live demo, use Apple Notes or TextEdit for dictation demos.
For action demos, enable Juno Actions and make sure Notes, Reminders, and
Calendar permissions are already granted.

## Quick Demo Run

Use this sequence for a short live demo.

1. Spoken punctuation
   - Say: `Hello comma world full stop`
   - Expected: `Hello, world.`
   - Test anchor: `test_spoken_punctuation_preserves_literal_mentions_but_converts_commands`

2. Structured bullets
   - Say: `Create three bullets. First point is Passport. Second point is Charger. Third point is Korea, customer meeting location.`
   - Expected:
     ```text
     Create three bullets:
     1. Passport.
     2. Charger.
     3. Korea, customer meeting location.
     ```
   - Test anchor: `test_spoken_bullet_points_strip_ordinal_item_labels`

3. Unpunctuated bullet structure
   - Say: `Create 3 bullets first point is Passport Second point is Charger Third point is Korea customer meeting location`
   - Expected:
     ```text
     Create 3 bullets:
     1. Passport.
     2. Charger.
     3. Korea customer meeting location.
     ```
   - Test anchor: `test_unpunctuated_spoken_bullets_render_from_real_audio_shape`

4. Context and protected terms
   - Setup: open or title a note with `Project Atlas` and `Silvia Gamache`.
   - Say: `The screen says Silvia Gamache next to Project Atlas.`
   - Expected: those terms should stay intact.
   - Test anchor: `test_post_asr_context_enrichment_is_visible_in_qwen_packet`

5. Product-name repair
   - Say: `First section is Luma Ray battery risk.`
   - Expected: `First section is LumaRay battery risk.`
   - Test anchor: `test_split_candidate_repair_does_not_absorb_function_words`

6. Self-correction
   - Say: `Japan, no actually Korea, is the customer meeting location.`
   - Expected: final meaning should resolve to Korea, not Japan.
   - Test anchor: `test_transcript_adjudication_prompt_includes_slot_correction_example`

7. Self-correction plus screen-name repair
   - Setup: keep `Silvia Gamache` visible in the window title or note body.
   - Say: `Start a note. The visible screen says Sylvia Gamache. Japan, no actually Korea, is the customer meeting location.`
   - Expected: `Start a note. The visible screen says Silvia Gamache. Korea is the customer meeting location.`
   - Test anchor: `test_oneshot_salvages_qwen_self_correction_after_protected_phrase_repair`

8. Reminder action
   - Say: `Hey Juno remind me to call Sam tomorrow at five.`
   - Expected: Juno routes this as a reminder action and uses the current local time for `tomorrow at five`.
   - Test anchor: `test_pipeline_passes_current_time_into_action_detection`

9. Note action
   - Say: `Juno, take a note about the Q3 plan.`
   - Expected: Juno routes this to Apple Notes and saves it in the Juno folder.
   - Source anchor: in-app action catalog and `actions_intent_v3` allowed action kinds.

10. Alarm action
    - Say: `Juno, wake me up in 25 minutes.`
    - Expected: Juno routes this to a calendar alert alarm.
    - Source anchor: in-app action catalog and `actions_intent_v3` allowed action kinds.

## Longer Dictation Set

Use these if you want to show more range.

### Plain Dictation

1. Say: `Decision log includes keeping Qwen for final polish.`
   - Expected: preserves `Decision log` and `Qwen`.
   - Test anchor: `test_transcript_validation_allows_duplicate_protected_term_cleanup`

2. Say: `Do not invent docs. doggs, thank you, or send.`
   - Expected: stays lowercase in the middle of the sentence.
   - Test anchor: `test_transcript_adjudicator_repairs_low_signal_mid_sentence_caps_before_validation`

3. Say: `Not every pause is a full stop and the words full stop should stay as text.`
   - Expected: keeps the literal phrase `full stop`.
   - Test anchor: `test_spoken_punctuation_preserves_literal_mentions_but_converts_commands`

### Names, Memory, And Context

1. Say: `SilviaGamachi owns Project Atlas.`
   - Expected: preserves both `SilviaGamachi` and `Project Atlas`.
   - Test anchor: `test_final_formatting_prompt_carries_preservation_terms_to_qwen`

2. Say: `SilviaGamachi owns the Project Atlas review.`
   - Expected: preview candidates include `SilviaGamachi` and `Project Atlas` when visible in screen context.
   - Test anchor: `test_preview_candidates_include_session_context_tape_screen_terms`

3. Say: `The visible screen says Sylvia Gamache.`
   - Expected: repairs near miss to `The visible screen says Silvia Gamache.`
   - Test anchor: `test_protected_phrase_token_repair_handles_screen_name_near_miss`

4. Say: `LumaRay battery risk is assigned.`
   - Expected: preserves `LumaRay`; after insertion, Juno can learn this context term.
   - Test anchor: `test_seed_memory_observes_context_terms_only_after_successful_commit`

5. Say: `Write this as a concise status update. The launch is on track and the risk owner is Nilofar.`
   - Expected: triggers explicit rewrite formatting in docs surfaces and preserves `Nilofar`.
   - Test anchor: `test_explicit_rewrite_promotes_default_docs_without_selection_transform`

### Structure And Formatting

1. Say: `Start research notes. We need four sections. First battery risk. Second rollout. Third launch metric.`
   - Expected: in docs surfaces, promotes to structured notes.
   - Test anchor: `test_spoken_structure_promotes_default_docs_to_structured_notes`

2. Say: `Second section is decisions, no actually decision log.`
   - Expected: resolves the correction to `Second section is decision log.`
   - Test anchor: `test_transcript_validation_allows_no_actually_noun_correction`

3. Say: `First section is risks scratch that make it open risks. At the end say the final word is complete.`
   - Expected: preserves the explicit final tail: `At the end say the final word is complete.`
   - Test anchor: `test_transcript_adjudicator_restores_explicit_final_word_tail`

4. Say: `Do not turn this into bullets. First run pytest. Second run git diff.`
   - Expected: stays minimal / paragraph style, especially in Terminal or when no-bullets is explicit.
   - Test anchor: `test_spoken_structure_does_not_promote_terminal_or_explicit_no_formatting`

### Edit Commands

Use these after selecting text or after Juno has a recent paste.

1. Setup selection text:
   ```text
   This is a long selected paragraph that should be rewritten.
   ```
   Say: `make that shorter`
   - Expected: Juno transforms the selected text instead of inserting the command text.
   - Test anchor: `test_selected_transform_command_does_not_fall_into_final_formatting`

2. Setup recent clipboard text:
   ```text
   This is the recent Juno paste that should be rewritten.
   ```
   Say: `make that text more clear`
   - Expected: Juno transforms the recent clipboard text.
   - Test anchor: `test_recent_transform_command_uses_recent_clipboard_in_default_mode`

3. Say: `make that shorter and more direct`
   - Expected: maps to the deterministic recent-edit instruction: make concise and direct while preserving meaning.
   - Test anchor: `test_recent_transform_command_grammar_covers_natural_variants`

### Voice Actions

Run these only when action permissions are granted.

1. Reminder
   - Say: `Hey Juno remind me to call Sam tomorrow at five.`
   - Expected: reminder action with a parsed due time.

2. Note
   - Say: `Hey Juno, take a note: action items from the standup are review Project Atlas, confirm the Q3 plan, and follow up with Sam.`
   - Expected: saved note in Apple Notes under the Juno folder.

3. Short note
   - Say: `Juno, note that the new API key rotates on Mondays.`
   - Expected: saved note in Apple Notes.

4. Alarm
   - Say: `Juno, wake me up in 25 minutes.`
   - Expected: calendar alert alarm.

5. Absolute alarm
   - Say: `Hey Juno, set an alarm for 7 AM tomorrow.`
   - Expected: calendar alert alarm at the requested time.

## Demo Notes

- Keep the microphone close and speak the action words clearly: `Hey Juno`, `take a note`, `remind me`, `wake me up`.
- For context demos, put the important terms visibly on screen before dictating.
- For memory demos, use exact known terms first, such as `LumaRay`, `Project Atlas`, `Silvia Gamache`, and `Nilofar`.
- If you want to demonstrate the memory-repair path for aliases like `Niloufar` or `Nilefer`, first make sure the memory entry for `Nilofar` includes those aliases.
- Avoid chaining many demos into one long utterance. The tests exercise these as focused utterances, and the live app is easiest to demo the same way.
