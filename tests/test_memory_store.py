"""Unit tests for juno_v2/memory/store.py JsonMemoryStore.

clear_all is covered by tests/test_memory_clear.py — not duplicated here.
"""

from __future__ import annotations

import pytest

from juno_v2.memory.store import JsonMemoryStore


# --------------------------------------------------------------------- #
# Boot seeding
# --------------------------------------------------------------------- #


def test_fresh_store_seeds_protected_vocabulary(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    snapshot = memory.snapshot()
    assert [entry.canonical_form for entry in snapshot.lexicon] == ["Juno"]
    assert snapshot.lexicon[0].source == "builtin"


def test_protected_seed_is_not_resurrected_after_user_deletion(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    memory = JsonMemoryStore(memory_dir)
    assert memory.remove_lexicon_entry("Juno")
    # A fresh boot of the same directory must keep the deletion sticky.
    reloaded = JsonMemoryStore(memory_dir)
    assert reloaded.snapshot().lexicon == []


# --------------------------------------------------------------------- #
# add_lexicon_entry
# --------------------------------------------------------------------- #


def test_add_lexicon_entry_basic(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(term="Qwen", canonical_form="Qwen", boost=2.0)
    entries = {e.canonical_form: e for e in memory.snapshot().lexicon}
    assert "Qwen" in entries
    assert entries["Qwen"].boost == 2.0


def test_add_lexicon_entry_dedupes_by_fold_key(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(term="polka foundation", canonical_form="Polkafoundation")
    memory.add_lexicon_entry(
        term="Polka-Foundation",
        canonical_form="Polkafoundation",
        aliases=["POLKA FOUNDATION"],
        boost=3.0,
    )
    rows = [e for e in memory.snapshot().lexicon if e.canonical_form == "Polkafoundation"]
    assert len(rows) == 1
    # Merge keeps the maximum boost seen.
    assert rows[0].boost == 3.0


def test_add_lexicon_entry_alias_dedup_against_canonical(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(
        term="Polkafoundation",
        canonical_form="Polkafoundation",
        aliases=["polka foundation", "Polka-Foundation", "POLKAFOUNDATION"],
    )
    rows = [e for e in memory.snapshot().lexicon if e.canonical_form == "Polkafoundation"]
    assert len(rows) == 1
    # All aliases fold to the canonical key, so none survive as aliases.
    assert rows[0].aliases == []


def test_add_lexicon_entry_rejects_short_terms(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(term="ab")
    memory.add_lexicon_entry(term="!!")
    memory.add_lexicon_entry(term="")
    canon = [e.canonical_form for e in memory.snapshot().lexicon]
    assert canon == ["Juno"]  # only the protected seed survives


def test_add_lexicon_entry_accepts_unicode_names(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(term="José")
    memory.add_lexicon_entry(term="日本語")
    canon = {e.canonical_form for e in memory.snapshot().lexicon}
    assert "José" in canon
    assert "日本語" in canon


def test_remove_lexicon_entry_matches_fold_variants(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(term="Sign-Off", canonical_form="Sign-Off")
    assert memory.remove_lexicon_entry("sign off")
    assert not memory.remove_lexicon_entry("sign off")  # already gone


# --------------------------------------------------------------------- #
# add_replacement
# --------------------------------------------------------------------- #


def test_add_replacement_basic(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_replacement(trigger="my email", replacement="me@example.com")
    rules = memory.snapshot().replacements
    assert len(rules) == 1
    assert rules[0].trigger == "my email"
    assert rules[0].replacement == "me@example.com"
    assert rules[0].scope == "global"


def test_add_replacement_dedupes_fold_variants_and_updates(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_replacement(trigger="my email", replacement="old@example.com")
    memory.add_replacement(trigger="My-Email", replacement="new@example.com")
    rules = memory.snapshot().replacements
    assert len(rules) == 1
    assert rules[0].replacement == "new@example.com"


def test_add_replacement_scopes_are_distinct_rows(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_replacement(trigger="sig", replacement="a", scope="global")
    memory.add_replacement(trigger="sig", replacement="b", scope="email")
    assert len(memory.snapshot().replacements) == 2


def test_add_replacement_case_sensitive_rows_are_distinct(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_replacement(trigger="Foo", replacement="x", case_sensitive=True)
    memory.add_replacement(trigger="foo", replacement="y", case_sensitive=False)
    assert len(memory.snapshot().replacements) == 2


def test_add_replacement_rejects_punctuation_only_trigger(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_replacement(trigger="!!!", replacement="boom")
    memory.add_replacement(trigger="   ", replacement="boom")
    memory.add_replacement(trigger="", replacement="boom")
    assert memory.snapshot().replacements == []


def test_remove_replacement_via_fold_key(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_replacement(trigger="signoff", replacement="Best,\nSam")
    assert memory.remove_replacement("Sign-Off")
    assert memory.snapshot().replacements == []
    assert not memory.remove_replacement("Sign-Off")


# --------------------------------------------------------------------- #
# record_correction
# --------------------------------------------------------------------- #


def test_record_correction_accepts_safe_pair(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert memory.record_correction("karvix", "juno")
    pairs = memory.snapshot().corrections
    assert len(pairs) == 1
    assert pairs[0].observed == "karvix"
    assert pairs[0].corrected == "juno"
    assert pairs[0].count == 1


def test_record_correction_dedup_increments_count_via_fold(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert memory.record_correction("u r", "you are")
    # Same intent, different surface punctuation/case — must merge.
    assert memory.record_correction("U.R", "You are")
    pairs = memory.snapshot().corrections
    assert len(pairs) == 1
    assert pairs[0].count == 2


def test_record_correction_rejects_identity_and_empty(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert not memory.record_correction("same text", "same text")
    assert not memory.record_correction("", "fixed")
    assert not memory.record_correction("observed", "")
    assert memory.snapshot().corrections == []


def test_record_correction_rejects_overlong_text(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert not memory.record_correction("x" * 200, "fixed thing")
    assert not memory.record_correction("observed thing", "y" * 200)


def test_record_correction_rejects_hallucinated_text(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert not memory.record_correction("thu thu thu thu thu", "thursday")


def test_record_correction_rejects_fragment_expansion(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    # Short fragment "corrected" into a much longer sentence is the user
    # rewriting, not a transcription fix.
    assert not memory.record_correction(
        "send it", "please send the quarterly report to the whole team today"
    )


def test_record_correction_rejects_low_signal_case_only_edit(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert not memory.record_correction("the live", "The live")


def test_remove_correction(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    assert memory.record_correction("karvix", "juno")
    assert memory.remove_correction("Karvix")
    assert memory.snapshot().corrections == []


# --------------------------------------------------------------------- #
# upsert_session_entities
# --------------------------------------------------------------------- #


def test_upsert_session_entities_inserts_and_counts(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.upsert_session_entities(["Karvix", "Acme Corp"])
    # Lowercase single tokens are not valid session entities under the
    # entity policy (proper-noun casing is the learn signal), so the
    # lowercase re-upsert is filtered and the original surface survives.
    memory.upsert_session_entities(["karvix"])
    entities = {e.value: e for e in memory.snapshot().session_entities}
    assert set(entities) == {"Karvix", "Acme Corp"}
    assert entities["Karvix"].count == 1
    assert entities["Acme Corp"].count == 1


def test_upsert_session_entities_skips_short_and_empty(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.upsert_session_entities(["ab", "", "   ", "!!"])
    assert memory.snapshot().session_entities == []


def test_upsert_session_entities_dedups_within_one_call(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    # The all-lowercase variant is filtered by the entity policy, so only
    # the cased surface inserts; nothing duplicates.
    memory.upsert_session_entities(["Polka-Foundation", "polka foundation"])
    entities = memory.snapshot().session_entities
    assert len(entities) == 1
    assert entities[0].value == "Polka-Foundation"
    assert entities[0].count == 1


# --------------------------------------------------------------------- #
# add_snippet / resolve_snippet / remove_snippet
# --------------------------------------------------------------------- #


def test_add_snippet_and_resolve_fold_variants(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_snippet(trigger="signoff", body="Best,\nSam")
    hit = memory.resolve_snippet("Sign Off")
    assert hit is not None
    assert hit.body == "Best,\nSam"


def test_add_snippet_scoped_resolution_prefers_scope_over_global(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_snippet(trigger="sig", body="global body", scope="global")
    memory.add_snippet(trigger="sig", body="email body", scope="email")
    assert memory.resolve_snippet("sig", scope="email").body == "email body"
    # Unrelated scope falls back to the global row.
    assert memory.resolve_snippet("sig", scope="code").body == "global body"


def test_add_snippet_dedupes_fold_variants_in_same_scope(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_snippet(trigger="signoff", body="v1")
    memory.add_snippet(trigger="Sign-Off", body="v2")
    assert len(memory.snippets.raw()) == 1
    assert memory.resolve_snippet("signoff").body == "v2"


def test_add_snippet_rejects_empty_trigger_or_body(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_snippet(trigger="", body="body")
    memory.add_snippet(trigger="sig", body="")
    memory.add_snippet(trigger="!!!", body="body")  # empty fold key
    assert memory.snippets.raw() == []


def test_remove_snippet(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_snippet(trigger="signoff", body="Best,\nSam")
    assert memory.remove_snippet("Sign-Off")
    assert memory.resolve_snippet("signoff") is None
    assert not memory.remove_snippet("signoff")


# --------------------------------------------------------------------- #
# Persistence across reload
# --------------------------------------------------------------------- #


def test_everything_persists_across_reload(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    first = JsonMemoryStore(memory_dir)
    first.add_lexicon_entry(term="Qwen", canonical_form="Qwen", boost=2.0)
    first.add_replacement(trigger="my email", replacement="me@example.com", scope="email")
    assert first.record_correction("karvix", "juno")
    assert first.record_correction("karvix", "juno")
    first.upsert_session_entities(["Acme Corp"])
    first.add_snippet(trigger="sig", body="Best,\nSam", scope="email", description="signature")

    second = JsonMemoryStore(memory_dir)
    snapshot = second.snapshot()

    assert {e.canonical_form for e in snapshot.lexicon} == {"Juno", "Qwen"}
    assert [(r.trigger, r.replacement, r.scope) for r in snapshot.replacements] == [
        ("my email", "me@example.com", "email")
    ]
    assert [(c.observed, c.corrected, c.count) for c in snapshot.corrections] == [
        ("karvix", "juno", 2)
    ]
    assert [(e.value, e.count) for e in snapshot.session_entities] == [("Acme Corp", 1)]
    hit = second.resolve_snippet("sig", scope="email")
    assert hit is not None
    assert hit.body == "Best,\nSam"
    assert hit.description == "signature"


def test_reload_does_not_duplicate_protected_seed(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    JsonMemoryStore(memory_dir)
    JsonMemoryStore(memory_dir)
    third = JsonMemoryStore(memory_dir)
    juno_rows = [e for e in third.snapshot().lexicon if e.canonical_form == "Juno"]
    assert len(juno_rows) == 1


def test_unicode_content_survives_reload(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    first = JsonMemoryStore(memory_dir)
    first.add_lexicon_entry(term="café", canonical_form="Café")
    first.add_lexicon_entry(term="日本語", canonical_form="日本語")
    second = JsonMemoryStore(memory_dir)
    snapshot = second.snapshot()
    canon = {e.canonical_form for e in snapshot.lexicon}
    assert "Café" in canon
    assert "日本語" in canon


def test_cjk_session_entities_are_stored(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.upsert_session_entities(["日本語クラス"])
    assert "日本語クラス" in {e.value for e in memory.snapshot().session_entities}


def test_cjk_session_entities_dedupe_on_repeat_upsert(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.upsert_session_entities(["日本語クラス"])
    memory.upsert_session_entities(["日本語クラス"])
    entities = [e for e in memory.snapshot().session_entities if e.value == "日本語クラス"]
    assert len(entities) == 1
    assert entities[0].count == 2
