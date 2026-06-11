from __future__ import annotations

from juno_v2.memory.store import JsonMemoryStore
from juno_v2.personalization.seed.learned_state import JunoPersonalizationLearnedStore


def test_clear_all_removes_learned_memory_and_preserves_protected_vocab(tmp_path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory")
    memory.add_lexicon_entry(term="Karvix", canonical_form="Karvix", source="context_promoted")
    memory.add_replacement(trigger="hey karvix", replacement="hey juno")
    assert memory.record_correction("karvix", "juno")
    memory.upsert_session_entities(["Karvix"])
    memory.add_snippet(trigger="sig", body="Thanks,\nJuno")

    learned = JunoPersonalizationLearnedStore(memory.memory_dir)
    learned.increment_observation("Karvix", from_suppressed_context=False)
    learned.increment_acceptance("Karvix", from_suppressed_context=False)

    result = memory.clear_all()

    assert result["before"] == {
        "lexicon": 2,
        "replacements": 1,
        "corrections": 1,
        "session_entities": 1,
        "snippets": 1,
    }
    assert result["after"] == {
        "lexicon": 1,
        "replacements": 0,
        "corrections": 0,
        "session_entities": 0,
        "snippets": 0,
    }
    assert result["removed"] == {
        "lexicon": 1,
        "replacements": 1,
        "corrections": 1,
        "session_entities": 1,
        "snippets": 1,
    }

    snapshot = memory.snapshot()
    assert [entry.canonical_form for entry in snapshot.lexicon] == ["Juno"]
    assert snapshot.replacements == []
    assert snapshot.corrections == []
    assert snapshot.session_entities == []
    assert memory.snippets.raw() == []
    assert learned.observation_snapshot("Karvix") is None

    packet = memory.serving_packet()
    assert "Juno" in packet.lexicon_terms
    assert "Karvix" not in packet.lexicon_terms
