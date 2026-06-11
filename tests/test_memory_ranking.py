"""Unit tests for juno_v2/memory/ranking.py rank_memory_for_context."""

from __future__ import annotations

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import (
    CorrectionPair,
    LexiconEntry,
    MemorySnapshot,
    ReplacementRule,
    SessionEntity,
)
from juno_v2.memory.ranking import rank_memory_for_context


def _snapshot(**kwargs) -> MemorySnapshot:
    return MemorySnapshot(schema_version=1, **kwargs)


# --------------------------------------------------------------------- #
# Replacement scope ranking
# --------------------------------------------------------------------- #


def test_replacements_app_scoped_beats_global(tmp_path) -> None:
    snapshot = _snapshot(
        replacements=[
            ReplacementRule(trigger="ga", replacement="g", scope="global"),
            ReplacementRule(trigger="oa", replacement="o", scope="app:mail"),
            ReplacementRule(trigger="na", replacement="n", scope="slack"),
            ReplacementRule(trigger="pa", replacement="p", scope="app:slack"),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(app_name="Slack"),
        transcript_hint="pa na ga oa",
    )
    order = [r["trigger"] for r in packet.replacements]
    # app:<current app> (+10) > bare app scope (+8) > global (+1) >
    # other app's scope (+0).
    assert order == ["pa", "na", "ga", "oa"]


def test_replacements_app_scope_casefolds(tmp_path) -> None:
    snapshot = _snapshot(
        replacements=[
            ReplacementRule(trigger="ga", replacement="g", scope="global"),
            ReplacementRule(trigger="pa", replacement="p", scope="app:slack"),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(app_name="  SLACK  "),
        transcript_hint="pa ga",
    )
    assert [r["trigger"] for r in packet.replacements] == ["pa", "ga"]


def test_replacements_hint_token_match_outranks_app_scope(tmp_path) -> None:
    snapshot = _snapshot(
        replacements=[
            ReplacementRule(trigger="scoped", replacement="s", scope="app:slack"),
            ReplacementRule(trigger="qbr deck", replacement="Quarterly Business Review deck", scope="global"),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(app_name="Slack"),
        transcript_hint="let's review the scopes QBR deck tomorrow",
    )
    # Hint-token overlap (+12, plus global +1) beats app:slack (+10).
    assert [r["trigger"] for r in packet.replacements] == ["qbr deck", "scoped"]


def test_replacements_mode_scope_bonus(tmp_path) -> None:
    snapshot = _snapshot(
        replacements=[
            ReplacementRule(trigger="ga", replacement="g", scope="global"),
            ReplacementRule(trigger="ma", replacement="m", scope="mode:notes"),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(),
        effective_mode="Notes",
        transcript_hint="ga ma",
    )
    # mode-key substring match (+6) beats plain global (+1).
    assert [r["trigger"] for r in packet.replacements] == ["ma", "ga"]


def test_replacements_capped_at_eight(tmp_path) -> None:
    snapshot = _snapshot(
        replacements=[
            ReplacementRule(trigger=f"trigger{i}", replacement="x", scope="global")
            for i in range(12)
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(),
        transcript_hint=" ".join(f"trigger{i}" for i in range(12)),
    )
    assert len(packet.replacements) == 8
    assert packet.metadata["replacement_total"] == 12


# --------------------------------------------------------------------- #
# Lexicon hint prioritization + low-signal filtering
# --------------------------------------------------------------------- #


def test_lexicon_hint_token_match_outranks_boost(tmp_path) -> None:
    snapshot = _snapshot(
        lexicon=[
            LexiconEntry(term="Zebra", canonical_form="Zebra", boost=5.0),
            LexiconEntry(term="Kubernetes", canonical_form="Kubernetes", boost=1.0),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(),
        transcript_hint="deploying to kubernetes today",
    )
    assert packet.lexicon_terms == ["Kubernetes", "Zebra"]


def test_lexicon_hint_can_come_from_context_fields(tmp_path) -> None:
    snapshot = _snapshot(
        lexicon=[
            LexiconEntry(term="Zebra", canonical_form="Zebra", boost=5.0),
            LexiconEntry(term="Polkafoundation", canonical_form="Polkafoundation", boost=1.0),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(
            window_title="Polkafoundation — quarterly sync"
        ),
    )
    assert packet.lexicon_terms[0] == "Polkafoundation"
    assert packet.metadata["ranking"]["hint_token_count"] >= 2


def test_lexicon_low_signal_pairs_filtered(tmp_path) -> None:
    snapshot = _snapshot(
        lexicon=[
            LexiconEntry(term="the", canonical_form="the"),
            LexiconEntry(term="You", canonical_form="you"),
            LexiconEntry(term="Qwen", canonical_form="Qwen"),
        ]
    )
    packet = rank_memory_for_context(snapshot, context=TypedContextBundle())
    assert packet.lexicon_terms == ["Qwen"]
    # Totals still report the raw store size.
    assert packet.metadata["lexicon_total"] == 3


def test_lexicon_stopword_with_distinct_canonical_is_kept(tmp_path) -> None:
    # Only pairs where term == canonical AND both are low-signal get
    # dropped; a stopword-looking alias mapping to a real canonical stays.
    snapshot = _snapshot(
        lexicon=[LexiconEntry(term="the", canonical_form="Theo")]
    )
    packet = rank_memory_for_context(snapshot, context=TypedContextBundle())
    assert packet.lexicon_terms == ["Theo"]


def test_lexicon_capped_at_twelve_and_aliases_served(tmp_path) -> None:
    lexicon = [
        LexiconEntry(term=f"Term{i}", canonical_form=f"Term{i}") for i in range(15)
    ]
    lexicon.append(
        LexiconEntry(
            term="polka foundation",
            canonical_form="Polkafoundation",
            aliases=["Polka-Foundation"],
            boost=9.0,
        )
    )
    packet = rank_memory_for_context(
        _snapshot(lexicon=lexicon), context=TypedContextBundle()
    )
    assert len(packet.lexicon_terms) == 12
    assert packet.lexicon_terms[0] == "Polkafoundation"  # highest boost
    aliases = packet.metadata["lexicon_aliases"]["Polkafoundation"]
    assert "polka foundation" in aliases
    assert "Polka-Foundation" in aliases


# --------------------------------------------------------------------- #
# Session entity filtering
# --------------------------------------------------------------------- #


def test_session_entities_low_signal_single_words_filtered(tmp_path) -> None:
    snapshot = _snapshot(
        session_entities=[
            SessionEntity(value="you", count=50),
            SessionEntity(value="the", count=50),
            SessionEntity(value="Karvix", count=1),
            SessionEntity(value="Acme Corp", count=1),
        ]
    )
    packet = rank_memory_for_context(snapshot, context=TypedContextBundle())
    assert set(packet.session_entities) == {"Karvix", "Acme Corp"}
    assert packet.metadata["session_entity_total"] == 4


def test_session_entities_all_stopword_phrase_is_filtered(tmp_path) -> None:
    # The entity policy filters phrases made entirely of common English
    # words ("The Who" is a famous false negative we accept): serving
    # common-word phrases as ASR bias corrupts ordinary dictation far more
    # often than it helps. Phrases with at least one rare token survive.
    snapshot = _snapshot(
        session_entities=[
            SessionEntity(value="The Who", count=1),
            SessionEntity(value="Karvix Labs", count=1),
        ]
    )
    packet = rank_memory_for_context(snapshot, context=TypedContextBundle())
    assert packet.session_entities == ["Karvix Labs"]


def test_session_entities_hint_match_outranks_count(tmp_path) -> None:
    snapshot = _snapshot(
        session_entities=[
            SessionEntity(value="Gamma Project", count=10),
            SessionEntity(value="Karvix", count=1),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(selected_text="ping Karvix about the deck"),
    )
    assert packet.session_entities[0] == "Karvix"


# --------------------------------------------------------------------- #
# Corrections ranking
# --------------------------------------------------------------------- #


def test_corrections_hint_match_outranks_count(tmp_path) -> None:
    snapshot = _snapshot(
        corrections=[
            CorrectionPair(observed="alpha", corrected="Alphabet", count=9),
            CorrectionPair(observed="chino", corrected="Juno", count=1),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(),
        transcript_hint="tell chino we shipped",
    )
    assert packet.corrections[0]["corrected"] == "Juno"
    assert packet.corrections[0]["count"] == 1


# --------------------------------------------------------------------- #
# Packet metadata / empty input
# --------------------------------------------------------------------- #


def test_empty_snapshot_produces_empty_packet(tmp_path) -> None:
    packet = rank_memory_for_context(_snapshot(), context=TypedContextBundle())
    assert packet.lexicon_terms == []
    assert packet.replacements == []
    assert packet.corrections == []
    assert packet.session_entities == []
    assert packet.metadata["ranking"]["app_scope"] is None
    assert packet.metadata["ranking"]["hint_token_count"] == 0


def test_metadata_records_ranking_inputs(tmp_path) -> None:
    packet = rank_memory_for_context(
        _snapshot(),
        context=TypedContextBundle(app_name="Slack"),
        effective_mode="Notes",
        transcript_hint="review the QBR deck",
        session_terms=["Karvix"],
    )
    ranking = packet.metadata["ranking"]
    assert ranking["app_scope"] == "slack"
    assert ranking["mode"] == "notes"
    assert ranking["session_term_count"] == 1
    # Tokens: review/the/qbr/deck + chino (>=2 chars each).
    assert ranking["hint_token_count"] == 5


def test_replacements_not_served_without_trigger_presence(tmp_path) -> None:
    # Production 2026-06-11: a seeded "launch code" replacement injected
    # LAUNCH-CODE-991 into a selected rewrite that never mentioned it.
    snapshot = _snapshot(
        replacements=[
            ReplacementRule(trigger="launch code", replacement="LAUNCH-CODE-991", scope="global"),
        ]
    )
    packet = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(app_name="Mail"),
        transcript_hint="please make this paragraph more formal",
    )
    assert packet.replacements == []
    # Near-miss of the trigger is still admissible (ASR drift coverage).
    packet2 = rank_memory_for_context(
        snapshot,
        context=TypedContextBundle(app_name="Mail"),
        transcript_hint="set the launch codes to ready",
    )
    assert [r["trigger"] for r in packet2.replacements] == ["launch code"]
