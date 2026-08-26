"""Lexicon canonicalization rails in juno_v2/memory/bias.py.

Covers which lexicon rows are allowed to rewrite committed transcript text.
The provenance of a row decides: screen-harvested rows only bias the
recognizer, user-taught rows are trusted, and everything else has to look
like an identifier or be visible on screen right now.
"""

from __future__ import annotations

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import LexiconEntry, MemorySnapshot
from juno_v2.memory.bias import RecognitionBiasEngine


def _plan(engine: RecognitionBiasEngine, snapshot: MemorySnapshot, context: TypedContextBundle):
    return engine.build_plan(
        utterance_id="utt-lexicon-canonicalization",
        snapshot=snapshot,
        context=context,
        memory_packet=engine.build_serving_packet(snapshot=snapshot),
    )


def _normalize(
    lexicon: list[LexiconEntry],
    text: str,
    *,
    context: TypedContextBundle | None = None,
) -> str:
    engine = RecognitionBiasEngine()
    snapshot = MemorySnapshot(schema_version=1, lexicon=lexicon)
    ctx = context or TypedContextBundle(app_name="TextEdit", app_category="notes")
    return engine.normalize_transcript(
        text,
        snapshot=snapshot,
        plan=_plan(engine, snapshot, ctx),
        scope="final",
    ).normalized_text


# --------------------------------------------------------------------- #
# Screen-harvested rows: bias only, never a rewrite
# --------------------------------------------------------------------- #


def test_context_promoted_single_token_does_not_force_case() -> None:
    lexicon = [
        LexiconEntry(
            term="Leaderboard",
            canonical_form="Leaderboard",
            aliases=["Leaderboard"],
            source="context_promoted",
        ),
        LexiconEntry(
            term="P2P",
            canonical_form="P2P",
            aliases=["peer to peer"],
            source="context_promoted",
        ),
    ]

    assert (
        _normalize(lexicon, "a leaderboard for the peer-to-peer game")
        == "a leaderboard for the peer-to-peer game"
    )


def test_context_promoted_alias_does_not_rewrite_spoken_phrase() -> None:
    lexicon = [
        LexiconEntry(
            term="P2P",
            canonical_form="P2P",
            aliases=["peer to peer"],
            source="context_promoted",
        )
    ]

    assert _normalize(lexicon, "we shipped peer to peer sync") == "we shipped peer to peer sync"


def test_context_promoted_row_stays_visible_on_screen_without_rewriting() -> None:
    """Even with the term on screen the harvested row must not rewrite output."""
    lexicon = [
        LexiconEntry(term="Leaderboard", canonical_form="Leaderboard", source="context_promoted")
    ]
    context = TypedContextBundle(app_name="Chrome", window_title="Leaderboard - Season 4")

    assert _normalize(lexicon, "open the leaderboard", context=context) == "open the leaderboard"


def test_context_promoted_terms_still_bias_the_recognizer() -> None:
    engine = RecognitionBiasEngine()
    snapshot = MemorySnapshot(
        schema_version=1,
        lexicon=[
            LexiconEntry(term="Leaderboard", canonical_form="Leaderboard", source="context_promoted"),
            LexiconEntry(term="P2P", canonical_form="P2P", source="context_promoted"),
        ],
    )
    plan = _plan(engine, snapshot, TypedContextBundle(app_name="TextEdit"))

    assert "Leaderboard" in plan.bias_phrases
    assert "P2P" in plan.bias_phrases


# --------------------------------------------------------------------- #
# User-taught rows keep canonicalizing
# --------------------------------------------------------------------- #


def test_user_taught_single_token_still_canonicalizes() -> None:
    lexicon = [LexiconEntry(term="kubernetes", canonical_form="Kubernetes", source="user")]

    assert _normalize(lexicon, "we deploy on kubernetes today") == "we deploy on Kubernetes today"


def test_user_taught_multi_token_alias_still_canonicalizes() -> None:
    lexicon = [
        LexiconEntry(
            term="luma ray",
            canonical_form="LumaRay",
            aliases=["luma ray", "lumaray"],
            source="user",
        )
    ]

    assert _normalize(lexicon, "ask luma ray about it") == "ask LumaRay about it"


def test_user_edit_row_still_canonicalizes() -> None:
    lexicon = [
        LexiconEntry(
            term="Polkafoundation",
            canonical_form="Polkafoundation",
            aliases=["polka foundation"],
            source="user_edit",
        )
    ]

    assert (
        _normalize(lexicon, "the polka foundation shipped it")
        == "the Polkafoundation shipped it"
    )


def test_voice_command_taught_single_token_still_canonicalizes() -> None:
    lexicon = [LexiconEntry(term="Karvix", canonical_form="Karvix", source="voice_command")]

    assert _normalize(lexicon, "the karvix build is green") == "the Karvix build is green"


def test_correction_promoted_row_still_rewrites() -> None:
    lexicon = [
        LexiconEntry(
            term="SilviaGamachi",
            canonical_form="SilviaGamachi",
            aliases=["Silvia Gamache"],
            source="correction_promoted",
        )
    ]

    assert _normalize(lexicon, "I met Silvia Gamache today") == "I met SilviaGamachi today"


def test_default_source_rows_are_treated_as_user_taught() -> None:
    """``LexiconEntry.source`` defaults to ``user``: the Memory UI lane."""
    lexicon = [LexiconEntry(term="qdrant", canonical_form="Qdrant")]

    assert _normalize(lexicon, "the qdrant index is warm") == "the Qdrant index is warm"


# --------------------------------------------------------------------- #
# Shipped seed rows: identifier shape or on-screen evidence
# --------------------------------------------------------------------- #


def test_seed_promoted_mixed_case_identifier_still_canonicalizes() -> None:
    lexicon = [
        LexiconEntry(term="BitsAndBytes", canonical_form="BitsAndBytes", source="seed_promotion")
    ]

    assert _normalize(lexicon, "install bitsandbytes first") == "install BitsAndBytes first"


def test_seed_promoted_plain_word_needs_context_evidence() -> None:
    lexicon = [LexiconEntry(term="Cursor", canonical_form="Cursor", source="seed_promotion")]

    assert _normalize(lexicon, "put the cursor at the end") == "put the cursor at the end"


def test_seed_promoted_plain_word_canonicalizes_with_context_evidence() -> None:
    lexicon = [LexiconEntry(term="Qwen", canonical_form="Qwen", source="seed_promotion")]
    context = TypedContextBundle(app_name="Terminal", window_title="Qwen benchmark")

    assert _normalize(lexicon, "run qwen locally", context=context) == "run Qwen locally"


# --------------------------------------------------------------------- #
# Predicate-level rails
# --------------------------------------------------------------------- #


def test_single_token_rows_reach_the_guard_rails() -> None:
    engine = RecognitionBiasEngine()
    context = TypedContextBundle(app_name="TextEdit")

    # Harvested provenance is rejected outright.
    assert not engine._lexicon_canonicalization_allowed(  # noqa: SLF001
        "Leaderboard", "Leaderboard", context, source="context_promoted"
    )
    # Unknown provenance falls through to the shape/evidence rails instead of
    # the old "single replacement token -> always allowed" short circuit.
    assert not engine._lexicon_canonicalization_allowed(  # noqa: SLF001
        "Leaderboard", "Leaderboard", context, source="seed_promotion"
    )
    assert engine._lexicon_canonicalization_allowed(  # noqa: SLF001
        "GPU", "GPU", context, source="seed_promotion"
    )
    assert engine._lexicon_canonicalization_allowed(  # noqa: SLF001
        "Leaderboard", "Leaderboard", context, source="user"
    )
