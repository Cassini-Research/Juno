"""Lexicon canonicalization rails in juno_v2/memory/bias.py.

Covers which lexicon rows are allowed to rewrite committed transcript text.
The provenance of a row decides: rows harvested off the user's screen only
bias the recognizer, rows somebody vouched for (user-taught, or the curated
packs shipped in ``seed_data``) canonicalize as before, and rows of unknown
provenance have to look like an identifier or be visible on screen right now.
"""

from __future__ import annotations

import re
from pathlib import Path

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import LexiconEntry, MemorySnapshot
from juno_v2.memory import bias as bias_module
from juno_v2.memory.bias import RecognitionBiasEngine, _is_low_signal_lexicon_pair
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.personalization.seed.load_bundle import load_seed_bundle
from juno_v2.personalization.seed.promotion import PromotionCoordinator


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


def test_context_promoted_camel_case_identifier_stays_bias_only() -> None:
    """Screen-harvested code identifiers are the same hazard as plain nouns.

    ``useEffect`` scraped out of an editor used to pass the identifier-shape
    rail, and ``_phrase_pattern`` tolerates the space, so dictating the
    ordinary words "use effect" came back as "useEffect".
    """
    lexicon = [
        LexiconEntry(term="useEffect", canonical_form="useEffect", source="context_promoted")
    ]

    assert (
        _normalize(lexicon, "that had a strange use effect on the team")
        == "that had a strange use effect on the team"
    )


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
# Shipped seed packs are curated by the project, so they stay trusted
# --------------------------------------------------------------------- #


def test_seed_promoted_mixed_case_identifier_still_canonicalizes() -> None:
    lexicon = [
        LexiconEntry(term="BitsAndBytes", canonical_form="BitsAndBytes", source="seed_promotion")
    ]

    assert _normalize(lexicon, "install bitsandbytes first") == "install BitsAndBytes first"


def test_seed_promoted_plain_word_keeps_its_branded_casing() -> None:
    lexicon = [LexiconEntry(term="Qwen", canonical_form="Qwen", source="seed_promotion")]

    assert _normalize(lexicon, "run qwen locally") == "run Qwen locally"


def _seed_lexicon(tmp_path: Path):
    seed = load_seed_bundle(Path(__file__).resolve().parents[1] / "seed_data")
    memory = JsonMemoryStore(tmp_path / "memory")
    PromotionCoordinator(seed=seed, memory_store=memory, learned_store=None).run_initial_promotion(
        memory
    )
    return memory.snapshot()


def _rules(entry):
    yield entry.canonical_form, entry.canonical_form
    yield entry.term, entry.canonical_form
    for alias in entry.aliases:
        yield alias, entry.canonical_form


def _allowed_before_provenance_gating(engine, trigger: str, replacement: str, context) -> bool:
    """The predicate exactly as it read before provenance gating was added.

    Kept here so the shipped-seed corpus can be pinned against the previous
    behaviour: promoting ``seed_data`` must not cost a single rewrite rule.
    """
    t, r = (trigger or "").strip(), (replacement or "").strip()
    if not t or not r:
        return False
    trigger_tokens = re.findall(r"[A-Za-z0-9]+", t)
    replacement_tokens = re.findall(r"[A-Za-z0-9]+", r)
    if not trigger_tokens or not replacement_tokens:
        return False
    if len(trigger_tokens) == 1 and trigger_tokens[0].casefold() in (
        bias_module._LOW_SIGNAL_SESSION_ENTITY_WORDS  # noqa: SLF001
        | bias_module._GENERIC_SINGLE_CANONICALIZATION_ALIAS_WORDS  # noqa: SLF001
    ):
        return engine._seed_phrase_visible_in_context(r, context)  # noqa: SLF001
    if len(trigger_tokens) >= 2 or len(replacement_tokens) <= 1:
        return True
    if any(ch.isdigit() for ch in t) or any(ch in t for ch in {"_", "-", ".", "/", "#"}):
        return True
    if re.search(r"[a-z][A-Z]|\b[A-Z]{2,}\b", t):
        return True
    return engine._seed_phrase_visible_in_context(r, context)  # noqa: SLF001


def test_shipped_seed_packs_keep_every_canonicalization_rule(tmp_path: Path) -> None:
    snapshot = _seed_lexicon(tmp_path)
    engine = RecognitionBiasEngine()
    context = TypedContextBundle()
    assert len(snapshot.lexicon) > 300

    lost: list[tuple[str, str]] = []
    identity_denied: list[str] = []
    for entry in snapshot.lexicon:
        assert entry.source == "seed_promotion"
        if _is_low_signal_lexicon_pair(entry.term, entry.canonical_form):
            continue
        for trigger, replacement in _rules(entry):
            allowed = engine._lexicon_canonicalization_allowed(  # noqa: SLF001
                trigger, replacement, context, source=entry.source
            )
            if not allowed and _allowed_before_provenance_gating(
                engine, trigger, replacement, context
            ):
                lost.append((trigger, replacement))
            if not allowed and trigger == entry.canonical_form:
                identity_denied.append(entry.canonical_form)

    assert lost == []
    assert identity_denied == []


def test_shipped_seed_packs_still_fix_branded_casing(tmp_path: Path) -> None:
    lexicon = _seed_lexicon(tmp_path).lexicon

    assert _normalize(lexicon, "open juno settings") == "open Juno settings"
    assert _normalize(lexicon, "build in xcode") == "build in Xcode"
    assert _normalize(lexicon, "post it in slack") == "post it in Slack"
    assert _normalize(lexicon, "run sglang on the box") == "run SGLang on the box"
    assert _normalize(lexicon, "install rockm drivers") == "install ROCm drivers"


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
        "Leaderboard", "Leaderboard", context, source="imported_elsewhere"
    )
    assert engine._lexicon_canonicalization_allowed(  # noqa: SLF001
        "GPU", "GPU", context, source="imported_elsewhere"
    )
    # Vouched-for provenance keeps the plain-word rewrite.
    for trusted in ("user", "seed_promotion"):
        assert engine._lexicon_canonicalization_allowed(  # noqa: SLF001
            "Leaderboard", "Leaderboard", context, source=trusted
        )


def test_leading_acronym_identifiers_pass_the_shape_rail() -> None:
    """``SGLang``/``ESLint``/``OLMo``/``ROCm`` are identifiers, whoever stored them."""
    engine = RecognitionBiasEngine()
    context = TypedContextBundle(app_name="TextEdit")

    for identifier in ("SGLang", "ESLint", "OLMo", "ROCm"):
        assert engine._lexicon_canonicalization_allowed(  # noqa: SLF001
            identifier, identifier, context, source="imported_elsewhere"
        ), identifier
