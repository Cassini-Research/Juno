"""Issue #79 — sentence punctuation must not leak into the Whisper prompt.

The "Prefer exact forms" line is joined with ``", "``. Any phrase that keeps
its sentence-final ``.?!`` produces ``.,`` / ``?,`` runs, and Whisper copies
the prompt's punctuation style into the transcript. These tests pin both the
serving-time strip and the promotion-time rejection that keeps whole sentences
out of the lexicon in the first place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from juno_v2.context.compiler import _pack_prefer_line
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import LexiconEntry, MemorySnapshot
from juno_v2.memory.bias import RecognitionBiasEngine
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.term_policy import (
    learned_term_is_sentence_like,
    strip_terminal_sentence_punctuation,
)
from juno_v2.personalization.seed.learned_state import JunoPersonalizationLearnedStore
from juno_v2.personalization.seed.load_bundle import load_seed_bundle
from juno_v2.personalization.seed.promotion import PromotionCoordinator


def _seed_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "seed_data"


ISSUE_PHRASES = (
    "PR",
    "Widget",
    "The quick brown fox jumps over Widget.",
    "Does the lazy dog sleep?",
)


def test_pack_prefer_line_strips_sentence_punctuation() -> None:
    line = _pack_prefer_line(ISSUE_PHRASES, max_chars=768)

    assert line == (
        "Prefer exact forms: PR, Widget, "
        "The quick brown fox jumps over Widget, Does the lazy dog sleep"
    )
    assert ".," not in line
    assert "?," not in line
    assert not line.endswith((".", "?", "!"))


def test_bias_engine_prefer_line_strips_sentence_punctuation() -> None:
    engine = RecognitionBiasEngine()

    line, embedded = engine._build_prefer_exact_forms_line(list(ISSUE_PHRASES), max_chars=768)

    assert embedded == 4
    assert ".," not in line
    assert "?," not in line
    assert "jumps over Widget," in line


def test_bias_plan_initial_prompt_has_no_comma_spliced_punctuation() -> None:
    snapshot = MemorySnapshot(
        schema_version=1,
        lexicon=[
            LexiconEntry(term="Karvix", canonical_form="Karvix"),
            LexiconEntry(
                term="The quick brown fox jumps over Widget.",
                canonical_form="The quick brown fox jumps over Widget.",
            ),
        ],
    )

    plan = RecognitionBiasEngine().build_plan(
        utterance_id="u-79",
        snapshot=snapshot,
        context=TypedContextBundle(app_name="TextEdit", app_category="notes"),
    )

    assert plan.initial_prompt is not None
    assert ".," not in plan.initial_prompt
    assert "?," not in plan.initial_prompt


@pytest.mark.parametrize(
    "phrase",
    [
        "Node.js",
        "v1.2",
        "C++",
        "e.g.",
        "Acme Corp.",
        "made in the U.S.",
        "St. Louis",
        "Karvix",
    ],
)
def test_term_internal_punctuation_is_preserved(phrase: str) -> None:
    assert strip_terminal_sentence_punctuation(phrase) == phrase
    assert _pack_prefer_line((phrase,), max_chars=768) == f"Prefer exact forms: {phrase}"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("The quick brown fox jumps over Widget.", "The quick brown fox jumps over Widget"),
        ("Does the lazy dog sleep?", "Does the lazy dog sleep"),
        ("Ship the Widget!", "Ship the Widget"),
        ("Widget model.", "Widget model"),
        ("wait for it...", "wait for it"),
        ("really Widget?!", "really Widget"),
        ("Widget ships soon…", "Widget ships soon"),
        ("Node.js runtime.", "Node.js runtime"),
        ("  trailing space Widget.  ", "  trailing space Widget"),
    ],
)
def test_sentence_final_punctuation_is_stripped(phrase: str, expected: str) -> None:
    assert strip_terminal_sentence_punctuation(phrase) == expected


def test_pack_prefer_line_budget_uses_stripped_lengths() -> None:
    prefix_len = len("Prefer exact forms: ")
    # "Widget" (6) + ", " + "Alpha Beta Gamma" (16, once its period is gone).
    line = _pack_prefer_line(
        ("Widget", "Alpha Beta Gamma."),
        max_chars=prefix_len + 6 + 2 + 16,
    )

    assert line == "Prefer exact forms: Widget, Alpha Beta Gamma"


@pytest.mark.parametrize(
    "candidate",
    [
        "The quick brown fox jumps over Widget.",
        "Does the lazy dog sleep?",
        "Widget model.",
        "Ship it. Widget",
        "one two three four five six seven",
    ],
)
def test_sentence_like_candidates_detected(candidate: str) -> None:
    assert learned_term_is_sentence_like(candidate) is True


@pytest.mark.parametrize(
    "candidate",
    ["Karvix", "Node.js", "v1.2", "C++", "e.g.", "Acme Corp.", "Karvix Widget Platform"],
)
def test_term_candidates_are_not_sentence_like(candidate: str) -> None:
    assert learned_term_is_sentence_like(candidate) is False


def _coordinator(tmp_path: Path) -> tuple[PromotionCoordinator, JsonMemoryStore]:
    memory = JsonMemoryStore(tmp_path / "memory")
    coordinator = PromotionCoordinator(
        seed=load_seed_bundle(_seed_data_root()),
        memory_store=memory,
        learned_store=JunoPersonalizationLearnedStore(tmp_path / "memory"),
    )
    return coordinator, memory


def test_correction_promotion_rejects_sentence_candidates(tmp_path: Path) -> None:
    coordinator, memory = _coordinator(tmp_path)
    observed = "the quick brown fix jumps over widget"
    corrected = "The quick brown fox jumps over Widget."
    assert memory.corrections.record(observed, corrected) is True

    result = coordinator.maybe_promote_correction_to_lexicon(
        observed=observed,
        corrected=corrected,
        durable_memory_suppressed=False,
    )

    assert result == {"promoted": False, "reason": "term_sentence_like"}
    assert not [row for row in memory.vocabulary.raw() if "quick brown" in str(row.get("term", ""))]


def test_correction_promotion_still_promotes_real_terms(tmp_path: Path) -> None:
    coordinator, memory = _coordinator(tmp_path)
    assert memory.corrections.record("car vix", "Karvix") is True

    result = coordinator.maybe_promote_correction_to_lexicon(
        observed="car vix",
        corrected="Karvix",
        durable_memory_suppressed=False,
    )

    assert result["promoted"] is True
    assert "Karvix" in {str(row.get("term", "")) for row in memory.vocabulary.raw()}


def test_context_promotion_rejects_sentence_candidates(tmp_path: Path) -> None:
    coordinator, memory = _coordinator(tmp_path)
    learned = JunoPersonalizationLearnedStore(tmp_path / "memory")
    token = "Does the lazy dog sleep?"
    for _ in range(3):
        learned.increment_observation(token, from_suppressed_context=False)
    learned.increment_acceptance(token, from_suppressed_context=False)

    result = coordinator.maybe_promote_context_entity_to_lexicon(
        token=token,
        durable_memory_suppressed=False,
    )

    assert result == {"promoted": False, "reason": "term_sentence_like"}
    assert not [row for row in memory.vocabulary.raw() if "lazy dog" in str(row.get("term", ""))]


def test_context_promotion_still_promotes_real_terms(tmp_path: Path) -> None:
    coordinator, memory = _coordinator(tmp_path)
    learned = JunoPersonalizationLearnedStore(tmp_path / "memory")
    for _ in range(3):
        learned.increment_observation("Karvix", from_suppressed_context=False)
    learned.increment_acceptance("Karvix", from_suppressed_context=False)

    result = coordinator.maybe_promote_context_entity_to_lexicon(
        token="Karvix",
        durable_memory_suppressed=False,
    )

    assert result["promoted"] is True
    assert "Karvix" in {str(row.get("term", "")) for row in memory.vocabulary.raw()}
