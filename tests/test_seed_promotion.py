from __future__ import annotations

from pathlib import Path

from juno_v2.context.compiler import compile_context
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.personalization.seed.load_bundle import load_seed_bundle
from juno_v2.personalization.seed.models import BiasBundleTerm, SeedBiasAttachment, StructuredBiasBundle
from juno_v2.personalization.seed.promotion import PromotionCoordinator


def test_initial_promotion_keeps_default_packs_out_of_durable_memory(tmp_path: Path) -> None:
    seed = load_seed_bundle(Path(__file__).resolve().parents[1] / "seed_data")
    memory = JsonMemoryStore(tmp_path / "memory")

    result = PromotionCoordinator(
        seed=seed,
        memory_store=memory,
        learned_store=None,
    ).run_initial_promotion(memory)

    canonicals = {str(row.get("canonical_form") or row.get("term") or "") for row in memory.vocabulary.raw()}
    assert result["ok"] is True
    # Contract change 2026-06-10: default packs DO promote into the serving
    # lexicon. "Runtime-only" pack serving never actually delivered terms to
    # the bias/editor lanes, so domain identifiers (e.g. the model-name pack)
    # were unusable and ASR mishears shipped unrepaired.
    assert "Juno" in canonicals
    assert "Qwen" in canonicals
    assert len(canonicals) > 50
    assert "Gemma" in canonicals


def test_seed_extra_bias_phrases_do_not_compile_as_memory_terms() -> None:
    attachment = SeedBiasAttachment(
        extra_bias_phrases=("Gemma", "Gemma 4"),
        structured_bundle=StructuredBiasBundle(
            terms=(
                BiasBundleTerm(
                    canonical="Gemma",
                    bias_strings=("Gemma",),
                    tags=("ai", "models"),
                    pack_name="domain_ai_models_terms",
                    rank_key=(0, "domain_ai_models_terms", "gemma"),
                ),
            ),
            flattened_bias_phrases=("Gemma",),
            clipped=False,
            clip_reason=None,
        ),
        canonicalization_tuples=(),
        metadata={"active_pack_ids": ["domain_ai_models_terms"]},
    )

    compiled = compile_context(
        utterance_id="utt-seed-extra-bias",
        context=TypedContextBundle(app_name="TextEdit", app_category="notes"),
        memory_snapshot=MemorySnapshot(schema_version=1),
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        mode_policy=BUILTIN_MODES["default_surface"],
        transcript_hint=None,
        session_terms=None,
        language="en",
        stage="final_delivery",
        seed_attachment=attachment,
    )

    terms = {term.text for term in compiled.terms}
    asr_terms = {term.text for term in compiled.asr_bias_packet().terms}
    assert "Gemma" not in terms
    assert "Gemma" not in asr_terms
