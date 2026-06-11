from __future__ import annotations

import pytest

from juno_core_v3.model_registry.contracts import (
    RUNTIME_SUPPORTED_BACKENDS,
    ModelPromotionStage,
    ModelSlot,
    RuntimeBackend,
    SurfaceClass,
)
from juno_core_v3.model_registry.defaults import build_default_registry
from juno_core_v3.model_registry.manifest import CapabilityManifest
from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry
from juno_core_v3.model_registry.routing import RouteChooser, RouteRequest


def make_pkg(
    package_id: str,
    *,
    slot: ModelSlot = ModelSlot.FINAL_ASR,
    promotion: ModelPromotionStage = ModelPromotionStage.CANDIDATE,
    languages: tuple[str, ...] = ("en",),
    streaming: bool = False,
    min_ram_mb: int = 0,
    disallow_surfaces: tuple[SurfaceClass, ...] = (),
) -> ModelPackage:
    return ModelPackage(
        package_id=package_id,
        version="0.1",
        manifest=CapabilityManifest(
            slot=slot,
            backend=RuntimeBackend.FASTER_WHISPER,
            languages=languages,
            streaming=streaming,
            min_ram_mb=min_ram_mb,
            disallow_surfaces=disallow_surfaces,
        ),
        promotion=promotion,
    )


def registry_of(*packages: ModelPackage) -> ModelRegistry:
    reg = ModelRegistry()
    for pkg in packages:
        reg.add(pkg)
    return reg


# ---- CapabilityManifest ----------------------------------------------


def test_manifest_supports_language_none_always_true() -> None:
    m = make_pkg("p", languages=("en",)).manifest
    assert m.supports_language(None) is True
    assert m.supports_language("en") is True
    assert m.supports_language("zh") is False


def test_manifest_allows_surface() -> None:
    m = make_pkg("p", disallow_surfaces=(SurfaceClass.KEYBOARD_EXTENSION,)).manifest
    assert m.allows_surface(SurfaceClass.DESKTOP) is True
    assert m.allows_surface(SurfaceClass.KEYBOARD_EXTENSION) is False


def test_manifest_meets_streaming() -> None:
    non_streaming = make_pkg("p", streaming=False).manifest
    streaming = make_pkg("q", streaming=True).manifest
    assert non_streaming.meets_streaming(False) is True
    assert non_streaming.meets_streaming(True) is False
    assert streaming.meets_streaming(True) is True
    assert streaming.meets_streaming(False) is True


def test_manifest_defaults() -> None:
    m = CapabilityManifest(slot=ModelSlot.WRITER, backend=RuntimeBackend.MLX_LM)
    assert m.languages == ("en",)
    assert m.streaming is False
    assert m.min_ram_mb == 0
    assert m.disallow_surfaces == ()
    assert m.wer_p50 is None
    assert m.latency_ms_p50 is None


# ---- contracts enums --------------------------------------------------


def test_contracts_enum_values() -> None:
    assert ModelSlot.FINAL_ASR.value == "final_asr"
    assert ModelSlot.PREVIEW_ASR.value == "preview_asr"
    assert SurfaceClass.DESKTOP.value == "desktop"
    assert ModelPromotionStage.PROMOTED.value == "promoted"
    assert ModelPromotionStage.RETIRED.value == "retired"
    # Enums are str-valued for stable serialization.
    assert isinstance(ModelSlot.TINY_ROUTER, str)
    assert isinstance(SurfaceClass.PHONE_CLASS, str)


def test_runtime_supported_backends_cover_all_declared_backends() -> None:
    assert set(RUNTIME_SUPPORTED_BACKENDS) == set(RuntimeBackend)


# ---- RouteChooser ------------------------------------------------------


def test_choose_prefers_promoted_over_staged_over_candidate() -> None:
    reg = registry_of(
        make_pkg("a.candidate", promotion=ModelPromotionStage.CANDIDATE),
        make_pkg("b.promoted", promotion=ModelPromotionStage.PROMOTED),
        make_pkg("c.staged", promotion=ModelPromotionStage.STAGED),
    )
    chooser = RouteChooser(reg)
    result = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR))
    assert result.reason == "ok"
    assert result.chosen is not None
    assert result.chosen.package_id == "b.promoted"


def test_choose_staged_beats_candidate() -> None:
    reg = registry_of(
        make_pkg("a.candidate", promotion=ModelPromotionStage.CANDIDATE),
        make_pkg("b.staged", promotion=ModelPromotionStage.STAGED),
    )
    result = RouteChooser(reg).choose(RouteRequest(slot=ModelSlot.FINAL_ASR))
    assert result.chosen is not None
    assert result.chosen.package_id == "b.staged"


def test_choose_tie_break_lower_ram_then_package_id() -> None:
    reg = registry_of(
        make_pkg("z.small", promotion=ModelPromotionStage.STAGED, min_ram_mb=100),
        make_pkg("a.big", promotion=ModelPromotionStage.STAGED, min_ram_mb=200),
    )
    result = RouteChooser(reg).choose(RouteRequest(slot=ModelSlot.FINAL_ASR))
    assert result.chosen is not None
    assert result.chosen.package_id == "z.small"

    reg2 = registry_of(
        make_pkg("b.pkg", promotion=ModelPromotionStage.STAGED, min_ram_mb=100),
        make_pkg("a.pkg", promotion=ModelPromotionStage.STAGED, min_ram_mb=100),
    )
    result2 = RouteChooser(reg2).choose(RouteRequest(slot=ModelSlot.FINAL_ASR))
    assert result2.chosen is not None
    assert result2.chosen.package_id == "a.pkg"


def test_choose_never_picks_retired() -> None:
    reg = registry_of(
        make_pkg("a.retired", promotion=ModelPromotionStage.RETIRED),
        make_pkg("b.candidate", promotion=ModelPromotionStage.CANDIDATE),
    )
    result = RouteChooser(reg).choose(RouteRequest(slot=ModelSlot.FINAL_ASR))
    assert result.chosen is not None
    assert result.chosen.package_id == "b.candidate"


def test_choose_filters_by_language() -> None:
    reg = registry_of(
        make_pkg("en.only", promotion=ModelPromotionStage.PROMOTED, languages=("en",)),
        make_pkg("multi", promotion=ModelPromotionStage.CANDIDATE, languages=("en", "zh")),
    )
    chooser = RouteChooser(reg)
    result = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, language="zh"))
    assert result.chosen is not None
    assert result.chosen.package_id == "multi"
    # language=None matches everything; PROMOTED en-only wins.
    result_none = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, language=None))
    assert result_none.chosen is not None
    assert result_none.chosen.package_id == "en.only"


def test_choose_filters_by_surface_disallow() -> None:
    reg = registry_of(
        make_pkg(
            "desktop.only",
            promotion=ModelPromotionStage.PROMOTED,
            disallow_surfaces=(SurfaceClass.PHONE_CLASS, SurfaceClass.KEYBOARD_EXTENSION),
        ),
        make_pkg("anywhere", promotion=ModelPromotionStage.CANDIDATE),
    )
    chooser = RouteChooser(reg)
    result = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, surface=SurfaceClass.PHONE_CLASS))
    assert result.chosen is not None
    assert result.chosen.package_id == "anywhere"
    result_desktop = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, surface=SurfaceClass.DESKTOP))
    assert result_desktop.chosen is not None
    assert result_desktop.chosen.package_id == "desktop.only"


def test_choose_filters_by_ram_budget() -> None:
    reg = registry_of(
        make_pkg("heavy", promotion=ModelPromotionStage.PROMOTED, min_ram_mb=4000),
        make_pkg("light", promotion=ModelPromotionStage.CANDIDATE, min_ram_mb=500),
    )
    chooser = RouteChooser(reg)
    result = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, ram_budget_mb=1000))
    assert result.chosen is not None
    assert result.chosen.package_id == "light"
    # Budget exactly at min_ram_mb is allowed (strict > comparison).
    result_exact = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, ram_budget_mb=4000))
    assert result_exact.chosen is not None
    assert result_exact.chosen.package_id == "heavy"


def test_choose_filters_by_streaming_requirement() -> None:
    reg = registry_of(
        make_pkg("batch", promotion=ModelPromotionStage.PROMOTED, streaming=False),
        make_pkg("stream", promotion=ModelPromotionStage.CANDIDATE, streaming=True),
    )
    result = RouteChooser(reg).choose(
        RouteRequest(slot=ModelSlot.FINAL_ASR, requires_streaming=True)
    )
    assert result.chosen is not None
    assert result.chosen.package_id == "stream"


def test_choose_no_eligible_packages_reason() -> None:
    reg = registry_of(make_pkg("en.only", languages=("en",)))
    chooser = RouteChooser(reg)
    result = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, language="fi"))
    assert result.chosen is None
    assert result.reason == "no_eligible_packages"
    # Wrong slot also yields no candidates.
    result_slot = chooser.choose(RouteRequest(slot=ModelSlot.WRITER))
    assert result_slot.chosen is None
    assert result_slot.reason == "no_eligible_packages"


def test_choose_only_considers_requested_slot() -> None:
    reg = registry_of(
        make_pkg("final.pkg", slot=ModelSlot.FINAL_ASR, promotion=ModelPromotionStage.PROMOTED),
        make_pkg("preview.pkg", slot=ModelSlot.PREVIEW_ASR, promotion=ModelPromotionStage.CANDIDATE),
    )
    result = RouteChooser(reg).choose(RouteRequest(slot=ModelSlot.PREVIEW_ASR))
    assert result.chosen is not None
    assert result.chosen.package_id == "preview.pkg"


# ---- ModelRegistry -----------------------------------------------------


def test_registry_rejects_duplicate_package_id() -> None:
    reg = registry_of(make_pkg("dup"))
    with pytest.raises(ValueError, match="already registered"):
        reg.add(make_pkg("dup"))


def test_registry_rejects_unsigned_package_when_trust_keys_set() -> None:
    reg = ModelRegistry(trust_keys={"k": b"\x01" * 32})
    with pytest.raises(ValueError, match="signature_rejected: missing_signature"):
        reg.add(make_pkg("unsigned"))


def test_registry_accepts_signed_package_when_trust_keys_set() -> None:
    from juno_core_v3.model_registry.keystore import sign_package

    key = b"\x01" * 32
    pkg = make_pkg("signed")
    sign_package(pkg, key_id="k", key=key)
    reg = ModelRegistry(trust_keys={"k": key})
    reg.add(pkg)
    assert reg.get("signed") is pkg


def test_registry_promotion_lifecycle_and_rollback() -> None:
    reg = registry_of(
        make_pkg("base"),
        ModelPackage(
            package_id="next",
            version="0.2",
            manifest=make_pkg("x").manifest,
            rollback_target="base",
        ),
    )
    reg.stage("next")
    assert reg.get("next").promotion is ModelPromotionStage.STAGED
    reg.promote("next")
    assert reg.get("next").promotion is ModelPromotionStage.PROMOTED
    reg.retire("next")
    assert reg.get("next").promotion is ModelPromotionStage.RETIRED
    assert reg.rollback("next").package_id == "base"
    with pytest.raises(ValueError, match="no rollback_target"):
        reg.rollback("base")
    with pytest.raises(KeyError):
        reg.promote("missing")


# ---- build_default_registry (hermetic configurations) -------------------


@pytest.fixture()
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the default registry independent of the host configuration.
    monkeypatch.delenv("JUNO_KEYSTORE", raising=False)
    monkeypatch.delenv("JUNO_ALLOW_EXAMPLE_KEYSTORE", raising=False)
    monkeypatch.delenv("JUNO_EVAL_REPORT", raising=False)


def test_build_default_registry_unsigned(_hermetic_env: None) -> None:
    reg = build_default_registry(sign=False)
    ids = {p.package_id for p in reg.list()}
    # Qwen ASR packages were removed with the launch stabilization: ASR is
    # Whisper end-to-end; Qwen serves the writer/planner lanes only.
    assert {
        "tiny.functiongemma",
        "preview.faster-whisper-small-en",
        "final.faster-whisper-medium-en",
        "final.mlx-whisper-large-v3",
        "writer.gemma4-e2b",
        "writer.gemma4-e4b",
    } <= ids
    assert not {pid for pid in ids if "qwen3-asr" in pid}
    assert all(p.signature is None for p in reg.list())


def test_build_default_registry_signed_with_explicit_trust_keys(_hermetic_env: None) -> None:
    from juno_core_v3.model_registry.signature import canonical_payload, verify_signature

    key = b"\xab" * 32
    reg = build_default_registry(trust_keys={"test-key": key})
    packages = reg.list()
    assert packages
    for pkg in packages:
        assert pkg.signature is not None
        assert pkg.signature.algo == "hmac-sha256"
        verdict = verify_signature(
            payload=canonical_payload(pkg.to_dict()),
            signature=pkg.signature,
            trust_keys={"test-key": key},
        )
        assert verdict.ok, f"{pkg.package_id}: {verdict.reason}"


def test_build_default_registry_routing_defaults(_hermetic_env: None) -> None:
    reg = build_default_registry(sign=False)
    chooser = RouteChooser(reg)

    # English final ASR: the STAGED faster-whisper medium wins over candidates.
    final_en = chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, language="en"))
    assert final_en.chosen is not None
    assert final_en.chosen.package_id == "final.faster-whisper-medium-en"

    # Chinese streaming preview: the Qwen ASR packages were removed with
    # the launch stabilization (ASR is Whisper end-to-end), and the staged
    # Whisper preview package is en-only — zh has no eligible package.
    preview_zh = chooser.choose(
        RouteRequest(slot=ModelSlot.PREVIEW_ASR, language="zh", requires_streaming=True)
    )
    assert preview_zh.chosen is None
    assert preview_zh.reason == "no_eligible_packages"
