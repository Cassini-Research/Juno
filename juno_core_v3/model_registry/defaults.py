from __future__ import annotations

import logging
from typing import Mapping

from juno_core_v3.model_registry.contracts import ModelPromotionStage, ModelSlot, RuntimeBackend, SurfaceClass
from juno_core_v3.model_registry.eval_report import apply_eval_report, load_eval_report
from juno_core_v3.model_registry.keystore import load_keystore, sign_packages
from juno_core_v3.model_registry.manifest import CapabilityManifest
from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry

_logger = logging.getLogger(__name__)


def build_default_registry(
    *,
    trust_keys: Mapping[str, bytes] | None = None,
    sign: bool | None = None,
) -> ModelRegistry:
    """Build the default local model registry.

    ``trust_keys`` is an explicit override used by tests. When it is
    ``None`` we call :func:`load_keystore`, which honours explicit
    configuration only. The committed example key is ignored unless
    ``JUNO_ALLOW_EXAMPLE_KEYSTORE=1`` is set.

    Set ``sign=False`` to skip signing (returns an unsigned,
    ``trust_keys=None`` registry) — useful for legacy callers and
    tests that construct their own signed packages. Default is
    ``True`` when a keystore could be resolved, ``False`` otherwise.
    """
    if trust_keys is None:
        try:
            trust_keys = load_keystore()
        except ValueError as exc:
            # A broken keystore is a configuration error. Surface it
            # loudly instead of silently running with no trust keys,
            # which would regress to the pre-P2 behaviour.
            raise RuntimeError(f"default_registry_keystore_invalid: {exc}") from exc

    if sign is None:
        sign = trust_keys is not None

    reg = ModelRegistry(trust_keys=trust_keys if sign else None)

    # Build every bundled package first, sign the collection, then add
    # to the registry. ``ModelRegistry.add`` verifies signatures when
    # ``trust_keys`` is set, so signing up-front keeps the flow
    # symmetric with externally supplied trust configuration.
    packages: list[ModelPackage] = []

    packages.append(
        ModelPackage(
            package_id="tiny.functiongemma",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.TINY_ROUTER,
                backend=RuntimeBackend.LOCAL_HTTP_JSON,
                languages=("en",),
                quantizations=("int8", "fp16"),
                streaming=False,
                min_ram_mb=256,
                warm_load_target_s=0.2,
                thermal_class="desktop",
            ),
            rollback_target=None,
            promotion=ModelPromotionStage.CANDIDATE,
            metadata={"note": "optional tiny router; rules-first remains primary"},
        )
    )

    # Preview ASR slot — only backends supported by the runtime factory.
    packages.append(
        ModelPackage(
            package_id="preview.faster-whisper-small-en",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.PREVIEW_ASR,
                backend=RuntimeBackend.FASTER_WHISPER,
                languages=("en",),
                quantizations=("int8", "fp16"),
                streaming=True,
                min_ram_mb=600,
                warm_load_target_s=1.0,
                thermal_class="desktop",
            ),
            rollback_target=None,
            promotion=ModelPromotionStage.STAGED,
            metadata={"note": "default preview ASR — faster_whisper small.en", "model_path": ".juno_v2_demo/models/faster-whisper-small-en"},
        )
    )
    # Qwen3-ASR preview (0.6B) — optional multilingual streaming backend via
    # the moona3k/mlx-qwen3-asr MLX port. CANDIDATE only; operators opt
    # in via the registry CLI once weights have been pulled.
    packages.append(
        ModelPackage(
            package_id="preview.qwen3-asr-0.6b",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.PREVIEW_ASR,
                backend=RuntimeBackend.QWEN_ASR,
                languages=("en", "zh", "es", "ja", "ko", "fr", "de", "pt", "ru", "ar"),
                quantizations=("fp16", "8bit", "4bit"),
                streaming=True,
                min_ram_mb=1400,
                warm_load_target_s=1.5,
                thermal_class="desktop",
                disallow_surfaces=(SurfaceClass.PHONE_CLASS, SurfaceClass.KEYBOARD_EXTENSION),
            ),
            rollback_target=None,
            promotion=ModelPromotionStage.CANDIDATE,
            metadata={
                "note": "optional multilingual streaming backend (Qwen3-ASR-0.6B via MLX)",
                "model_path": "Qwen/Qwen3-ASR-0.6B",
                "hf_repo_id": "Qwen/Qwen3-ASR-0.6B",
            },
        )
    )

    # Final ASR slot — faster_whisper is the universally supported runtime backend.
    packages.append(
        ModelPackage(
            package_id="final.faster-whisper-medium-en",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.FINAL_ASR,
                backend=RuntimeBackend.FASTER_WHISPER,
                languages=("en",),
                quantizations=("int8", "fp16"),
                streaming=False,
                min_ram_mb=1500,
                warm_load_target_s=2.0,
                thermal_class="desktop",
            ),
            rollback_target=None,
            promotion=ModelPromotionStage.STAGED,
            metadata={"note": "default final ASR — faster_whisper medium.en", "model_path": ".juno_v2_demo/models/faster-whisper-medium-en"},
        )
    )
    packages.append(
        ModelPackage(
            package_id="final.mlx-whisper-large-v3",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.FINAL_ASR,
                backend=RuntimeBackend.MLX_WHISPER,
                languages=("en", "hi", "es", "zh", "ja", "ko"),
                quantizations=("int4", "int8"),
                streaming=False,
                min_ram_mb=3000,
                warm_load_target_s=3.5,
                thermal_class="desktop",
                disallow_surfaces=(SurfaceClass.PHONE_CLASS, SurfaceClass.KEYBOARD_EXTENSION),
            ),
            rollback_target="final.faster-whisper-medium-en",
            promotion=ModelPromotionStage.CANDIDATE,
            metadata={"note": "Mac-only MLX final backend; rollback to faster_whisper", "hf_repo_id": "mlx-community/whisper-large-v3-turbo"},
        )
    )
    # Qwen3-ASR final (1.7B) — optional multilingual one-shot backend via the
    # moona3k/mlx-qwen3-asr MLX port. CANDIDATE; rolls back to the
    # faster_whisper medium default if withdrawn.
    packages.append(
        ModelPackage(
            package_id="final.qwen3-asr-1.7b",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.FINAL_ASR,
                backend=RuntimeBackend.QWEN_ASR,
                languages=("en", "zh", "es", "ja", "ko", "fr", "de", "pt", "ru", "ar", "hi"),
                quantizations=("fp16", "8bit", "4bit"),
                streaming=False,
                min_ram_mb=3600,
                warm_load_target_s=3.0,
                thermal_class="desktop",
                disallow_surfaces=(SurfaceClass.PHONE_CLASS, SurfaceClass.KEYBOARD_EXTENSION),
            ),
            rollback_target="final.faster-whisper-medium-en",
            promotion=ModelPromotionStage.CANDIDATE,
            metadata={
                "note": "multilingual Mac-only final backend (Qwen3-ASR-1.7B via MLX); rollback to faster_whisper",
                "model_path": "Qwen/Qwen3-ASR-1.7B",
                "hf_repo_id": "Qwen/Qwen3-ASR-1.7B",
            },
        )
    )

    # Writer slot candidates.
    packages.append(
        ModelPackage(
            package_id="writer.gemma4-e2b",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.WRITER,
                backend=RuntimeBackend.MLX_LM,
                languages=("en",),
                quantizations=("4bit", "8bit"),
                streaming=False,
                min_ram_mb=1200,
                warm_load_target_s=1.0,
                thermal_class="phone_class",
            ),
            rollback_target=None,
            promotion=ModelPromotionStage.CANDIDATE,
        )
    )
    packages.append(
        ModelPackage(
            package_id="writer.gemma4-e4b",
            version="0.1",
            manifest=CapabilityManifest(
                slot=ModelSlot.WRITER,
                backend=RuntimeBackend.MLX_LM,
                languages=("en",),
                quantizations=("4bit", "8bit"),
                streaming=False,
                min_ram_mb=1800,
                warm_load_target_s=1.2,
                thermal_class="desktop",
            ),
            rollback_target="writer.gemma4-e2b",
            promotion=ModelPromotionStage.STAGED,
            metadata={"note": "Mac-class writer candidate; rollback to E2B"},
        )
    )

    # Overlay measured metrics (WER / latency) before signing so the
    # signed payload is a single self-describing unit. Missing report
    # → packages keep ``None`` for metrics, which the chooser must
    # treat as "unknown, don't rank on quality/latency".
    try:
        report = load_eval_report()
    except ValueError as exc:
        raise RuntimeError(f"default_registry_eval_report_invalid: {exc}") from exc
    if report:
        updated = apply_eval_report(packages, report)
        _logger.debug(
            "default_registry_eval_metrics_applied: %d packages updated", updated
        )

    if sign and trust_keys:
        sign_packages(packages, trust_keys=trust_keys)
        _logger.debug(
            "default_registry_signed: %d packages, key_ids=%s",
            len(packages),
            sorted(trust_keys.keys()),
        )

    for pkg in packages:
        reg.add(pkg)

    return reg
