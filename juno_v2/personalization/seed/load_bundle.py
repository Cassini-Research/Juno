from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from juno_v2.personalization.seed.models import (
    JunoPersonalizationSeedLayer,
    JunoSeedManifest,
    PromotionPolicy,
    RoutingPolicy,
    SeedTermEntry,
    SuppressedSurfacePolicy,
    SurfaceRoute,
)
from juno_v2.personalization.seed.validate_schema import (
    validate_manifest,
    validate_promotion_policy,
    validate_routing_policy,
    validate_seed_term_row,
    validate_suppressed_surface_policy,
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"juno seed: missing file {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_promotion(data: dict[str, Any]) -> PromotionPolicy:
    validate_promotion_policy(data)
    return PromotionPolicy(
        version=int(data["version"]),
        manual_seed_entries_trusted_immediately=bool(data["manual_seed_entries_trusted_immediately"]),
        manual_user_added_entries_trusted_immediately=bool(
            data["manual_user_added_entries_trusted_immediately"]
        ),
        correction_promote_after=int(data["correction_promote_after"]),
        context_observation_min_count=int(data["context_observation_min_count"]),
        context_acceptance_min_count=int(data["context_acceptance_min_count"]),
        screen_derived_candidates_runtime_only_until_accepted=bool(
            data["screen_derived_candidates_runtime_only_until_accepted"]
        ),
        never_auto_promote_from_suppressed_context=bool(data["never_auto_promote_from_suppressed_context"]),
        max_runtime_candidates_per_surface=int(data["max_runtime_candidates_per_surface"]),
        max_bias_bundle_items_before_clipping=int(data["max_bias_bundle_items_before_clipping"]),
        specialized_pack_requires_explicit_route=bool(
            data.get("specialized_pack_requires_explicit_route", True)
        ),
        notes=str(data.get("notes", "") or ""),
    )


def load_seed_bundle(bundle_dir: Path | str) -> JunoPersonalizationSeedLayer:
    """Load and validate the full seed bundle from ``bundle_dir``."""
    root = Path(bundle_dir).resolve()
    manifest_raw = _read_json(root / "manifest.json")
    validate_manifest(manifest_raw)
    manifest = JunoSeedManifest(
        version=int(manifest_raw["version"]),
        default_enabled_packs=tuple(str(x).strip() for x in manifest_raw["default_enabled_packs"]),
        optional_packs=tuple(str(x).strip() for x in manifest_raw["optional_packs"]),
        promotion_policy=dict(manifest_raw.get("promotion_policy") or {}),
        generated_bundle=str(manifest_raw.get("generated_bundle", "") or ""),
    )

    routing_raw = _read_json(root / "routing_policy.json")
    validate_routing_policy(routing_raw)
    routes: list[SurfaceRoute] = []
    for r in routing_raw["surface_routes"]:
        routes.append(
            SurfaceRoute(
                surface_keywords=tuple(str(x).strip().casefold() for x in r["surface_keywords"]),
                enable_packs=tuple(str(x).strip() for x in r["enable_packs"]),
            )
        )
    routing = RoutingPolicy(
        version=int(routing_raw["version"]),
        always_include=tuple(str(x).strip() for x in routing_raw["always_include"]),
        surface_routes=tuple(routes),
        fallback_enabled_packs=tuple(str(x).strip() for x in routing_raw["fallback_enabled_packs"]),
        notes=str(routing_raw.get("notes", "") or ""),
    )

    promo_path = root / "promotion_policy.json"
    if promo_path.is_file():
        promotion = _parse_promotion(_read_json(promo_path))
    else:
        merged = {
            "version": int(manifest_raw["version"]),
            "manual_seed_entries_trusted_immediately": True,
            "manual_user_added_entries_trusted_immediately": True,
            "correction_promote_after": 1,
            "context_observation_min_count": 3,
            "context_acceptance_min_count": 1,
            "screen_derived_candidates_runtime_only_until_accepted": True,
            "never_auto_promote_from_suppressed_context": True,
            "max_runtime_candidates_per_surface": 48,
            "max_bias_bundle_items_before_clipping": 128,
            "specialized_pack_requires_explicit_route": True,
            "notes": "",
            **manifest.promotion_policy,
        }
        promotion = _parse_promotion(merged)

    sup_raw = _read_json(root / "suppressed_surface_policy.json")
    validate_suppressed_surface_policy(sup_raw)
    suppressed = SuppressedSurfacePolicy(
        version=int(sup_raw["version"]),
        never_persist_from_surfaces=tuple(str(x).strip().casefold() for x in sup_raw["never_persist_from_surfaces"]),
        allow_runtime_only_bias_from_suppressed_surfaces=bool(
            sup_raw["allow_runtime_only_bias_from_suppressed_surfaces"]
        ),
        notes=str(sup_raw.get("notes", "") or ""),
    )

    packs_dir = root / "packs"
    if not packs_dir.is_dir():
        raise FileNotFoundError(f"juno seed: missing packs directory {packs_dir}")

    packs: dict[str, tuple[SeedTermEntry, ...]] = {}
    for path in sorted(packs_dir.glob("*.json")):
        pack_name = path.stem
        raw_list = _read_json(path)
        if not isinstance(raw_list, list):
            raise ValueError(f"juno seed: pack {pack_name} must be a JSON array")
        entries: list[SeedTermEntry] = []
        for idx, row in enumerate(raw_list):
            if not isinstance(row, dict):
                raise ValueError(f"juno seed: pack {pack_name}[{idx}] must be an object")
            norm = validate_seed_term_row(row, where=f"packs/{path.name}[{idx}]")
            entries.append(
                SeedTermEntry(
                    pack_name=pack_name,
                    canonical=norm["canonical"],
                    kind=norm["kind"],
                    aliases=tuple(norm["aliases"]),
                    spoken_forms=tuple(norm["spoken_forms"]),
                    tags=tuple(norm["tags"]),
                    locale=norm["locale"],
                    scope_defaults=tuple(norm["scope_defaults"]),
                    source=norm["source"],
                    confidence=float(norm["confidence"]),
                    notes=norm["notes"],
                )
            )
        packs[pack_name] = tuple(entries)

    entities_path = root / "entities" / "personal_entities.json"
    pe_raw = _read_json(entities_path)
    if not isinstance(pe_raw, list):
        raise ValueError("juno seed: personal_entities.json must be a JSON array")
    personal: list[SeedTermEntry] = []
    for idx, row in enumerate(pe_raw):
        if not isinstance(row, dict):
            raise ValueError(f"juno seed: personal_entities.json[{idx}] must be an object")
        norm = validate_seed_term_row(row, where=f"entities/personal_entities.json[{idx}]")
        personal.append(
            SeedTermEntry(
                pack_name="personal_entities",
                canonical=norm["canonical"],
                kind=norm["kind"],
                aliases=tuple(norm["aliases"]),
                spoken_forms=tuple(norm["spoken_forms"]),
                tags=tuple(norm["tags"]),
                locale=norm["locale"],
                scope_defaults=tuple(norm["scope_defaults"]),
                source=norm["source"],
                confidence=float(norm["confidence"]),
                notes=norm["notes"],
            )
        )

    vr_path = root / "validation_report.json"
    validation_report: dict[str, Any] = {}
    if vr_path.is_file():
        vr = _read_json(vr_path)
        if not isinstance(vr, dict):
            raise ValueError("juno seed: validation_report.json must be an object")
        validation_report = vr

    declared = set(manifest.default_enabled_packs) | set(manifest.optional_packs)
    for path in packs_dir.glob("*.json"):
        if path.stem not in declared:
            raise ValueError(
                f"juno seed: pack file {path.name!r} is not listed in manifest "
                "default_enabled_packs or optional_packs"
            )
    for name in declared:
        if name not in packs:
            raise ValueError(f"juno seed: manifest references pack {name!r} but packs/{name}.json is missing")

    return JunoPersonalizationSeedLayer(
        bundle_dir=root,
        manifest=manifest,
        routing=routing,
        promotion=promotion,
        suppressed_surface=suppressed,
        packs=packs,
        personal_entities=tuple(personal),
        validation_report=validation_report,
    )
