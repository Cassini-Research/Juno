from __future__ import annotations

from typing import Any

from juno_v2.memory.term_policy import learned_term_allowed


def _require_str(obj: dict[str, Any], key: str, *, where: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where}: missing or invalid string field {key!r}")
    return v.strip()


def _require_list_str(obj: dict[str, Any], key: str, *, where: str) -> list[str]:
    v = obj.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"{where}: {key!r} must be a list")
    out: list[str] = []
    for i, item in enumerate(v):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{where}: {key}[{i}] must be a non-empty string")
        out.append(item.strip())
    return out


def _require_float(obj: dict[str, Any], key: str, *, where: str) -> float:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: {key!r} must be a number")
    return float(v)


def validate_seed_term_row(raw: dict[str, Any], *, where: str) -> dict[str, Any]:
    """Validate one pack / entity row; returns normalized dict."""
    canonical = _require_str(raw, "canonical", where=where)
    if "personal_entities" in where and not learned_term_allowed(canonical):
        raise ValueError(f"{where}: canonical must be at least 3 alphanumeric characters")
    kind = _require_str(raw, "kind", where=f"{where} canonical={canonical!r}")
    aliases = _require_list_str(raw, "aliases", where=f"{where} canonical={canonical!r}")
    spoken = _require_list_str(raw, "spoken_forms", where=f"{where} canonical={canonical!r}")
    tags = _require_list_str(raw, "tags", where=f"{where} canonical={canonical!r}")
    locale = _require_str(raw, "locale", where=f"{where} canonical={canonical!r}")
    scope_defaults = _require_list_str(raw, "scope_defaults", where=f"{where} canonical={canonical!r}")
    source = _require_str(raw, "source", where=f"{where} canonical={canonical!r}")
    confidence = _require_float(raw, "confidence", where=f"{where} canonical={canonical!r}")
    notes = raw.get("notes", "")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{where}: notes must be a string")
    return {
        "canonical": canonical,
        "kind": kind,
        "aliases": aliases,
        "spoken_forms": spoken,
        "tags": tags,
        "locale": locale,
        "scope_defaults": scope_defaults,
        "source": source,
        "confidence": confidence,
        "notes": (notes or "").strip(),
    }


def validate_routing_policy(data: dict[str, Any]) -> None:
    where = "routing_policy.json"
    if int(data.get("version", -1)) < 0:
        raise ValueError(f"{where}: invalid version")
    ai = data.get("always_include")
    if not isinstance(ai, list) or not all(isinstance(x, str) and x.strip() for x in ai):
        raise ValueError(f"{where}: always_include must be a list of non-empty strings")
    routes = data.get("surface_routes")
    if not isinstance(routes, list):
        raise ValueError(f"{where}: surface_routes must be a list")
    for i, r in enumerate(routes):
        if not isinstance(r, dict):
            raise ValueError(f"{where}: surface_routes[{i}] must be an object")
        sk = r.get("surface_keywords")
        ep = r.get("enable_packs")
        if not isinstance(sk, list) or not sk:
            raise ValueError(f"{where}: surface_routes[{i}].surface_keywords invalid")
        if not isinstance(ep, list) or not ep:
            raise ValueError(f"{where}: surface_routes[{i}].enable_packs invalid")
        for j, s in enumerate(sk):
            if not isinstance(s, str) or not s.strip():
                raise ValueError(f"{where}: surface_routes[{i}].surface_keywords[{j}] invalid")
        for j, p in enumerate(ep):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"{where}: surface_routes[{i}].enable_packs[{j}] invalid")
    fb = data.get("fallback_enabled_packs")
    if not isinstance(fb, list) or not fb:
        raise ValueError(f"{where}: fallback_enabled_packs invalid")


def validate_promotion_policy(data: dict[str, Any]) -> None:
    where = "promotion_policy.json"
    if int(data.get("version", -1)) < 0:
        raise ValueError(f"{where}: invalid version")
    keys = (
        "manual_seed_entries_trusted_immediately",
        "manual_user_added_entries_trusted_immediately",
        "screen_derived_candidates_runtime_only_until_accepted",
        "never_auto_promote_from_suppressed_context",
    )
    for k in keys:
        if k not in data or not isinstance(data[k], bool):
            raise ValueError(f"{where}: {k} must be a bool")
    for k in (
        "correction_promote_after",
        "context_observation_min_count",
        "context_acceptance_min_count",
        "max_runtime_candidates_per_surface",
        "max_bias_bundle_items_before_clipping",
    ):
        v = data.get(k)
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{where}: {k} must be a non-negative int")
    if "specialized_pack_requires_explicit_route" in data and not isinstance(
        data["specialized_pack_requires_explicit_route"], bool
    ):
        raise ValueError(f"{where}: specialized_pack_requires_explicit_route must be bool")


def validate_suppressed_surface_policy(data: dict[str, Any]) -> None:
    where = "suppressed_surface_policy.json"
    if int(data.get("version", -1)) < 0:
        raise ValueError(f"{where}: invalid version")
    lst = data.get("never_persist_from_surfaces")
    if not isinstance(lst, list) or not all(isinstance(x, str) and x.strip() for x in lst):
        raise ValueError(f"{where}: never_persist_from_surfaces invalid")
    if "allow_runtime_only_bias_from_suppressed_surfaces" not in data or not isinstance(
        data["allow_runtime_only_bias_from_suppressed_surfaces"], bool
    ):
        raise ValueError(f"{where}: allow_runtime_only_bias_from_suppressed_surfaces must be bool")


def validate_manifest(data: dict[str, Any]) -> None:
    where = "manifest.json"
    if int(data.get("version", -1)) < 0:
        raise ValueError(f"{where}: invalid version")
    for key in ("default_enabled_packs", "optional_packs"):
        v = data.get(key)
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            raise ValueError(f"{where}: {key} invalid")
    pp = data.get("promotion_policy")
    if pp is not None and not isinstance(pp, dict):
        raise ValueError(f"{where}: promotion_policy must be an object or omitted")
