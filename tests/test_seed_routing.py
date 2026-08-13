from __future__ import annotations

from pathlib import Path

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.personalization.seed.load_bundle import load_seed_bundle
from juno_v2.personalization.seed.routing import select_active_packs


def _seed():
    return load_seed_bundle(Path(__file__).resolve().parents[1] / "seed_data")


def _packs(app_name: str, *, window_title: str = "") -> set[str]:
    selection = select_active_packs(
        _seed(),
        TypedContextBundle(app_name=app_name, window_title=window_title),
    )
    return set(selection.pack_ids)


def test_single_letter_x_route_does_not_match_inside_textedit() -> None:
    assert "domain_colloquial_slang" not in _packs("TextEdit")


def test_single_letter_x_route_does_not_match_inside_xcode() -> None:
    packs = _packs("Xcode")
    assert "domain_coding_engineering" in packs
    assert "domain_colloquial_slang" not in packs


def test_single_letter_x_route_matches_complete_app_token() -> None:
    assert "domain_colloquial_slang" in _packs("X", window_title="X - Home")


def test_multiword_surface_routes_still_match_complete_phrases() -> None:
    assert "domain_product_work" in _packs(
        "Chrome",
        window_title="Launch plan - Google Docs",
    )


def test_surface_keyword_does_not_match_inside_longer_word() -> None:
    assert "domain_colloquial_slang" not in _packs("Keynotes")
