from __future__ import annotations

from juno_v2.contracts.transforms import CatalogTransform
from juno_v2.transforms.catalog import BUILTIN_CATALOG, get_builtin

EXPECTED_IDS = {
    "polish",
    "fix_grammar",
    "make_shorter",
    "make_longer",
    "make_clearer",
    "make_more_formal",
    "make_more_casual",
    "bulletize",
    "numbered_list",
    "summarize",
    "simplify",
    "translate_preserve_meaning",
    "email_rewrite",
    "slack_rewrite",
    "notes_rewrite",
    "checklist_rewrite",
}


# ---------------------------------------------------------------------------
# get_builtin
# ---------------------------------------------------------------------------


def test_get_builtin_known_id_returns_metadata() -> None:
    t = get_builtin("polish")
    assert isinstance(t, CatalogTransform)
    assert t.transform_id == "polish"
    assert t.display_name == "Polish"
    assert t.model_prompt_template == "Polish wording and spacing. Preserve meaning."


def test_get_builtin_known_id_for_every_catalog_entry() -> None:
    for transform_id in EXPECTED_IDS:
        t = get_builtin(transform_id)
        assert t is not None, transform_id
        assert t.transform_id == transform_id


def test_get_builtin_strips_whitespace() -> None:
    t = get_builtin("  fix_grammar  ")
    assert t is not None and t.transform_id == "fix_grammar"


def test_get_builtin_unknown_id_returns_none() -> None:
    assert get_builtin("does_not_exist") is None
    assert get_builtin("POLISH") is None  # ids are case-sensitive


def test_get_builtin_empty_and_none_return_none() -> None:
    assert get_builtin("") is None
    assert get_builtin("   ") is None
    assert get_builtin(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Catalog enumeration sanity checks
# ---------------------------------------------------------------------------


def test_catalog_contains_exactly_expected_ids() -> None:
    assert set(BUILTIN_CATALOG) == EXPECTED_IDS


def test_catalog_keys_match_entry_transform_ids() -> None:
    for key, entry in BUILTIN_CATALOG.items():
        assert entry.transform_id == key


def test_catalog_entries_have_sane_fields() -> None:
    for entry in BUILTIN_CATALOG.values():
        assert entry.display_name.strip(), entry.transform_id
        assert entry.target_types_supported, entry.transform_id
        assert set(entry.target_types_supported) <= {"selected", "recent", "explicit"}
        assert entry.post_processors == ("trace_metadata",), entry.transform_id
        assert entry.fallback_behavior == "degrade_to_polish", entry.transform_id
        # Every transform must be executable: either a model prompt or at
        # least one deterministic preprocessor.
        assert entry.model_prompt_template or entry.deterministic_preprocessors, (
            entry.transform_id
        )


def test_deterministic_transforms_have_no_model_prompt() -> None:
    for transform_id, preprocessor in (("bulletize", "bullets"), ("numbered_list", "numbered")):
        entry = BUILTIN_CATALOG[transform_id]
        assert entry.model_prompt_template == ""
        assert preprocessor in entry.deterministic_preprocessors


def test_catalog_display_names_are_unique() -> None:
    names = [entry.display_name for entry in BUILTIN_CATALOG.values()]
    assert len(names) == len(set(names))
