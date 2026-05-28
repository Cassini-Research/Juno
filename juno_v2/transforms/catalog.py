from __future__ import annotations

from juno_v2.contracts.transforms import CatalogTransform

_IDS = (
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
)


def _ct(
    transform_id: str,
    display_name: str,
    *,
    targets: tuple[str, ...],
    template: str,
    preprocessors: tuple[str, ...] = (),
    modes: tuple[str, ...] = (),
) -> CatalogTransform:
    return CatalogTransform(
        transform_id=transform_id,
        display_name=display_name,
        target_types_supported=targets,
        mode_constraints=modes,
        deterministic_preprocessors=preprocessors,
        model_prompt_template=template,
        post_processors=("trace_metadata",),
        fallback_behavior="degrade_to_polish",
    )


BUILTIN_CATALOG: dict[str, CatalogTransform] = {
    "polish": _ct("polish", "Polish", targets=("selected", "recent", "explicit"), template="Polish wording and spacing. Preserve meaning."),
    "fix_grammar": _ct("fix_grammar", "Fix grammar", targets=("selected", "recent", "explicit"), template="Fix grammar, spelling, and punctuation. Preserve meaning."),
    "make_shorter": _ct("make_shorter", "Make shorter", targets=("selected", "recent", "explicit"), template="Make the text more concise. Preserve meaning."),
    "make_longer": _ct("make_longer", "Make longer", targets=("selected", "recent", "explicit"), template="Expand with useful detail. Preserve meaning."),
    "make_clearer": _ct("make_clearer", "Make clearer", targets=("selected", "recent", "explicit"), template="Improve clarity. Preserve meaning."),
    "make_more_formal": _ct("make_more_formal", "More formal", targets=("selected", "recent", "explicit"), template="Rewrite in a formal professional tone. Preserve meaning."),
    "make_more_casual": _ct("make_more_casual", "More casual", targets=("selected", "recent", "explicit"), template="Rewrite in a casual friendly tone. Preserve meaning."),
    "bulletize": _ct("bulletize", "Bullets", targets=("selected", "recent", "explicit"), template="", preprocessors=("bullets",)),
    "numbered_list": _ct("numbered_list", "Numbered list", targets=("selected", "recent", "explicit"), template="", preprocessors=("numbered",)),
    "summarize": _ct("summarize", "Summarize", targets=("selected", "recent", "explicit"), template="Summarize into concise key points."),
    "simplify": _ct("simplify", "Simplify", targets=("selected", "recent", "explicit"), template="Simplify the language. Preserve meaning."),
    "translate_preserve_meaning": _ct(
        "translate_preserve_meaning",
        "Translate",
        targets=("selected", "recent", "explicit"),
        template="Translate faithfully. Preserve meaning and tone where possible.",
    ),
    "email_rewrite": _ct("email_rewrite", "Email rewrite", targets=("selected", "recent", "explicit"), template="Rewrite as a polished email."),
    "slack_rewrite": _ct("slack_rewrite", "Slack rewrite", targets=("selected", "recent", "explicit"), template="Rewrite as a concise Slack message."),
    "notes_rewrite": _ct("notes_rewrite", "Notes rewrite", targets=("selected", "recent", "explicit"), template="Rewrite as structured notes with bullets where helpful."),
    "checklist_rewrite": _ct("checklist_rewrite", "Checklist", targets=("selected", "recent", "explicit"), template="Rewrite as a short checklist."),
}


def get_builtin(transform_id: str) -> CatalogTransform | None:
    return BUILTIN_CATALOG.get((transform_id or "").strip())
