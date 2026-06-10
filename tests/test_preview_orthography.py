from juno_v2.preview.orthography import normalize_preview_orthography


def test_preview_orthography_does_not_trust_mid_sentence_capitalization() -> None:
    committed, tail, meta = normalize_preview_orthography(
        "Don't say that to me there is image I don't care Make one.",
        "",
    )

    assert committed == "Don't say that to me there is image I don't care make one."
    assert tail == ""
    assert meta["preview_orthography_committed_changed"] is True


def test_preview_orthography_lowers_false_sentence_boundaries_inside_committed_text() -> None:
    committed, _, _ = normalize_preview_orthography(
        "based on my... Reputation. And credibility. Uh, need you to send a mail.",
        "",
    )

    assert committed == (
        "Based on my... reputation. and credibility. uh, need you to send a mail."
    )


def test_preview_orthography_preserves_protected_terms_after_false_boundary() -> None:
    committed, _, _ = normalize_preview_orthography(
        "use the local model. Gemma should stay capitalized. Gamma should not be forced.",
        "",
        protected_terms=["Gemma"],
    )

    assert committed == (
        "Use the local model. Gemma should stay capitalized. gamma should not be forced."
    )


def test_preview_orthography_does_not_lower_unknown_names_without_boundary() -> None:
    committed, _, _ = normalize_preview_orthography(
        "I met Ishida and Lumare is in the roadmap.",
        "",
    )

    assert committed == "I met Ishida and Lumare is in the roadmap."


def test_preview_orthography_lowers_ordinary_mid_sentence_inflections() -> None:
    committed, _, _ = normalize_preview_orthography(
        "we need to fix Formatting especially Finally earlier also.",
        "",
    )

    assert committed == "We need to fix formatting especially finally earlier also."
