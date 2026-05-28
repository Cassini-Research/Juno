"""Lightweight app-category classifier.

Maps ``app_name`` + ``window_title`` to one of the
:class:`~juno_v2.writer.deterministic.AppCategory` string values
(``messaging``, ``email``, ``docs``, ``code``, ``terminal``, ``forms``,
``unknown``).

The classifier is intentionally rule-based and offline: no ML, no
network, no user configuration. It exists so the writer and context
plane have a consistent presentation category regardless of which
surface produced the context bundle. When we're unsure we return ``unknown`` — the writer's
``unknown`` policy is a pass-through, so a wrong classification never
makes the output worse than the raw dictation.

Rules live in three small tables so they're easy to extend:

* ``_BUNDLE_ID_RULES`` — exact macOS bundle-id prefix matches. These
  are the strongest signal because the shell usually has them.
* ``_APP_NAME_RULES`` — case-insensitive substring matches on the
  human-facing app name. Catches VS Code, Slack, Mail, etc.
* ``_WINDOW_TITLE_RULES`` — substring matches on the window title.
  Used as a fallback when the app name isn't specific enough (e.g.
  "Chrome" hosting Gmail → ``email``).

The classifier does not look at URL, hostname, or OCR. When a better
signal is available the surface can fill the category directly on the
bundle; this module is the default.
"""

from __future__ import annotations

from typing import Iterable

# Ordered from most specific to least specific. First match wins.
_BUNDLE_ID_RULES: tuple[tuple[str, str], ...] = (
    # Terminals
    ("com.apple.terminal", "terminal"),
    ("com.googlecode.iterm2", "terminal"),
    ("co.zeit.hyper", "terminal"),
    ("dev.warp.warp", "terminal"),
    ("io.alacritty", "terminal"),
    ("org.gnu.emacs", "code"),
    # IDEs / code
    ("com.microsoft.vscode", "code"),
    ("com.visualstudio.code", "code"),
    ("com.microsoft.visualstudio.code.insiders", "code"),
    ("com.jetbrains.", "code"),
    ("com.sublimetext.", "code"),
    ("com.github.atom", "code"),
    ("com.apple.dt.xcode", "code"),
    ("com.sourcegear.", "code"),
    # Messaging
    ("com.tinyspeck.slackmacgap", "messaging"),
    ("com.hnc.discord", "messaging"),
    ("com.microsoft.teams", "messaging"),
    ("com.apple.messages", "messaging"),
    ("com.apple.ichat", "messaging"),
    ("ru.keepcoder.telegram", "messaging"),
    ("com.tdesktop.telegram", "messaging"),
    ("com.whatsapp.", "messaging"),
    # Email
    ("com.apple.mail", "email"),
    ("com.microsoft.outlook", "email"),
    ("com.readdle.smartemail-mac", "email"),
    ("com.airmail.", "email"),
    # Dedicated meeting-note surfaces. Conferencing clients are not
    # promoted here; content-based meeting detection happens downstream.
    ("ai.grain.", "meeting"),
    # Docs / notes
    ("com.apple.notes", "docs"),
    ("com.apple.pages", "docs"),
    ("com.apple.textedit", "docs"),
    ("com.microsoft.word", "docs"),
    ("com.google.chrome.app.docs.google.com", "docs"),
    ("notion.id", "docs"),
    ("md.obsidian", "docs"),
    ("com.literatureandlatte.scrivener3", "docs"),
    ("com.culturedcode.thingsmac", "docs"),
)

_APP_NAME_RULES: tuple[tuple[str, str], ...] = (
    ("grain", "meeting"),
    ("meeting notes", "meeting"),
    ("call notes", "meeting"),
    ("transcript notes", "meeting"),
    ("terminal", "terminal"),
    ("iterm", "terminal"),
    ("warp", "terminal"),
    ("hyper", "terminal"),
    ("alacritty", "terminal"),
    ("kitty", "terminal"),
    ("visual studio code", "code"),
    ("vscode", "code"),
    ("xcode", "code"),
    ("intellij", "code"),
    ("pycharm", "code"),
    ("webstorm", "code"),
    ("goland", "code"),
    ("clion", "code"),
    ("android studio", "code"),
    ("sublime text", "code"),
    ("neovim", "code"),
    ("zed", "code"),
    ("slack", "messaging"),
    ("discord", "messaging"),
    ("microsoft teams", "messaging"),
    ("messages", "messaging"),
    ("imessage", "messaging"),
    ("whatsapp", "messaging"),
    ("telegram", "messaging"),
    ("signal", "messaging"),
    ("mail", "email"),
    ("outlook", "email"),
    ("spark", "email"),
    ("airmail", "email"),
    ("thunderbird", "email"),
    ("notes", "docs"),
    ("pages", "docs"),
    ("word", "docs"),
    ("notion", "docs"),
    ("obsidian", "docs"),
    ("bear", "docs"),
    ("craft", "docs"),
    ("scrivener", "docs"),
    ("textedit", "docs"),
    ("ulysses", "docs"),
)

# Window-title keywords. Match is only considered when the app name
# category is ``unknown`` (e.g. a browser hosting a webmail client).
_WINDOW_TITLE_RULES: tuple[tuple[str, str], ...] = (
    ("gmail", "email"),
    ("inbox", "email"),
    ("outlook.com", "email"),
    ("google docs", "docs"),
    ("docs.google.com", "docs"),
    ("notion.so", "docs"),
    ("github.com", "code"),
    ("gitlab.com", "code"),
    ("- stackoverflow", "code"),
    ("slack", "messaging"),
    ("discord", "messaging"),
    ("microsoft teams", "messaging"),
    ("form - ", "forms"),
    ("forms.gle", "forms"),
    ("docs.google.com/forms", "forms"),
    ("typeform", "forms"),
    ("jira", "forms"),
)


def classify_app_category(
    app_name: str | None,
    window_title: str | None,
    *,
    app_bundle_id: str | None = None,
) -> str:
    """Return the coarse app category for the focused surface.

    ``app_bundle_id`` (macOS reverse-DNS identifier) is the strongest
    signal when available. We fall back to a substring match over the
    app name, and finally to the window title. Always returns one of
    the :class:`AppCategory` string values; never raises.
    """
    bundle_id = (app_bundle_id or "").strip().lower()
    if bundle_id:
        for prefix, category in _BUNDLE_ID_RULES:
            if bundle_id.startswith(prefix):
                return category

    name = (app_name or "").strip().lower()
    if name:
        for needle, category in _APP_NAME_RULES:
            if needle in name:
                return category

    title = (window_title or "").strip().lower()
    if title:
        for needle, category in _WINDOW_TITLE_RULES:
            if needle in title:
                return category

    return "unknown"


def iter_known_categories() -> Iterable[str]:
    """Iterate the closed set of categories the classifier can return.

    Useful for tests and UI surfaces that need a dropdown of valid
    values without importing the enum.
    """
    return (
        "messaging",
        "email",
        "docs",
        "code",
        "terminal",
        "forms",
        "meeting",
        "unknown",
    )


__all__ = ["classify_app_category", "iter_known_categories"]
