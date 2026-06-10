from __future__ import annotations

import re

from juno_v2.commands.grammar import parse_deterministic_command
from juno_v2.contracts.modes import ModePolicy
from juno_v2.contracts.writer import WriterIntent, WriterIntentKind, WriterMode
from juno_v2.language.normalize import summarize_scripts

_COMMAND_STARTERS = frozenset({
    'make', 'rewrite', 'fix', 'convert', 'turn', 'change', 'add', 'remember',
    'new', 'switch', 'use', 'activate', 'summarize', 'summarise', 'correct',
    'expand', 'shorten', 'simplify', 'please', 'can', 'could', 'rephrase',
    'paraphrase', 'clean', 'tighten', 'improve', 'update', 'uppercase', 'lowercase', 'titlecase', 'title',
    'start', 'stop', 'end', 'begin', 'next', 'continue',
})
_MAX_COMMAND_WORDS = 32
_MAX_SELECTION_FALLBACK_WORDS = 14
_MAX_SELECTION_TRANSFORM_WORDS = 64
_SEMANTIC_TARGET_RE = re.compile(r"\b(?:that|this|it|selection|selected)\b", re.I)

_INLINE_FORMATTING_RULES = [
    (re.compile(r'^new\s+paragraph\.?$', re.I), 'new_paragraph', '\n\n'),
    (re.compile(r'^(?:go\s+to\s+)?(?:new\s+line|newline)\.?$', re.I), 'new_line', '\n'),
    (re.compile(r'^(new\s+bullet|next\s+(bullet|item))\.?$', re.I), 'new_bullet', '\n- '),
    (re.compile(r'^next\s+number\.?$', re.I), 'next_numbered', ''),
]

_STRUCTURE_MODE_RULES = [
    (re.compile(r'^(start|begin)\s+(bullet\s+list|bullets?)\.?$', re.I), 'bullets'),
    (re.compile(r'^(start|begin)\s+(numbered\s+list|numbering)\.?$', re.I), 'numbered'),
    (re.compile(r'^(stop|end)\s+(bullet\s+list|bullets?|numbered\s+list|numbering|list)\.?$', re.I), None),
]

_MODE_RULES = [
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:email|mail|outlook)\s*(?:mode)?\b', re.I), WriterMode.FORMAL_EMAIL),
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:casual|chat|relaxed|informal)\s*(?:mode)?\b', re.I), WriterMode.CASUAL_CHAT),
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:verbatim|exact|transcription|raw)\s*(?:mode)?\b', re.I), WriterMode.VERBATIM),
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:command|commands)\s*(?:mode)?\b', re.I), WriterMode.COMMAND_MODE),
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:code|technical|dev|coding)\s*(?:mode)?\b', re.I), WriterMode.TECHNICAL_PRECISE),
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:notes?|doc(?:ument)?|structured|writing)\s*(?:mode)?\b', re.I), WriterMode.STRUCTURED_NOTES),
    (re.compile(r'\b(?:switch|use|go\s+to|activate|enable)\s+(?:to\s+)?(?:rewrite|creative|free)\s*(?:mode)?\b', re.I), WriterMode.EXPLICIT_REWRITE),
]

_ADD_TERM_RULES = [
    re.compile(r"\badd\s+(.+?)\s+to\s+(?:my\s+)?(?:dictionary|lexicon|vocab(?:ulary)?|wordlist)\b", re.I),
    re.compile(r"\bremember\s+(?:the\s+word\s+|the\s+term\s+)?[\"']?(.+?)[\"']?\s+(?:as\s+a\s+word|in\s+my\s+dictionary)\b", re.I),
]

_ADD_REPLACEMENT_RULES = [
    (re.compile(r"\bremember\s+(?:that\s+)?[\"']?(.+?)[\"']?\s+(?:is|means|equals|should\s+be)\s+[\"']?(.+?)[\"']?\s*$", re.I), 1, 2),
    (re.compile(r"\balways\s+(?:replace|change|use)\s+[\"']?(.+?)[\"']?\s+(?:with|as)\s+[\"']?(.+?)[\"']?\s*$", re.I), 1, 2),
    (re.compile(r"\b(?:replace|change)\s+[\"']?(.+?)[\"']?\s+(?:to|with)\s+[\"']?(.+?)[\"']?\s*$", re.I), 1, 2),
]

_DETERMINISTIC_TRANSFORMS = [
    (re.compile(r'\b(?:(?:convert|turn|change|put)\s+)?(this|it)\s*(?:into\s+)?(bullet\s+points?|bullets|a\s+bulleted\s+list|a\s+list)\b', re.I), 'bullets'),
    (re.compile(r'\b(?:(?:convert|turn|change)\s+)?(this|it)\s*(?:into\s+)?(numbered|a\s+numbered\s+list)\b', re.I), 'numbered'),
    (re.compile(r'\b(?:uppercase\s+)?(this|it)\s*(?:in\s+)?uppercase\b|\buppercase\s+(this|it)\b', re.I), 'uppercase'),
    (re.compile(r'\b(?:lowercase\s+)?(this|it)\s*(?:in\s+)?lowercase\b|\blowercase\s+(this|it)\b', re.I), 'lowercase'),
    (re.compile(r'\b(?:title\s*case|titlecase)\s+(this|it)\b|\b(this|it)\s+(?:in\s+)?(?:title\s*case|titlecase)\b', re.I), 'title_case'),
]

_MODEL_TRANSFORMS = [
    (re.compile(r'\b(?:(?:make|rewrite)\s+)?(this|it)\s+(?:more\s+)?(formal|professional|polished)\b|\b(make|rewrite)\s+(this|it)\s+(?:more\s+)?(formal|professional|polished)\b', re.I), 'Rewrite in a formal, professional tone. Preserve meaning.'),
    (re.compile(r'\b(?:(?:make|rewrite)\s+)?(this|it)\s+(?:more\s+)?(casual|informal|friendly|conversational)\b|\b(make|rewrite)\s+(this|it)\s+(?:more\s+)?(casual|informal|friendly|conversational)\b', re.I), 'Rewrite in a casual, friendly tone. Preserve meaning.'),
    (re.compile(r'\b(fix|correct|clean\s+up)\s+(the\s+)?(grammar|spelling|punctuation|errors|typos)\b', re.I), 'Fix grammar, spelling, and punctuation. Preserve meaning exactly.'),
    (re.compile(r'\b(summarize|summarise|give\s+me\s+a\s+summary|create\s+a\s+summary)\b', re.I), 'Summarize the selected text into concise key points.'),
    (re.compile(r'\b(simplify|make\s+(this|it)\s+simpler|make\s+(this|it)\s+easier\s+to\s+read)\b', re.I), 'Simplify the selected text. Preserve meaning.'),
    (re.compile(r'\b(improve|tighten|polish)\s+(this|it)\b', re.I), 'Improve clarity and flow. Preserve meaning.'),
    (re.compile(r'\b(make|keep)\s+(this|it)\s+(more\s+)?(concise|shorter|brief|short)\b', re.I), 'Make the selected text more concise while preserving meaning.'),
    (re.compile(r'\b(make|expand)\s+(this|it)\s+(more\s+)?(longer|detailed|elaborate|comprehensive)\b', re.I), 'Expand the selected text with more detail and context.'),
    # Phase 6a — close the silent-failure gap on six catalog transforms
    # whose spoken commands previously routed to DICTATE (translate /
    # email / slack / notes / checklist) or to semantic_candidate
    # (clearer). These map to the corresponding entries in
    # ``juno_v2/transforms/catalog.py``.
    #
    # All anchored at ``^`` so they only fire when the utterance LEADS
    # with a command verb (after polite-prefix strip). Anchoring stops
    # mid-sentence false positives like
    #   "let's make this clearer in the next meeting agenda"   (9 words)
    #   "we should turn this into an email campaign"            (8 words)
    # from being misclassified as transform commands. Existing
    # _MODEL_TRANSFORMS above are NOT anchored — that's a pre-existing
    # parser behaviour we don't change in this PR.
    #
    # ``(this|it)`` is optional so both forms work:
    #   "rewrite this as an email"   (with target pronoun)
    #   "rewrite as an email"        (without — natural shorthand)
    (re.compile(r'^(?:make|rewrite)\s+(this|it)\s+(?:more\s+)?(clearer|easier\s+to\s+understand)\b|^clarify\s+(this|it)\b', re.I), 'Improve clarity. Preserve meaning.'),
    (re.compile(r'^(?:rewrite|turn|make)(?:\s+(?:this|it))?(?:\s+(?:into|as|for))?\s+(?:an?\s+)?email\b', re.I), 'Rewrite as a polished email.'),
    (re.compile(r'^(?:rewrite|turn|make)(?:\s+(?:this|it))?(?:\s+(?:into|as|for))?\s+(?:an?\s+)?slack(?:\s+message)?\b', re.I), 'Rewrite as a concise Slack message.'),
    (re.compile(r'^(?:rewrite|turn|make)(?:\s+(?:this|it))?(?:\s+(?:into|as|for))?\s+(?:an?\s+)?(?:structured\s+)?notes\b', re.I), 'Rewrite as structured notes with bullets where helpful.'),
    (re.compile(r'^(?:rewrite|turn|make)(?:\s+(?:this|it))?(?:\s+(?:into|as|for))?\s+(?:an?\s+)?checklist\b', re.I), 'Rewrite as a short checklist.'),
]


# Languages we can recognize as the target of a "translate to ..." command.
# The list is conservative — only common written languages — so we don't
# misinterpret (e.g.) a person's name or a topic as a translation target.
# Multi-word entries first so the alternation matches "brazilian portuguese"
# before the bare "portuguese" alternative.
_TRANSLATE_LANGS = (
    r"(?:brazilian\s+portuguese|simplified\s+chinese|traditional\s+chinese|"
    r"latin\s+american\s+spanish|"
    r"english|spanish|french|german|italian|portuguese|dutch|polish|"
    r"swedish|norwegian|danish|finnish|russian|ukrainian|"
    r"czech|slovak|hungarian|romanian|bulgarian|greek|turkish|"
    r"hebrew|arabic|persian|farsi|hindi|urdu|bengali|tamil|telugu|"
    r"marathi|gujarati|punjabi|kannada|malayalam|"
    r"chinese|mandarin|cantonese|japanese|korean|"
    r"thai|vietnamese|indonesian|malay|tagalog|filipino|"
    r"swahili|afrikaans|amharic)"
)
# Translate is its own pattern so we can capture the target language
# group and inject it into the LLM instruction. Anchored at ``^`` for
# the same reason as the patterns above — without the anchor a
# sentence like "I'll send the translation to spanish team" would
# misroute to MODEL_TRANSFORM.
_TRANSLATE_PATTERN = re.compile(
    rf'^translate\s+(?:(?:this|it)\s+)?to\s+(?P<lang>{_TRANSLATE_LANGS})\b',
    re.IGNORECASE,
)
_TRANSLATE_RECENT_PATTERN = re.compile(
    rf'^translate\s+(?:that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+to\s+(?P<lang>{_TRANSLATE_LANGS})\b',
    re.IGNORECASE,
)

_RECENT_DETERMINISTIC_TRANSFORMS = [
    (re.compile(r'\b(?:make|turn|rewrite|change|convert)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:into\s+)?(bullet\s+points?|bullets)\b', re.I), 'bullets'),
    (re.compile(r'\b(?:make|turn|rewrite|change|convert)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:into\s+)?(numbered|a\s+numbered\s+list)\b', re.I), 'numbered'),
]

_RECENT_MODEL_TRANSFORMS = [
    (re.compile(r'\b(?:make|rewrite)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:more\s+)?(formal|professional|polished)\b', re.I), 'Rewrite in a formal, professional tone. Preserve meaning.'),
    (re.compile(r'\b(?:make|rewrite)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:more\s+)?(casual|informal|friendly|conversational)\b', re.I), 'Rewrite in a casual, friendly tone. Preserve meaning.'),
    (re.compile(r'\b(?:summarize|summarise|shorten|simplify)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\b', re.I), 'Summarize the selected text into concise key points.'),
    (re.compile(r'\b(?:fix|correct|clean\s+up)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s*(?:grammar|spelling|punctuation|errors|typos)?\b', re.I), 'Fix grammar, spelling, and punctuation. Preserve meaning exactly.'),
    # Phase 6a — recent-target ("that") variants of the FOUR catalog
    # transforms added to _MODEL_TRANSFORMS that don't already have a
    # deterministic-command grammar entry. ``commands/grammar.py``
    # already handles ``make that clearer``/``translate that to <lang>``
    # via parse_deterministic_command's _RECENT_PATTERNS — those route
    # to ``COMMAND_RESULT`` with kind=recent_edit/translate, which the
    # writer service then dispatches to the LLM. Adding redundant
    # patterns here would shadow the existing routing without any
    # benefit.
    (re.compile(r'^(?:rewrite|turn|make)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:into\s+)?(?:as\s+)?(?:an?\s+)?email\b', re.I), 'Rewrite as a polished email.'),
    (re.compile(r'^(?:rewrite|turn|make)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:into\s+|for\s+|as\s+)?(?:an?\s+)?slack(?:\s+message)?\b|^rewrite\s+(?:that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+for\s+slack\b', re.I), 'Rewrite as a concise Slack message.'),
    (re.compile(r'^(?:rewrite|turn|make)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:into\s+)?(?:as\s+)?(?:an?\s+)?(?:structured\s+)?notes\b', re.I), 'Rewrite as structured notes with bullets where helpful.'),
    (re.compile(r'^(?:rewrite|turn|make)\s+(that|the\s+last\s+(?:sentence|line|paragraph|answer|thing))\s+(?:into\s+)?(?:as\s+)?(?:an?\s+)?checklist\b', re.I), 'Rewrite as a short checklist.'),
]

# Patterns for inserting a field value from the current screen context.
# Each tuple is (pattern, context_field_name).
# `focused_file_path` — the document/file the user is editing.
# `symbol_under_cursor` — the identifier straddling the caret in a code editor.
_INSERT_CONTEXT_FIELD_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r'^(?:insert|type|paste|add|put|write)\s+(?:the\s+)?(?:current\s+)?(?:file\s+(?:name|path)|filename|filepath)\.?$',
        re.I,
    ), 'focused_file_path'),
    (re.compile(
        r'^(?:insert|type|paste|add|put|write)\s+(?:the\s+)?(?:current\s+)?(?:symbol|function\s+name|method\s+name|variable\s+name|identifier|function|method|variable)\.?$',
        re.I,
    ), 'symbol_under_cursor'),
    (re.compile(
        r'^(?:tag|at)\s+(?:the\s+)?(?:current\s+)?(?:file|filename|document)\.?$',
        re.I,
    ), 'focused_file_path'),
    (re.compile(
        r'^(?:tag|at)\s+(?:the\s+)?(?:current\s+)?(?:symbol|function|method|variable|identifier)\.?$',
        re.I,
    ), 'symbol_under_cursor'),
]


class WriterIntentParser:
    def parse(
        self,
        text: str,
        *,
        language_hint: str | None = None,
        selection_present: bool = False,
        active_mode: WriterMode | None = None,
        mode_policy: ModePolicy | None = None,
        partial_text: str | None = None,
    ) -> WriterIntent:
        return self._parse(
            text,
            language_hint=language_hint,
            selection_present=selection_present,
            active_mode=active_mode,
            mode_policy=mode_policy,
            partial_text=partial_text,
        )

    def _parse(
        self,
        text: str,
        *,
        language_hint: str | None = None,
        selection_present: bool = False,
        active_mode: WriterMode | None = None,
        mode_policy: ModePolicy | None = None,
        partial_text: str | None = None,
    ) -> WriterIntent:
        text = (text or '').strip()
        if not text:
            return WriterIntent(kind=WriterIntentKind.NOOP, raw_text=text, metadata={'reason': 'empty'})
        text = _strip_polite_prefix(text)
        words = text.split()
        if active_mode == WriterMode.COMMAND_MODE and len(words) > 12 and not selection_present:
            return WriterIntent(
                kind=WriterIntentKind.DICTATE,
                raw_text=text,
                metadata={'reason': 'command_mode_narrow_dictation', 'word_count': len(words)},
            )
        if not _english_command_safe(text, words, language_hint=language_hint):
            return WriterIntent(kind=WriterIntentKind.DICTATE, raw_text=text, metadata={'reason': 'non_command_language'})

        selection_transform = _selection_transform_intent(text, words, selection_present=selection_present)
        if selection_transform is not None:
            return selection_transform

        if len(words) > _MAX_COMMAND_WORDS:
            return WriterIntent(
                kind=WriterIntentKind.DICTATE,
                raw_text=text,
                metadata={'reason': 'too_long_for_command', 'word_count': len(words)},
            )

        det = parse_deterministic_command(text)
        if det is not None:
            return WriterIntent(
                kind=WriterIntentKind.COMMAND_RESULT,
                raw_text=text,
                metadata={'deterministic_command': {'name': det.name, 'kind': det.kind, 'payload': det.payload}},
            )

        for pattern, structure_mode in _STRUCTURE_MODE_RULES:
            if pattern.match(text):
                return WriterIntent(kind=WriterIntentKind.SET_STRUCTURE_MODE, raw_text=text, structure_mode=structure_mode)

        for pattern, transform_kind, insert_text in _INLINE_FORMATTING_RULES:
            if pattern.match(text):
                return WriterIntent(kind=WriterIntentKind.INSERT_FORMATTING, raw_text=text, insert_text=insert_text, transform_kind=transform_kind)

        for pattern, context_field in _INSERT_CONTEXT_FIELD_RULES:
            if pattern.match(text):
                return WriterIntent(kind=WriterIntentKind.INSERT_CONTEXT_FIELD, raw_text=text, context_field=context_field)

        for pattern, mode in _MODE_RULES:
            if pattern.search(text):
                return WriterIntent(kind=WriterIntentKind.SWITCH_MODE, raw_text=text, mode=mode)

        for pattern, trig_grp, repl_grp in _ADD_REPLACEMENT_RULES:
            match = pattern.search(text)
            if match:
                trigger = match.group(trig_grp).strip().strip("'\"")
                replacement = match.group(repl_grp).strip().strip("'\"")
                if trigger and replacement and len(trigger) < 80 and len(replacement) < 80:
                    return WriterIntent(kind=WriterIntentKind.ADD_REPLACEMENT, raw_text=text, trigger=trigger, replacement=replacement)

        for pattern in _ADD_TERM_RULES:
            match = pattern.search(text)
            if match:
                term = match.group(1).strip().strip("'\"")
                if term and len(term) < 80 and len(term.split()) <= 6:
                    return WriterIntent(kind=WriterIntentKind.ADD_TERM, raw_text=text, term=term)

        # Mode-policy-aware command gating.
        #
        # Three orthogonal axes:
        #   - recent-target commands ("rewrite that", "delete the last sentence") —
        #     gated by mode_policy.allow_recent_target_commands. Verbatim sets
        #     this True per c0feb74 ("verbatim allows explicit user commands").
        #   - LLM-rewrite commands (RECENT_MODEL_TRANSFORMS / MODEL_TRANSFORMS) —
        #     additionally gated by mode_policy.allow_model_insert_rewrite. Verbatim
        #     sets this False so the model never silently rewrites.
        #   - non-targeted deterministic transforms ("make this a bullet list" with
        #     no recent/selection qualifier) — these reshape the active dictation
        #     and are blocked when command_behavior == 'strict_narrow'. Verbatim
        #     uses strict_narrow so they stay blocked.
        #
        # The selection_present block below runs unconditionally and is governed
        # by mode_policy.allow_selection_commands. Verbatim sets that True too.
        narrow_command_behavior = active_mode == WriterMode.VERBATIM or (
            mode_policy is not None
            and getattr(mode_policy, 'command_behavior', '') == 'strict_narrow'
        )
        allow_recent_target = (
            mode_policy is None
            or getattr(mode_policy, 'allow_recent_target_commands', True)
        )
        allow_model_rewrite = (
            mode_policy is None
            or getattr(mode_policy, 'allow_model_insert_rewrite', True)
        )
        allow_unscoped_transform = _unscoped_transform_command_allowed(words)

        if allow_recent_target:
            for pattern, transform_kind in _RECENT_DETERMINISTIC_TRANSFORMS:
                if pattern.search(text):
                    return WriterIntent(kind=WriterIntentKind.RECENT_DETERMINISTIC_TRANSFORM, raw_text=text, transform_kind=transform_kind)

            if allow_model_rewrite:
                # Translate-recent: capture the target language so it
                # reaches the LLM instruction. Phase 6a.
                m_recent_translate = _TRANSLATE_RECENT_PATTERN.search(text)
                if m_recent_translate is not None:
                    lang = m_recent_translate.group("lang").strip()
                    return WriterIntent(
                        kind=WriterIntentKind.RECENT_MODEL_TRANSFORM,
                        raw_text=text,
                        instruction=f"Translate faithfully to {lang}. Preserve meaning and tone where possible.",
                    )
                for pattern, instruction in _RECENT_MODEL_TRANSFORMS:
                    if pattern.search(text):
                        return WriterIntent(kind=WriterIntentKind.RECENT_MODEL_TRANSFORM, raw_text=text, instruction=instruction)

        if not narrow_command_behavior and allow_unscoped_transform:
            for pattern, transform_kind in _DETERMINISTIC_TRANSFORMS:
                if pattern.search(text):
                    return WriterIntent(kind=WriterIntentKind.DETERMINISTIC_TRANSFORM, raw_text=text, transform_kind=transform_kind)

            if allow_model_rewrite:
                # Translate (selection / unscoped): capture the target
                # language so it reaches the LLM instruction. Phase 6a.
                m_translate = _TRANSLATE_PATTERN.search(text)
                if m_translate is not None:
                    lang = m_translate.group("lang").strip()
                    return WriterIntent(
                        kind=WriterIntentKind.MODEL_TRANSFORM,
                        raw_text=text,
                        instruction=f"Translate faithfully to {lang}. Preserve meaning and tone where possible.",
                    )
                for pattern, instruction in _MODEL_TRANSFORMS:
                    if pattern.search(text):
                        return WriterIntent(kind=WriterIntentKind.MODEL_TRANSFORM, raw_text=text, instruction=instruction)

        if selection_present and _selection_fallback_allowed(text, words):
            lower_text = text.casefold()
            if 'bullet' in lower_text:
                return WriterIntent(kind=WriterIntentKind.DETERMINISTIC_TRANSFORM, raw_text=text, transform_kind='bullets', metadata={'reason': 'selection_fallback'})
            if 'numbered' in lower_text:
                return WriterIntent(kind=WriterIntentKind.DETERMINISTIC_TRANSFORM, raw_text=text, transform_kind='numbered', metadata={'reason': 'selection_fallback'})
            if any(token in lower_text for token in ('formal', 'professional', 'polished')):
                return WriterIntent(kind=WriterIntentKind.MODEL_TRANSFORM, raw_text=text, instruction='Rewrite in a formal, professional tone. Preserve meaning.', metadata={'reason': 'selection_fallback'})
            if any(token in lower_text for token in ('casual', 'informal', 'friendly', 'conversational')):
                return WriterIntent(kind=WriterIntentKind.MODEL_TRANSFORM, raw_text=text, instruction='Rewrite in a casual, friendly tone. Preserve meaning.', metadata={'reason': 'selection_fallback'})
            if any(token in lower_text for token in ('grammar', 'spelling', 'punctuation', 'typos')):
                return WriterIntent(kind=WriterIntentKind.MODEL_TRANSFORM, raw_text=text, instruction='Fix grammar, spelling, and punctuation. Preserve meaning exactly.', metadata={'reason': 'selection_fallback'})

        if _looks_like_semantic_command(text, words):
            return WriterIntent(
                kind=WriterIntentKind.COMMAND_RESULT,
                raw_text=text,
                metadata={'deterministic_command': {'name': 'semantic_candidate', 'kind': 'semantic_candidate', 'payload': {}}},
            )

        return WriterIntent(kind=WriterIntentKind.DICTATE, raw_text=text)


def _english_command_safe(text: str, words: list[str], *, language_hint: str | None) -> bool:
    summary = summarize_scripts(text)
    starter_tokens = [re.sub(r'[^a-z]', '', token.lower()) for token in words[:3]]
    has_english_prefix = any(token in _COMMAND_STARTERS for token in starter_tokens if token)
    if summary.devanagari > 0 or summary.thai > 0:
        return has_english_prefix
    if (language_hint or '').lower().startswith(('hi', 'th')):
        return has_english_prefix
    return True


def _strip_polite_prefix(text: str) -> str:
    return re.sub(
        r'^\s*(?:please[,\s]+|can\s+you[,\s]+|could\s+you[,\s]+|would\s+you[,\s]+|hey[,\s]+|okay[,\s]+|ok[,\s]+|u+m+[,\.\s]+|uh+[,\.\s]+|a+h+[,\.\s]+|a{2,}[,\.\s]+)+',
        '',
        text,
        flags=re.I,
    ).strip()


def _selection_transform_intent(
    text: str,
    words: list[str],
    *,
    selection_present: bool,
) -> WriterIntent | None:
    if not selection_present:
        return None
    if len(words) > _MAX_SELECTION_TRANSFORM_WORDS:
        return None
    first = re.sub(r'[^a-z]', '', words[0].lower()) if words else ""
    if first not in _COMMAND_STARTERS:
        return None
    if _SEMANTIC_TARGET_RE.search(text) is None:
        return None
    lowered = text.casefold()
    if re.search(r"\b(?:bullet\s+points?|bullets|bulleted\s+list|a\s+list)\b", lowered):
        return WriterIntent(
            kind=WriterIntentKind.DETERMINISTIC_TRANSFORM,
            raw_text=text,
            transform_kind='bullets',
            metadata={'reason': 'selection_transform_command'},
        )
    if re.search(r"\b(?:numbered|numbered\s+list)\b", lowered):
        return WriterIntent(
            kind=WriterIntentKind.DETERMINISTIC_TRANSFORM,
            raw_text=text,
            transform_kind='numbered',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('formal', 'professional', 'polished')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Rewrite in a formal, professional tone. Preserve meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('casual', 'informal', 'friendly', 'conversational')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Rewrite in a casual, friendly tone. Preserve meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('grammar', 'spelling', 'punctuation', 'typos')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Fix grammar, spelling, and punctuation. Preserve meaning exactly.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('concise', 'shorter', 'brief')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Make the text more concise. Preserve meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('clearer', 'clarity')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Improve clarity. Preserve meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('summarize', 'summarise', 'summary')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Summarize into concise key points. Preserve core meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('expand', 'longer', 'detailed', 'elaborate')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Expand with useful detail while preserving meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    if any(token in lowered for token in ('simplify', 'simpler', 'easier to read')):
        return WriterIntent(
            kind=WriterIntentKind.MODEL_TRANSFORM,
            raw_text=text,
            instruction='Simplify the text. Preserve meaning.',
            metadata={'reason': 'selection_transform_command'},
        )
    return None


def _looks_like_semantic_command(text: str, words: list[str]) -> bool:
    if not words:
        return False
    starter = re.sub(r'[^a-z]', '', words[0].lower())
    if starter not in _COMMAND_STARTERS:
        return False
    lowered = text.casefold()
    if _SEMANTIC_TARGET_RE.search(text) is None:
        return False
    semantic_markers = (
        'concise', 'shorter', 'brief', 'clearer', 'clarity', 'formal', 'professional',
        'polished', 'casual', 'friendly', 'conversational', 'informal', 'grammar',
        'spelling', 'punctuation', 'typos', 'summarize', 'summarise', 'summary',
        'bullet', 'numbered', 'expand', 'detailed', 'simplify', 'simpler',
    )
    return any(marker in lowered for marker in semantic_markers)


def _unscoped_transform_command_allowed(words: list[str]) -> bool:
    if not words:
        return False
    starter = re.sub(r'[^a-z]', '', words[0].lower())
    return starter in _COMMAND_STARTERS


def _selection_fallback_allowed(text: str, words: list[str]) -> bool:
    if len(words) > _MAX_SELECTION_FALLBACK_WORDS:
        return False
    if _SEMANTIC_TARGET_RE.search(text):
        return True
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            'selected text',
            'highlighted text',
            'selection',
        )
    )
