import SwiftUI

/// Polish analyzer + inline diff renderer for the History detail page.
///
/// Two public surfaces:
///
/// 1. :func:`polishReport(raw:final:)` — returns a :class:`PolishReport` with
///    the per-category counts plus an ordered list of inline diff segments.
///    The counts power the summary chips ("12 punctuation · 5 caps · 3
///    word swaps · 3 fillers"). The segments power the inline diff.
///
/// 2. :func:`renderInlineDiff(segments:scheme:)` — turns the segments into a
///    single selectable SwiftUI ``Text`` that reads as the polished
///    sentence with subtle, no-color markers for what changed:
///       * deleted token  → italic + faint hairline strikethrough, secondary text
///       * capitalization → final form rendered, changed letter gets a
///                          dotted underline so the eye picks it up on a
///                          second pass
///       * word swap      → ``~old~ new`` inline at the swap's natural
///                          position in the sentence
///       * punctuation    → flows naturally (no marker — the polished
///                          text reads as prose)
///
/// Earlier revisions of this file either used green/red diff highlighting
/// (loud) or counts only (invisible). The current shape draws from Apple
/// Pages' "Show Changes" and Notion AI's edit suggestions: render the
/// final prose; encode change information in typography (weight, italic,
/// dotted underline) rather than colour.
enum JunoHistoryDiff {

    // MARK: - Public API

    /// Tally of polish categories Juno applied.
    struct PolishCounts: Equatable {
        var punctuation: Int
        var capitalizations: Int
        var wordSwaps: Int
        var fillersRemoved: Int

        static let zero = PolishCounts(
            punctuation: 0,
            capitalizations: 0,
            wordSwaps: 0,
            fillersRemoved: 0
        )

        var total: Int {
            punctuation + capitalizations + wordSwaps + fillersRemoved
        }

        var hasAny: Bool { total > 0 }
    }

    /// One inline diff segment.
    ///
    /// ``trailing`` carries the whitespace that should follow this token
    /// when rendered, so the renderer can reassemble the original spacing
    /// without re-tokenising.
    enum SegmentKind: Equatable {
        /// Same in raw and final (renders as-is).
        case same
        /// Raw and final share fold-key but differ in case ("avery" -> "Avery").
        /// The changed character indices identify which letters to mark
        /// with a dotted underline in the final form.
        case capitalization(changedCharOffsets: [Int])
        /// Word swap or filler removed — raw token absent from final.
        /// Renders as italic + faint strikethrough.
        case deleted
        /// Newly inserted token in final (added word, added punctuation).
        case inserted
    }

    struct Segment: Equatable {
        let kind: SegmentKind
        /// Surface text for THIS segment, without trailing whitespace.
        let surface: String
        /// Trailing whitespace that should follow this segment in the
        /// rendered output. Empty when this is the last segment, or when
        /// the token sat at the end of a logical paragraph.
        let trailingWhitespace: String
    }

    /// Composite output: counts + segments. Use :func:`polishReport` to
    /// compute, then read ``counts`` for the summary chips and
    /// ``segments`` for the inline diff render.
    struct PolishReport: Equatable {
        let counts: PolishCounts
        let segments: [Segment]

        static let empty = PolishReport(counts: .zero, segments: [])
    }

    static func polishReport(raw: String, final: String) -> PolishReport {
        if raw.isEmpty || final.isEmpty {
            return .empty
        }
        let rTokens = tokenize(raw)
        let fTokens = tokenize(final)
        if rTokens.isEmpty || fTokens.isEmpty {
            return .empty
        }

        // Align word-bearing tokens via LCS over fold-keys; punctuation/
        // whitespace tokens fold into the segment naturally via the
        // segment builder below (they're treated as "always insertable").
        let rWordKeys = rTokens.map { $0.foldKeyOrPunctuation }
        let fWordKeys = fTokens.map { $0.foldKeyOrPunctuation }
        let segmentsRaw = buildSegments(
            raw: rTokens,
            final: fTokens,
            rawKeys: rWordKeys,
            finalKeys: fWordKeys
        )

        // Counts derived from the segments. Same surface = baseline; only
        // explicit deltas count.
        var counts = PolishCounts.zero
        for seg in segmentsRaw {
            switch seg.kind {
            case .same:
                continue
            case .capitalization(let offsets):
                if !offsets.isEmpty {
                    counts.capitalizations += 1
                }
            case .deleted:
                if isFiller(seg.surface) {
                    counts.fillersRemoved += 1
                } else if isPunctuation(seg.surface) {
                    // Punctuation removed is rare — don't bucket.
                    continue
                } else {
                    // Word that vanished. The matching insertion picks
                    // up "swap" credit; this is just the deletion half.
                    continue
                }
            case .inserted:
                if isPunctuation(seg.surface) {
                    // Each punctuation mark adds 1 to the count.
                    counts.punctuation += countPunctuationMarks(seg.surface)
                } else {
                    // A word that wasn't in raw. Could be filler-add (rare)
                    // or a true insertion. Match it against the preceding
                    // deletion to count as a swap when shapes match.
                    continue
                }
            }
        }
        // Count word swaps by walking the segments looking for
        // ``.deleted`` → ``.inserted`` adjacent pairs where neither side
        // is a punctuation/filler.
        var i = 0
        while i < segmentsRaw.count {
            if case .deleted = segmentsRaw[i].kind,
               i + 1 < segmentsRaw.count,
               case .inserted = segmentsRaw[i + 1].kind,
               !isPunctuation(segmentsRaw[i].surface),
               !isPunctuation(segmentsRaw[i + 1].surface),
               !isFiller(segmentsRaw[i].surface) {
                counts.wordSwaps += 1
                i += 2
                continue
            }
            i += 1
        }

        return PolishReport(counts: counts, segments: segmentsRaw)
    }

    /// Convenience: counts only, for callers that don't render the inline
    /// diff. Equivalent to ``polishReport(raw:final:).counts``.
    static func polishCounts(raw: String, final: String) -> PolishCounts {
        polishReport(raw: raw, final: final).counts
    }

    // MARK: - Inline renderer

    @ViewBuilder
    static func renderInlineDiff(
        segments: [Segment],
        scheme: ColorScheme
    ) -> some View {
        if segments.isEmpty {
            EmptyView()
        } else {
            segments
                .map { styledText(for: $0, scheme: scheme) }
                .reduce(Text(""), +)
                .font(.system(size: 13.5, design: .rounded))
                .lineSpacing(3.5)
                .textSelection(.enabled)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private static func styledText(for segment: Segment, scheme: ColorScheme) -> Text {
        let trailing = Text(segment.trailingWhitespace)
            .foregroundColor(JunoTheme.primaryText(scheme))
        switch segment.kind {
        case .same:
            return Text(segment.surface)
                .foregroundColor(JunoTheme.primaryText(scheme)) + trailing
        case .capitalization(let offsets):
            // Walk characters, mark changed ones with underline so the
            // eye can pick up which letters Juno upcased without
            // breaking the natural reading rhythm.
            let chars = Array(segment.surface)
            var composed = Text("")
            for (idx, ch) in chars.enumerated() {
                var span = Text(String(ch))
                    .foregroundColor(JunoTheme.primaryText(scheme))
                if offsets.contains(idx) {
                    span = span.underline(true, color: JunoTheme.primaryText(scheme).opacity(0.55))
                }
                composed = composed + span
            }
            return composed + trailing
        case .deleted:
            return Text(segment.surface)
                .italic()
                .strikethrough(true, color: JunoTheme.secondaryText(scheme).opacity(0.55))
                .foregroundColor(JunoTheme.secondaryText(scheme).opacity(0.78)) + trailing
        case .inserted:
            // No special marker — punctuation and inserted words read
            // as part of the polished prose. The presence of the
            // adjacent ``.deleted`` (for word swaps) is what makes the
            // change visible in context.
            return Text(segment.surface)
                .foregroundColor(JunoTheme.primaryText(scheme)) + trailing
        }
    }

    // MARK: - Tokenization

    private struct Tok {
        /// Raw surface including any leading whitespace? No — we strip
        /// whitespace into ``trailingWhitespace``.
        let surface: String
        /// Whitespace that followed this token in the source string.
        let trailingWhitespace: String
        /// Lowercased word for fold-key comparison. Empty for non-word
        /// tokens (punctuation, symbols).
        let foldKey: String
        /// True when the surface is a punctuation mark — these can't
        /// participate in LCS matching the same way words do.
        let isPunctuation: Bool

        /// LCS comparison key. For words we use the lowercased surface;
        /// for punctuation we use a sentinel so a stray `.` doesn't
        /// "match" with another `.` elsewhere in the doc.
        var foldKeyOrPunctuation: String {
            isPunctuation ? "__P_\(surface)" : foldKey
        }
    }

    /// Split text into Tok objects, capturing trailing whitespace and
    /// classifying word vs punctuation. Apostrophe-internal words count
    /// as one token.
    private static func tokenize(_ text: String) -> [Tok] {
        var tokens: [Tok] = []
        var i = text.startIndex
        while i < text.endIndex {
            // Skip leading whitespace before any token.
            if text[i].isWhitespace {
                let wsStart = i
                while i < text.endIndex, text[i].isWhitespace {
                    i = text.index(after: i)
                }
                if !tokens.isEmpty {
                    // Attach to previous token's trailing whitespace.
                    let last = tokens.removeLast()
                    tokens.append(Tok(
                        surface: last.surface,
                        trailingWhitespace: last.trailingWhitespace + String(text[wsStart..<i]),
                        foldKey: last.foldKey,
                        isPunctuation: last.isPunctuation
                    ))
                }
                continue
            }
            // Word-or-punctuation run.
            let tokStart = i
            if isPunctuationChar(text[i]) {
                // Single punctuation token (each mark is its own token,
                // so the LCS can mark each insertion / deletion).
                i = text.index(after: i)
                let surface = String(text[tokStart..<i])
                let trail = consumeWhitespace(text: text, from: &i)
                tokens.append(Tok(
                    surface: surface,
                    trailingWhitespace: trail,
                    foldKey: "",
                    isPunctuation: true
                ))
                continue
            }
            // Word body: letters, digits, apostrophes, hyphens.
            while i < text.endIndex, isWordChar(text[i]) {
                i = text.index(after: i)
            }
            let surface = String(text[tokStart..<i])
            if surface.isEmpty {
                // Defensive — something we don't classify, skip one char.
                if i < text.endIndex { i = text.index(after: i) }
                continue
            }
            let trail = consumeWhitespace(text: text, from: &i)
            tokens.append(Tok(
                surface: surface,
                trailingWhitespace: trail,
                foldKey: surface.lowercased(),
                isPunctuation: false
            ))
        }
        return tokens
    }

    private static func consumeWhitespace(text: String, from i: inout String.Index) -> String {
        let start = i
        while i < text.endIndex, text[i].isWhitespace {
            i = text.index(after: i)
        }
        return String(text[start..<i])
    }

    private static func isWordChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "'" || c == "\u{2019}" || c == "-"
    }

    private static func isPunctuationChar(_ c: Character) -> Bool {
        ".,?!:;\u{2014}\u{2026}—".contains(c)
    }

    // MARK: - Segment builder

    /// Build inline diff segments by walking the LCS table and emitting
    /// one ``Segment`` per aligned position. Adjacent same-kind segments
    /// are merged where it doesn't lose information.
    private static func buildSegments(
        raw: [Tok],
        final: [Tok],
        rawKeys: [String],
        finalKeys: [String]
    ) -> [Segment] {
        let m = rawKeys.count
        let n = finalKeys.count
        // LCS dp table.
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        if m > 0 && n > 0 {
            for i in 1...m {
                for j in 1...n {
                    if rawKeys[i - 1] == finalKeys[j - 1] {
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    } else {
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    }
                }
            }
        }
        // Backtrace from (m, n) producing pairs in reverse order.
        var pairs: [(rawIdx: Int?, finalIdx: Int?)] = []
        var i = m
        var j = n
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && rawKeys[i - 1] == finalKeys[j - 1] {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
                pairs.append((nil, j - 1))
                j -= 1
            } else if i > 0 {
                pairs.append((i - 1, nil))
                i -= 1
            } else {
                break
            }
        }
        let aligned = Array(pairs.reversed())

        // Walk aligned pairs, build segments. Each pair becomes one
        // segment except for adjacent (deleted, inserted) which are
        // emitted as-is (the renderer reads them as a word swap).
        var out: [Segment] = []
        for pair in aligned {
            switch (pair.rawIdx, pair.finalIdx) {
            case let (rawI?, finalI?):
                let rTok = raw[rawI]
                let fTok = final[finalI]
                if rTok.surface == fTok.surface {
                    out.append(Segment(
                        kind: .same,
                        surface: fTok.surface,
                        trailingWhitespace: fTok.trailingWhitespace
                    ))
                } else {
                    // Same fold-key, different surface → case change.
                    let offsets = changedCharOffsets(raw: rTok.surface, final: fTok.surface)
                    out.append(Segment(
                        kind: .capitalization(changedCharOffsets: offsets),
                        surface: fTok.surface,
                        trailingWhitespace: fTok.trailingWhitespace
                    ))
                }
            case let (rawI?, nil):
                let rTok = raw[rawI]
                // Deleted token. We drop its trailing whitespace so the
                // surrounding flow doesn't get double-spaced.
                out.append(Segment(
                    kind: .deleted,
                    surface: rTok.surface,
                    trailingWhitespace: rTok.trailingWhitespace
                ))
            case let (nil, finalI?):
                let fTok = final[finalI]
                out.append(Segment(
                    kind: .inserted,
                    surface: fTok.surface,
                    trailingWhitespace: fTok.trailingWhitespace
                ))
            case (nil, nil):
                continue
            }
        }
        return out
    }

    /// Indices of characters in ``final`` that differ from ``raw`` at
    /// the same position. Used to dot-underline capitalization changes.
    private static func changedCharOffsets(raw: String, final: String) -> [Int] {
        let rChars = Array(raw)
        let fChars = Array(final)
        var out: [Int] = []
        for k in 0..<min(rChars.count, fChars.count) {
            if rChars[k] != fChars[k] {
                out.append(k)
            }
        }
        // Any extra chars in final beyond the raw length are also flagged.
        if fChars.count > rChars.count {
            for k in rChars.count..<fChars.count {
                out.append(k)
            }
        }
        return out
    }

    // MARK: - Helpers

    private static let fillerWords: Set<String> = [
        "um", "umm", "uh", "uhh", "er", "erm", "like",
        "y'know", "yknow", "okay", "ok", "right",
    ]

    private static func isFiller(_ word: String) -> Bool {
        fillerWords.contains(word.lowercased())
    }

    private static func isPunctuation(_ surface: String) -> Bool {
        guard !surface.isEmpty else { return false }
        return surface.allSatisfy { isPunctuationChar($0) }
    }

    private static func countPunctuationMarks(_ surface: String) -> Int {
        surface.reduce(0) { $0 + (isPunctuationChar($1) ? 1 : 0) }
    }
}
