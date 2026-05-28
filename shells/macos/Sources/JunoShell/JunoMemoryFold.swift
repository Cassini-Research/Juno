import Foundation

/// Match-normalization for the memory layer — Swift mirror of
/// `juno_v2.memory.fold.fold_key`.
///
/// Every Swift-side dedup, protected-term check, and conflict comparison
/// against vocabulary / snippets / replacements / corrections must use
/// the *same* canonical key the Python broker uses, otherwise the UI's
/// "is this already saved?" verdict will disagree with the server's
/// "already_known" / "vocab_conflict" verdict and the user sees ghost
/// duplicates / contradictory errors.
///
/// The rule is one structural character-class transform — not a list
/// of tuned cases:
///
///     fold_key(s) = NFKD decompose
///                 → drop combining marks (diacritics, accents)
///                 → casefold (Unicode-aware lowercasing)
///                 → drop everything outside [a-z0-9]
///
/// So `"signoff"`, `"sign off"`, `"Sign-Off"`, `"SIGNOFF"`, `"sign_off"`
/// all collapse to `"signoff"`. `"naïve"` and `"naive"` collapse to
/// `"naive"`. Empty input or all-punctuation input returns the empty
/// string — callers must treat `""` as "no usable key".
enum JunoMemoryFold {
    static func foldKey(_ raw: String?) -> String {
        guard let raw, !raw.isEmpty else { return "" }
        // Foundation's `folding(options:)` handles diacritic stripping
        // (NFKD + drop combining marks) and case-insensitive lowercasing
        // in one Unicode-aware pass. Width-insensitive folds halfwidth
        // CJK punctuation to ASCII so a stray fullwidth space collapses
        // along with regular whitespace.
        let collapsed = raw.folding(
            options: [.diacriticInsensitive, .caseInsensitive, .widthInsensitive],
            locale: nil
        )
        var out = ""
        out.reserveCapacity(collapsed.count)
        for scalar in collapsed.unicodeScalars {
            // [a-z0-9] only — same as Python's `[^a-z0-9]+` strip after
            // casefold. Non-Latin scripts survive the casefold but get
            // dropped here, which matches the Python behaviour and is the
            // intended bias: fold_key is for Latin-shape match
            // normalization. Non-Latin triggers fall through to exact
            // string equality in callers, which is still correct.
            switch scalar.value {
            case 0x30...0x39, 0x61...0x7A:
                out.unicodeScalars.append(scalar)
            default:
                continue
            }
        }
        return out
    }

    /// Convenience: returns nil for empty-fold inputs so callers can use
    /// `guard let key = JunoMemoryFold.foldKeyOrNil(raw)` pattern.
    static func foldKeyOrNil(_ raw: String?) -> String? {
        let key = foldKey(raw)
        return key.isEmpty ? nil : key
    }
}
