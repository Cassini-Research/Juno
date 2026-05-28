import Foundation

enum LivePreviewTextMerger {
    private struct Token {
        let folded: String
        let range: Range<String.Index>
    }

    static func merge(existingVisible: String, engineText: String) -> String? {
        let visible = existingVisible.trimmingCharacters(in: .whitespacesAndNewlines)
        let engine = engineText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !engine.isEmpty else { return nil }
        guard !visible.isEmpty else { return engine }

        let visibleFolded = foldedPhrase(visible)
        let engineFolded = foldedPhrase(engine)
        if engineFolded.hasPrefix(visibleFolded) {
            return engine
        }
        if visibleFolded.hasPrefix(engineFolded) {
            return visible
        }

        let visibleTokens = tokens(in: visible)
        let engineTokens = tokens(in: engine)
        guard visibleTokens.count >= 2, engineTokens.count >= 2 else { return nil }
        let limit = min(visibleTokens.count, engineTokens.count, 8)
        for size in stride(from: limit, through: 2, by: -1) {
            let visibleSuffix = visibleTokens.suffix(size).map(\.folded)
            let enginePrefix = engineTokens.prefix(size).map(\.folded)
            guard Array(visibleSuffix) == Array(enginePrefix) else { continue }
            guard engineTokens.count > size else { return visible }
            let tailStart = engineTokens[size].range.lowerBound
            let tail = String(engine[tailStart...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !tail.isEmpty else { return visible }
            return "\(visible) \(tail)"
        }
        return nil
    }

    private static func foldedPhrase(_ value: String) -> String {
        tokens(in: value).map(\.folded).joined(separator: " ")
    }

    private static func tokens(in value: String) -> [Token] {
        var out: [Token] = []
        var start: String.Index?
        var idx = value.startIndex
        while idx < value.endIndex {
            let scalar = value[idx]
            if scalar.isLetter || scalar.isNumber || scalar == "'" {
                if start == nil { start = idx }
            } else if let tokenStart = start {
                appendToken(value[tokenStart..<idx], range: tokenStart..<idx, to: &out)
                start = nil
            }
            idx = value.index(after: idx)
        }
        if let tokenStart = start {
            appendToken(value[tokenStart..<value.endIndex], range: tokenStart..<value.endIndex, to: &out)
        }
        return out
    }

    private static func appendToken(_ raw: Substring, range: Range<String.Index>, to out: inout [Token]) {
        let folded = raw.lowercased()
        switch folded {
        case "i'd":
            out.append(Token(folded: "i", range: range))
            out.append(Token(folded: "would", range: range))
        case "i'm":
            out.append(Token(folded: "i", range: range))
            out.append(Token(folded: "am", range: range))
        default:
            out.append(Token(folded: folded, range: range))
        }
    }
}
