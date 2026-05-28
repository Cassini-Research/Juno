import Foundation

enum HUDCommittedTextReveal {
    private static let maxSteps = 96

    static func steps(current: String, target: String) -> [String] {
        let currentTrimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetTrimmed = target.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !targetTrimmed.isEmpty else { return [] }
        if !currentTrimmed.isEmpty && !targetTrimmed.hasPrefix(currentTrimmed) {
            return []
        }
        let suffixStart = targetTrimmed.index(
            targetTrimmed.startIndex,
            offsetBy: currentTrimmed.count
        )
        let suffixCharacters = Array(targetTrimmed[suffixStart...])
        guard suffixCharacters.count > 1 else { return [] }
        let chunkSize = max(1, Int(ceil(Double(suffixCharacters.count) / Double(maxSteps))))
        var out: [String] = []
        var last = currentTrimmed
        var end = 0
        while end < suffixCharacters.count {
            end = min(suffixCharacters.count, end + chunkSize)
            let candidate = (currentTrimmed + String(suffixCharacters.prefix(end)))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if candidate != last {
                out.append(candidate)
                last = candidate
            }
        }
        if !out.isEmpty, out[out.count - 1] != targetTrimmed {
            out[out.count - 1] = targetTrimmed
        }
        return out
    }
}
