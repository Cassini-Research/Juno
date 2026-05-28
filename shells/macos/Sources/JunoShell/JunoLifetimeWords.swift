import Foundation

/// Tracks cumulative words pasted for milestone delights.
enum JunoLifetimeWords {
    private static let defaultsKey = "JunoShellLifetimeWordCount"
    private static let legacyKey = "JunoShellLifetimeWordCount"

    static func totalCount() -> Int {
        let ud = UserDefaults.standard
        if ud.object(forKey: defaultsKey) == nil,
           ud.object(forKey: legacyKey) != nil {
            ud.set(ud.integer(forKey: legacyKey), forKey: defaultsKey)
            ud.removeObject(forKey: legacyKey)
        }
        return ud.integer(forKey: defaultsKey)
    }

    /// Adds word count from `text`; returns whether a 100-word milestone was crossed.
    @discardableResult
    static func recordWords(from text: String) -> Bool {
        let words = text.split { $0.isWhitespace || $0.isNewline }.filter { !$0.isEmpty }.count
        guard words > 0 else { return false }
        _ = totalCount()
        let ud = UserDefaults.standard
        let prev = ud.integer(forKey: defaultsKey)
        let prevBucket = prev / 100
        let next = prev + words
        ud.set(next, forKey: defaultsKey)
        let nextBucket = next / 100
        return nextBucket > prevBucket && next >= 100
    }
}
