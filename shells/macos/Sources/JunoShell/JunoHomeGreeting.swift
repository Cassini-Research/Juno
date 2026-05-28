import Foundation

/// Time-of-day + locale-aware headline for the home hero.
/// Sublines rotate on a short clock seed. Used as **offline fallback** when
/// the broker writer greeting is unavailable.
enum JunoHomeGreeting {

    private static let launchSeed: Int = {
        let components = Calendar.current.dateComponents([.day, .hour], from: Date())
        // Changes every 3 hours so frequent users see variety without it being jarring
        return (components.day ?? 0) * 8 + (components.hour ?? 0) / 3
    }()

    static func systemFirstName() -> String? {
        let full = NSFullUserName()
        let trimmed = full.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed.split(separator: " ").first.map(String.init)
    }

    static func displayNameForGreeting() -> String {
        if let p = JunoUserDefaults.preferredDisplayName, !p.isEmpty { return p }
        if let s = systemFirstName(), !s.isEmpty { return s }
        return ""
    }

    /// Same 3-hour bucket as ``launchSeed`` / subline rotation — used to decide when
    /// a broker-backed home greeting may be stale (aligned with offline hero variety).
    static func stalenessBucket(now: Date = Date(), calendar: Calendar = .current) -> Int {
        let components = calendar.dateComponents([.day, .hour], from: now)
        return (components.day ?? 0) * 8 + (components.hour ?? 0) / 3
    }

    /// Identity for client-side refresh: time bucket plus display name used in prompts.
    static func greetingStalenessIdentity(now: Date = Date(), calendar: Calendar = .current) -> String {
        "\(stalenessBucket(now: now, calendar: calendar))|\(displayNameForGreeting())"
    }

    /// Primary headline and a rotated contextual subline.
    static func heroLines(now: Date = Date(), calendar: Calendar = .current) -> (headline: String, subline: String) {
        let name = displayNameForGreeting()
        let hour = calendar.component(.hour, from: now)
        let weekday = calendar.component(.weekday, from: now)
        let isWeekend = weekday == 1 || weekday == 7

        let headline = englishHeadline(name: name, hour: hour)
        let subline = englishSubline(hour: hour, isWeekend: isWeekend, seed: launchSeed)
        return (headline, subline)
    }

    // MARK: Private

    private static func englishHeadline(name: String, hour: Int) -> String {
        switch hour {
        case 5..<12:
            return name.isEmpty ? "Good morning" : "Good morning, \(name)"
        case 12..<17:
            return name.isEmpty ? "Good afternoon" : "Good afternoon, \(name)"
        case 17..<22:
            return name.isEmpty ? "Good evening" : "Good evening, \(name)"
        default:
            return name.isEmpty ? "Hey there" : "Hey, \(name)"
        }
    }

    private static func englishSubline(hour: Int, isWeekend: Bool, seed: Int) -> String {
        // Pool varies by time band — pick by seed so it's stable within the session.
        let pool: [String]

        switch hour {
        case 5..<9:
            pool = [
                "Say it out loud; it lands where you're already typing.",
                "Coffee first, then whatever sentence you've been avoiding.",
                "Rough notes are fine. You can clean them up later.",
                "If it's still in your head, it's not in the doc yet.",
            ]
        case 9..<12:
            pool = [
                "Dictate into the window that's already open.",
                "Same shortcut as always: hold, speak, release.",
                "Good for a messy first pass you wouldn't want to type.",
                "When your hands are full, your voice still works.",
            ]
        case 12..<14:
            pool = [
                "Short burst dictation fits the post-lunch slump.",
                "One paragraph out loud beats five minutes of tapping.",
                "Say the awkward sentence once; edit the wording after.",
                "You don't need a blank page first. Just start talking.",
            ]
        case 14..<17:
            pool = [
                "Audio and models stay on this machine unless you changed that.",
                "Useful when your wrists want a break from the trackpad.",
                "Good for a brain-dump before you rearrange it.",
                "If the idea's half-formed, talking usually unsticks it.",
            ]
        case 17..<21:
            pool = [
                "End of day: ramble into the field, then trim.",
                "Fine for a quick reply you don't want to thumb-type.",
                "Keep the lights low; Juno doesn't care.",
                "Say the thing you'd put off until tomorrow.",
            ]
        default:
            pool = [
                "Still up? Dictate a note and come back to it in the morning.",
                "Quiet house, loud thoughts. One utterance at a time.",
                "No one's judging the first draft.",
                "Type with your voice when the keyboard feels like too much.",
            ]
        }

        let weekendBonus: [String] = [
            "Saturday or Sunday: same shortcut if you need it.",
            "Low-stakes day. Good time to try a messy draft.",
        ]

        let fullPool = isWeekend ? pool + weekendBonus : pool
        return fullPool[seed % fullPool.count]
    }
}
