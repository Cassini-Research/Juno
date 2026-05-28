// JunoRecurrenceCopy.swift
//
// Render a JunoRecurrenceRule as plain-English copy for HUD chips and
// the action toast. Apple's RRULE-shaped types don't ship a localized
// describer, so we own the prose. The summary is one short sentence
// that always communicates frequency + bound + time-of-day, in the
// same order the user spoke them.
//
// Examples:
//
//   DAILY, count=10                        → "Daily for 10 days"
//   DAILY (open-ended)                     → "Every day"
//   WEEKLY, by_day=[MO,TU,WE,TH,FR]        → "Every weekday"
//   WEEKLY, by_day=[MO,WE,FR]              → "Mon, Wed, Fri"
//   WEEKLY, by_day=[SA,SU]                 → "Every weekend"
//   MONTHLY, by_month_day=[1]              → "1st of each month"
//   MONTHLY, by_month_day=[15]             → "15th of each month"
//   YEARLY                                 → "Every year"
//
// Summaries get a " at HH:MM" tail when first_occurrence_iso has a
// parseable, non-zero time-of-day component.

import Foundation

enum JunoRecurrenceCopy {

    /// Returns a one-line plain-English summary, or nil if the rule
    /// is malformed.
    static func summary(for rule: JunoRecurrenceRule) -> String? {
        guard let head = headlineForRule(rule) else { return nil }
        let bound = boundDescription(for: rule)
        let timeOfDay = formatTimeOfDay(rule.firstOccurrenceIso)
        var parts: [String] = [head]
        if let bound, !bound.isEmpty { parts.append(bound) }
        if let timeOfDay { parts.append("at \(timeOfDay)") }
        return parts.joined(separator: " ")
    }

    // MARK: - Headline

    private static func headlineForRule(_ rule: JunoRecurrenceRule) -> String? {
        switch rule.freq.uppercased() {
        case "DAILY":
            if rule.interval > 1 { return "Every \(rule.interval) days" }
            return "Every day"
        case "WEEKLY":
            return weeklyHeadline(rule)
        case "MONTHLY":
            return monthlyHeadline(rule)
        case "YEARLY":
            if rule.interval > 1 { return "Every \(rule.interval) years" }
            return "Every year"
        default:
            return nil
        }
    }

    private static func weeklyHeadline(_ rule: JunoRecurrenceRule) -> String {
        let codes = Set(rule.byDay.map { $0.uppercased() })
        if !codes.isEmpty {
            let weekdays: Set<String> = ["MO", "TU", "WE", "TH", "FR"]
            let weekend: Set<String> = ["SA", "SU"]
            if codes == weekdays {
                return rule.interval == 1 ? "Every weekday" : "Every \(rule.interval) weeks on weekdays"
            }
            if codes == weekend {
                return rule.interval == 1 ? "Every weekend" : "Every \(rule.interval) weeks on weekends"
            }
            let ordered = canonicalOrder.filter { codes.contains($0) }
            let names = ordered.compactMap { fullNameForCode($0) }
            let joined = sentenceJoin(names.map { String($0.prefix(3)) })
            if rule.interval == 1 { return joined }
            return "Every \(rule.interval) weeks on \(joined)"
        }
        if rule.interval == 1 { return "Every week" }
        return "Every \(rule.interval) weeks"
    }

    private static func monthlyHeadline(_ rule: JunoRecurrenceRule) -> String {
        if !rule.byMonthDay.isEmpty {
            let days = rule.byMonthDay.sorted().map { ordinal($0) }
            let joined = sentenceJoin(days)
            if rule.interval == 1 { return "\(joined) of each month" }
            return "Every \(rule.interval) months on the \(joined)"
        }
        if rule.interval == 1 { return "Every month" }
        return "Every \(rule.interval) months"
    }

    // MARK: - Bound (count / until)

    private static func boundDescription(for rule: JunoRecurrenceRule) -> String? {
        if let count = rule.count, count > 0 {
            switch rule.freq.uppercased() {
            case "DAILY":   return "for \(count) days"
            case "WEEKLY":  return "for \(count) weeks"
            case "MONTHLY": return "for \(count) months"
            case "YEARLY":  return "for \(count) years"
            default:        return "for \(count) times"
            }
        }
        if let untilIso = rule.untilIso, !untilIso.isEmpty,
           let until = parseISO(untilIso) {
            return "until " + formatShortDate(until)
        }
        return nil
    }

    // MARK: - Helpers

    private static func formatTimeOfDay(_ iso: String) -> String? {
        guard !iso.isEmpty, let date = parseISO(iso) else { return nil }
        let comps = Calendar.current.dateComponents([.hour, .minute, .second], from: date)
        if (comps.hour ?? 0) == 0 && (comps.minute ?? 0) == 0 && (comps.second ?? 0) == 0 {
            return nil
        }
        let f = DateFormatter()
        f.locale = .current
        f.setLocalizedDateFormatFromTemplate("h:mm a")
        return f.string(from: date)
    }

    private static func formatShortDate(_ date: Date) -> String {
        let f = DateFormatter()
        f.locale = .current
        f.setLocalizedDateFormatFromTemplate("MMM d")
        return f.string(from: date)
    }

    private static func parseISO(_ iso: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: iso) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: iso)
    }

    private static let canonicalOrder = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

    private static func fullNameForCode(_ code: String) -> String? {
        switch code {
        case "MO": return "Monday"
        case "TU": return "Tuesday"
        case "WE": return "Wednesday"
        case "TH": return "Thursday"
        case "FR": return "Friday"
        case "SA": return "Saturday"
        case "SU": return "Sunday"
        default:   return nil
        }
    }

    private static func ordinal(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .ordinal
        f.locale = .current
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func sentenceJoin(_ parts: [String]) -> String {
        switch parts.count {
        case 0: return ""
        case 1: return parts[0]
        case 2: return "\(parts[0]) and \(parts[1])"
        default:
            let head = parts.dropLast().joined(separator: ", ")
            return "\(head), and \(parts.last!)"
        }
    }
}
