// JunoActionDTOs.swift
//
// Wire models for the Juno Actions feature (notes / reminders).
// Mirrors `juno_core_v3/actions/contracts.py`. The macOS shell
// receives `JunoActionRequest` items inside the broker dictation
// response and posts back `JunoActionResult` items to
// `POST /api/broker/history/{utterance_id}/actions` once it has
// dispatched each action to its system sink (EKEventStore for
// reminders, AppleScript for notes).
//
// **Hard rule:** if the parser returns no actions, none of these
// types are involved and dictation behaves exactly as today.

import Foundation

enum JunoActionKind: String, Codable, Hashable, CaseIterable {
    case note
    case reminder
    case alarm
}

enum JunoActionStatus: String, Codable, Hashable {
    case ok
    case permissionDenied = "permission_denied"
    case sinkError = "sink_error"
    case timeParseFailed = "time_parse_failed"
    /// Action was parsed by the broker but never dispatched by this shell
    /// (toggle off, no permissions, app wasn't open at the time, etc.).
    /// Distinct from ``.ok`` so the UI doesn't lie about success.
    case pending
    /// Voice Actions toggle is OFF in Settings. Surfaces a toast asking
    /// the user to enable it.
    case blockedToggleOff = "blocked_toggle_off"
    /// At least one required permission is missing. Surfaces a toast with
    /// a one-tap deep link to the Actions page.
    case blockedNoPermission = "blocked_no_permission"
}

/// Parsed time clause attached to a reminder action. ``iso`` is an
/// ISO-8601 string with timezone offset.
///
/// The optional fields below mirror ``ParsedTime`` on the Python side and
/// drive the HUD chip's inferred-time badge:
///
/// * ``inferred`` is ``true`` when the resolver filled in missing data
///   (e.g. defaulted to 9 AM, rolled "5pm" forward to tomorrow).
/// * ``inferenceNote`` is a short human-readable explanation suitable for
///   the chip's secondary line ("rolled to tomorrow at 5pm").
/// * ``needsConfirmation`` flags genuinely ambiguous cases (e.g. a past
///   explicit date) the user should eyeball before the reminder fires.
struct JunoParsedTime: Codable, Hashable {
    let iso: String
    let confidence: Double
    let source: String  // "dateparser" | "llm" | "user_edit" | "default"
    let inferred: Bool
    let inferenceNote: String?
    let needsConfirmation: Bool

    init(
        iso: String,
        confidence: Double,
        source: String,
        inferred: Bool = false,
        inferenceNote: String? = nil,
        needsConfirmation: Bool = false
    ) {
        self.iso = iso
        self.confidence = confidence
        self.source = source
        self.inferred = inferred
        self.inferenceNote = inferenceNote
        self.needsConfirmation = needsConfirmation
    }

    private enum CodingKeys: String, CodingKey {
        case iso
        case confidence
        case source
        case inferred
        case inferenceNote = "inference_note"
        case needsConfirmation = "needs_confirmation"
    }

    static func fromDict(_ dict: [String: Any], defaultSource: String = "dateparser") -> JunoParsedTime? {
        guard let iso = dict["iso"] as? String else { return nil }
        return JunoParsedTime(
            iso: iso,
            confidence: (dict["confidence"] as? Double) ?? 1.0,
            source: (dict["source"] as? String) ?? defaultSource,
            inferred: (dict["inferred"] as? Bool) ?? false,
            inferenceNote: dict["inference_note"] as? String,
            needsConfirmation: (dict["needs_confirmation"] as? Bool) ?? false
        )
    }
}

// MARK: - v3 Schedule (Phase 1)
//
// JunoSchedule mirrors juno_core_v3/actions/contracts.py::Schedule —
// a discriminated union over instant / series / vague. Reminders and
// alarms may carry one; notes never do. Existing broker dicts without
// a ``schedule`` key keep working — JunoActionRequest.fromBrokerDict
// just leaves ``schedule`` nil and the executor falls back to ``when``.

/// Operation an Action performs against its sink. Phase 1 only ships
/// ``CREATE``; the executor refuses non-create operations until Phase 2.
enum JunoActionOperation: String, Codable, Hashable {
    case create
    case update
    case complete
    case snooze
    case delete
    case query
    case appendTo = "append_to"
    case removeFrom = "remove_from"
}

/// ICS-shaped recurrence rule. Maps directly to ``EKRecurrenceRule``
/// via ``JunoReminderSink.ekRecurrenceRule(for:)``.
struct JunoRecurrenceRule: Codable, Hashable {
    let freq: String  // "DAILY" | "WEEKLY" | "MONTHLY" | "YEARLY"
    let interval: Int
    let byDay: [String]
    let byMonthDay: [Int]
    let byMonth: [Int]
    let count: Int?
    let untilIso: String?
    let firstOccurrenceIso: String
    let tz: String?
    let excludeDatesIso: [String]

    init(
        freq: String,
        interval: Int = 1,
        byDay: [String] = [],
        byMonthDay: [Int] = [],
        byMonth: [Int] = [],
        count: Int? = nil,
        untilIso: String? = nil,
        firstOccurrenceIso: String,
        tz: String? = nil,
        excludeDatesIso: [String] = []
    ) {
        self.freq = freq
        self.interval = max(1, interval)
        self.byDay = byDay
        self.byMonthDay = byMonthDay
        self.byMonth = byMonth
        self.count = count
        self.untilIso = untilIso
        self.firstOccurrenceIso = firstOccurrenceIso
        self.tz = tz
        self.excludeDatesIso = excludeDatesIso
    }

    private enum CodingKeys: String, CodingKey {
        case freq
        case interval
        case byDay = "by_day"
        case byMonthDay = "by_month_day"
        case byMonth = "by_month"
        case count
        case untilIso = "until_iso"
        case firstOccurrenceIso = "first_occurrence_iso"
        case tz
        case excludeDatesIso = "exclude_dates_iso"
    }

    static func fromDict(_ dict: [String: Any]) -> JunoRecurrenceRule? {
        guard let freq = dict["freq"] as? String else { return nil }
        return JunoRecurrenceRule(
            freq: freq,
            interval: (dict["interval"] as? Int) ?? 1,
            byDay: (dict["by_day"] as? [String]) ?? [],
            byMonthDay: (dict["by_month_day"] as? [Int]) ?? [],
            byMonth: (dict["by_month"] as? [Int]) ?? [],
            count: dict["count"] as? Int,
            untilIso: dict["until_iso"] as? String,
            firstOccurrenceIso: (dict["first_occurrence_iso"] as? String) ?? "",
            tz: dict["tz"] as? String,
            excludeDatesIso: (dict["exclude_dates_iso"] as? [String]) ?? []
        )
    }
}

/// Vague schedule the user pronounced ("later", "tonight", "this
/// weekend"). Phase 3 will surface the amber HUD chip and tap-to-edit;
/// Phase 1 just plumbs the type through.
struct JunoVagueSchedule: Codable, Hashable {
    let bucket: String
    let defaultIso: String
    let tz: String?
    let needsConfirmation: Bool

    private enum CodingKeys: String, CodingKey {
        case bucket
        case defaultIso = "default_iso"
        case tz
        case needsConfirmation = "needs_confirmation"
    }

    static func fromDict(_ dict: [String: Any]) -> JunoVagueSchedule? {
        guard
            let bucket = dict["bucket"] as? String,
            let defaultIso = dict["default_iso"] as? String,
            !defaultIso.isEmpty
        else { return nil }
        return JunoVagueSchedule(
            bucket: bucket,
            defaultIso: defaultIso,
            tz: dict["tz"] as? String,
            needsConfirmation: (dict["needs_confirmation"] as? Bool) ?? true
        )
    }
}

/// Discriminated union over instant / series / vague.
struct JunoSchedule: Codable, Hashable {
    let kind: String
    let instant: JunoParsedTime?
    let series: JunoRecurrenceRule?
    let vague: JunoVagueSchedule?

    static func fromDict(_ dict: [String: Any]) -> JunoSchedule? {
        guard let kind = dict["kind"] as? String else { return nil }
        let instant = (dict["instant"] as? [String: Any]).flatMap {
            JunoParsedTime.fromDict($0, defaultSource: "llm")
        }
        let series = (dict["series"] as? [String: Any]).flatMap { JunoRecurrenceRule.fromDict($0) }
        let vague = (dict["vague"] as? [String: Any]).flatMap { JunoVagueSchedule.fromDict($0) }
        return JunoSchedule(kind: kind, instant: instant, series: series, vague: vague)
    }

    /// Best-effort single ISO instant. Used by HUD chips that only know
    /// how to render a single point in time.
    var primaryIso: String? {
        if let i = instant { return i.iso }
        if let s = series, !s.firstOccurrenceIso.isEmpty { return s.firstOccurrenceIso }
        if let v = vague { return v.defaultIso }
        return nil
    }
}

struct JunoActionContainer: Codable, Hashable {
    let listName: String?
    let folderName: String?

    private enum CodingKeys: String, CodingKey {
        case listName = "list_name"
        case folderName = "folder_name"
    }

    static func fromDict(_ dict: [String: Any]) -> JunoActionContainer? {
        let list = (dict["list_name"] as? String) ?? (dict["listName"] as? String)
        let folder = (dict["folder_name"] as? String) ?? (dict["folderName"] as? String)
        let cleanList = list?.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanFolder = folder?.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleanList?.isEmpty != false && cleanFolder?.isEmpty != false { return nil }
        return JunoActionContainer(listName: cleanList, folderName: cleanFolder)
    }
}

/// One action parsed by the Python broker. Treat as immutable input —
/// the executor produces ``JunoActionResult`` values to write back.
struct JunoActionRequest: Codable, Hashable, Identifiable {
    let junoId: String?
    let sinkId: String?
    let linkId: String?
    let linksTo: String?
    let kind: JunoActionKind
    let body: String
    let rawSpan: String
    let when: JunoParsedTime?
    /// v3 schedule (instant/series/vague). When present, supersedes
    /// ``when`` for series/vague. ``when`` is still populated for
    /// instant schedules so legacy code paths keep working.
    let schedule: JunoSchedule?
    let container: JunoActionContainer?
    /// v3 operation. v2 envelopes always set this to ``.create``.
    let operation: JunoActionOperation
    let snoozeOffsetSeconds: Int?
    let relativeOffsetSeconds: Int?

    init(
        junoId: String? = nil,
        sinkId: String? = nil,
        linkId: String? = nil,
        linksTo: String? = nil,
        kind: JunoActionKind,
        body: String,
        rawSpan: String,
        when: JunoParsedTime? = nil,
        schedule: JunoSchedule? = nil,
        container: JunoActionContainer? = nil,
        operation: JunoActionOperation = .create,
        snoozeOffsetSeconds: Int? = nil,
        relativeOffsetSeconds: Int? = nil
    ) {
        self.junoId = junoId
        self.sinkId = sinkId
        self.linkId = linkId
        self.linksTo = linksTo
        self.kind = kind
        self.body = body
        self.rawSpan = rawSpan
        self.when = when
        self.schedule = schedule
        self.container = container
        self.operation = operation
        self.snoozeOffsetSeconds = snoozeOffsetSeconds
        self.relativeOffsetSeconds = relativeOffsetSeconds
    }

    /// Stable ID for SwiftUI list/animation purposes within a single
    /// utterance; not persisted, not synced with the backend.
    var id: String {
        junoId ?? "\(kind.rawValue):\(rawSpan)"
    }

    /// Decode from the loose `[String: Any]` shape returned by
    /// `JSONSerialization` — the broker response handler bypasses the
    /// `JSONDecoder` path because it interleaves status checks and
    /// utterance bookkeeping. Returns nil rather than throwing so a
    /// malformed entry can never crash the dictation flow.
    static func fromBrokerDict(_ dict: [String: Any]) -> JunoActionRequest? {
        guard
            let kindRaw = dict["kind"] as? String,
            let kind = JunoActionKind(rawValue: kindRaw),
            let body = dict["body"] as? String
        else {
            return nil
        }
        let rawSpan = (dict["raw_span"] as? String) ?? body
        let when = (dict["when"] as? [String: Any]).flatMap {
            JunoParsedTime.fromDict($0)
        }
        // v3 fields. Both default-absent so v2 broker dicts decode unchanged.
        let schedule = (dict["schedule"] as? [String: Any]).flatMap { JunoSchedule.fromDict($0) }
        let container = (dict["container"] as? [String: Any]).flatMap { JunoActionContainer.fromDict($0) }
        let operationRaw = (dict["operation"] as? String) ?? "create"
        let operation = JunoActionOperation(rawValue: operationRaw) ?? .create
        let junoId = (dict["juno_id"] as? String) ?? (dict["junoId"] as? String)
        let sinkId = (dict["sink_id"] as? String) ?? (dict["sinkId"] as? String)
        let linkId = (dict["link_id"] as? String) ?? (dict["linkId"] as? String)
        let linksTo = (dict["links_to"] as? String) ?? (dict["linksTo"] as? String)
        let snoozeOffset = (dict["snooze_offset_seconds"] as? Int) ?? (dict["snoozeOffsetSeconds"] as? Int)
        let relativeOffset = (dict["relative_offset_seconds"] as? Int) ?? (dict["relativeOffsetSeconds"] as? Int)
        return JunoActionRequest(
            junoId: junoId,
            sinkId: sinkId,
            linkId: linkId,
            linksTo: linksTo,
            kind: kind,
            body: body,
            rawSpan: rawSpan,
            when: when,
            schedule: schedule,
            container: container,
            operation: operation,
            snoozeOffsetSeconds: snoozeOffset,
            relativeOffsetSeconds: relativeOffset
        )
    }

}

/// The outcome of executing one ``JunoActionRequest``. Posted back
/// to the broker history row as part of the ``actions`` JSON column.
///
/// **Decode tolerance:** the same column also stores the *parsed* action
/// shape (``kind / body / raw_span / when``) before the macOS shell has
/// dispatched it. The custom ``init(from:)`` accepts both shapes —
/// missing ``status`` defaults to ``.pending`` and ``body_preview`` falls
/// back to ``body``. Without this, every history row that has a parsed-
/// but-not-yet-dispatched action throws ``valueNotFound`` during decode
/// and the entire History tab fails to load.
struct JunoActionResult: Codable, Hashable {
    let junoId: String
    let operation: JunoActionOperation?
    let kind: JunoActionKind
    let status: JunoActionStatus
    let bodyPreview: String
    let body: String?
    let sinkId: String?
    let sinkUrl: String?
    let whenIso: String?
    let error: String?
    let extras: [String: String]?
    /// False when History decoded a parsed intent payload that never received
    /// shell execution results. UI uses this to avoid showing "Saving..."
    /// forever for stale action rows.
    let hasExplicitStatus: Bool

    init(
        junoId: String,
        operation: JunoActionOperation? = nil,
        kind: JunoActionKind,
        status: JunoActionStatus,
        bodyPreview: String,
        body: String? = nil,
        sinkId: String? = nil,
        sinkUrl: String? = nil,
        whenIso: String? = nil,
        error: String? = nil,
        extras: [String: String]? = nil,
        hasExplicitStatus: Bool = true
    ) {
        self.junoId = junoId
        self.operation = operation
        self.kind = kind
        self.status = status
        self.bodyPreview = bodyPreview
        self.body = body
        self.sinkId = sinkId
        self.sinkUrl = sinkUrl
        self.whenIso = whenIso
        self.error = error
        self.extras = extras
        self.hasExplicitStatus = hasExplicitStatus
    }

    private enum CodingKeys: String, CodingKey {
        case junoId
        case junoIdSnake = "juno_id"
        case operation
        case kind
        case status
        case bodyPreview
        case body
        case sinkId
        case sinkUrl
        case whenIso
        case when
        case error
        case extras
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let decodedJunoId = (try? c.decode(String.self, forKey: .junoId))
            ?? (try? c.decode(String.self, forKey: .junoIdSnake))
        let trimmedJunoId = decodedJunoId?.trimmingCharacters(in: .whitespacesAndNewlines)
        junoId = trimmedJunoId?.isEmpty == false ? trimmedJunoId! : UUID().uuidString
        operation = try? c.decode(JunoActionOperation.self, forKey: .operation)
        kind = try c.decode(JunoActionKind.self, forKey: .kind)
        let decodedStatus = try? c.decode(JunoActionStatus.self, forKey: .status)
        hasExplicitStatus = decodedStatus != nil
        status = decodedStatus ?? .pending
        let decodedBody = try? c.decode(String.self, forKey: .body)
        if let preview = try? c.decode(String.self, forKey: .bodyPreview) {
            bodyPreview = preview
        } else if let decodedBody {
            bodyPreview = decodedBody
        } else {
            bodyPreview = ""
        }
        body = decodedBody
        sinkId = try? c.decode(String.self, forKey: .sinkId)
        sinkUrl = try? c.decode(String.self, forKey: .sinkUrl)
        if let iso = try? c.decode(String.self, forKey: .whenIso) {
            whenIso = iso
        } else if let parsed = try? c.decode(JunoParsedTime.self, forKey: .when) {
            whenIso = parsed.iso
        } else {
            whenIso = nil
        }
        error = try? c.decode(String.self, forKey: .error)
        extras = try? c.decode([String: String].self, forKey: .extras)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(junoId, forKey: .junoId)
        try c.encodeIfPresent(operation, forKey: .operation)
        try c.encode(kind, forKey: .kind)
        try c.encode(status, forKey: .status)
        try c.encode(bodyPreview, forKey: .bodyPreview)
        try c.encodeIfPresent(body, forKey: .body)
        try c.encodeIfPresent(sinkId, forKey: .sinkId)
        try c.encodeIfPresent(sinkUrl, forKey: .sinkUrl)
        try c.encodeIfPresent(whenIso, forKey: .whenIso)
        try c.encodeIfPresent(error, forKey: .error)
        try c.encodeIfPresent(extras, forKey: .extras)
    }

    func withDisplayStatus(_ status: JunoActionStatus, error: String? = nil) -> JunoActionResult {
        JunoActionResult(
            junoId: junoId,
            operation: operation,
            kind: kind,
            status: status,
            bodyPreview: bodyPreview,
            body: body,
            sinkId: sinkId,
            sinkUrl: sinkUrl,
            whenIso: whenIso,
            error: error ?? self.error,
            extras: extras,
            hasExplicitStatus: true
        )
    }
}

// MARK: - Batch summary
//
// Shared formatter for "what does a batch of action results read like?".
// Three surfaces consume it: the transient HUD chip (one line), the
// auto-dismissing toast (rich card with rows), and the History detail
// pane's "What Juno did" summary. Without one place that owns the
// language, the surfaces drift — and we end up with the bug the user
// flagged: the HUD says "Permission needed" while the History row says
// "4/5 saved". Single source of truth keeps copy honest.

enum JunoActionBatchTone {
    /// All actions saved. Green.
    case allSaved
    /// Some saved, others didn't. Accent (orange/blue) — neither
    /// triumphant nor a failure.
    case partial
    /// Nothing saved AND every failure is a permission/toggle block.
    /// Orange.
    case blocked
    /// Nothing saved AND at least one failure is a sink/time error.
    /// Red.
    case failed
    /// Every action is still pending (action posts in flight, broker
    /// restart, etc.). Accent.
    case allPending
}

struct JunoActionBatchSummary: Equatable {
    /// Short headline ("Reminder saved", "Saved 4 of 5 actions"). Suitable
    /// for the toast headline and the History summary chip.
    let headline: String
    /// Compact one-line summary suitable for the transient HUD chip.
    /// Includes the friction tail ("— 1 needs permission") so a single
    /// line communicates both success count and remaining work.
    let oneLine: String
    /// Optional secondary line for surfaces that have room (toast,
    /// History detail). May be ``nil`` when the headline already says
    /// everything.
    let detail: String?
    let tone: JunoActionBatchTone
    /// Set when the batch is recoverable by clicking through (e.g.
    /// permission needed). The toast surfaces a CTA button when true.
    let needsResolveCTA: Bool
}

extension Array where Element == JunoActionResult {
    /// Group these results by kind preserving the order each kind first
    /// appears. Used by the formatter so "Saved 3 reminders, 2 notes"
    /// reads in the order the user said them.
    fileprivate var groupedByKind: [(kind: JunoActionKind, count: Int)] {
        var seen: [JunoActionKind] = []
        var counts: [JunoActionKind: Int] = [:]
        for r in self {
            if counts[r.kind] == nil { seen.append(r.kind) }
            counts[r.kind, default: 0] += 1
        }
        return seen.map { ($0, counts[$0] ?? 0) }
    }
}

enum JunoActionBatchFormatter {

    /// Singular/plural kind name. ``count`` is the number being described
    /// (1 → "reminder", 2 → "reminders").
    static func kindWord(_ kind: JunoActionKind, count: Int) -> String {
        let name = count == 1 ? kind.descriptor.displayName : kind.descriptor.pluralName
        return name.lowercased()
    }

    /// Comma-and-style joining: ["a"] → "a"; ["a","b"] → "a and b";
    /// ["a","b","c"] → "a, b, and c".
    static func sentenceJoin(_ parts: [String]) -> String {
        switch parts.count {
        case 0: return ""
        case 1: return parts[0]
        case 2: return "\(parts[0]) and \(parts[1])"
        default:
            let head = parts.dropLast().joined(separator: ", ")
            return "\(head), and \(parts.last!)"
        }
    }

    /// Render the saved-portion phrase. Returns "" when none saved.
    /// Examples: "1 reminder", "2 notes", "1 reminder and 2 notes",
    /// "1 reminder, 2 notes, and 1 alarm".
    static func renderSaved(_ saved: [JunoActionResult]) -> String {
        let groups = saved.groupedByKind
        let parts = groups.map { "\($0.count) \(kindWord($0.kind, count: $0.count))" }
        return sentenceJoin(parts)
    }

    /// Friction tail. Empty if nothing needs explanation.
    /// Examples: "1 reminder needs permission",
    /// "1 alarm has no time", "2 actions couldn't save", or
    /// "1 note is still syncing".
    static func renderFriction(_ unsaved: [JunoActionResult]) -> String {
        if unsaved.isEmpty { return "" }
        // Categorize outcomes into buckets the user can interpret or act on.
        let permissionMisses = unsaved.filter {
            $0.status == .permissionDenied || $0.status == .blockedNoPermission || $0.status == .blockedToggleOff
        }
        let timeMisses = unsaved.filter { $0.status == .timeParseFailed }
        let sinkMisses = unsaved.filter { $0.status == .sinkError }
        let pending = unsaved.filter { $0.status == .pending }

        var parts: [String] = []
        if !permissionMisses.isEmpty {
            let phrase = renderSaved(permissionMisses)
            parts.append("\(phrase) need\(permissionMisses.count == 1 ? "s" : "") permission")
        }
        if !timeMisses.isEmpty {
            let phrase = renderSaved(timeMisses)
            parts.append("\(phrase) ha\(timeMisses.count == 1 ? "s" : "ve") no time")
        }
        if !sinkMisses.isEmpty {
            let phrase = renderSaved(sinkMisses)
            parts.append("\(phrase) couldn\u{2019}t save")
        }
        if !pending.isEmpty {
            let phrase = renderSaved(pending)
            parts.append("\(phrase) \(pending.count == 1 ? "is" : "are") still syncing")
        }
        return sentenceJoin(parts)
    }

    static func operationLine(_ result: JunoActionResult) -> String? {
        guard result.status == .ok, let operation = result.operation else { return nil }
        let body = result.bodyPreview.isEmpty ? result.kind.descriptor.displayName.lowercased() : result.bodyPreview
        switch operation {
        case .create:
            return nil
        case .update:
            return "Updated: \(body)"
        case .complete:
            return "Marked done: \(body)"
        case .snooze:
            return "Snoozed: \(body)"
        case .delete:
            return "Cancelled: \(body)"
        case .query:
            return result.bodyPreview
        case .appendTo, .removeFrom:
            return nil
        }
    }

    static func summarize(_ results: [JunoActionResult]) -> JunoActionBatchSummary {
        if results.isEmpty {
            return JunoActionBatchSummary(
                headline: "Voice action",
                oneLine: "Voice action",
                detail: nil,
                tone: .allPending,
                needsResolveCTA: false
            )
        }

        let saved = results.filter { $0.status == .ok }
        let pending = results.filter { $0.status == .pending }
        let failed = results.filter { r in
            switch r.status {
            case .permissionDenied, .blockedNoPermission, .blockedToggleOff,
                 .sinkError, .timeParseFailed: return true
            case .ok, .pending: return false
            }
        }
        let resolvable = results.contains { r in
            r.status == .permissionDenied || r.status == .blockedNoPermission || r.status == .blockedToggleOff
        }

        // ---- All pending ---------------------------------------------------
        if pending.count == results.count {
            return JunoActionBatchSummary(
                headline: "Saving\u{2026}",
                oneLine: "Saving \(renderSaved(pending))\u{2026}",
                detail: "Syncing results back to History.",
                tone: .allPending,
                needsResolveCTA: false
            )
        }

        // ---- All saved -----------------------------------------------------
        if saved.count == results.count {
            if results.count == 1, let first = results.first {
                if let line = operationLine(first) {
                    return JunoActionBatchSummary(
                        headline: line,
                        oneLine: line,
                        detail: nil,
                        tone: .allSaved,
                        needsResolveCTA: false
                    )
                }
                return JunoActionBatchSummary(
                    headline: "\(first.kind.descriptor.displayName) saved",
                    oneLine: "\(first.kind.descriptor.displayName) saved",
                    detail: nil,
                    tone: .allSaved,
                    needsResolveCTA: false
                )
            }
            let operationLines = results.compactMap(operationLine)
            if !operationLines.isEmpty && operationLines.count == results.count {
                let line = sentenceJoin(operationLines)
                return JunoActionBatchSummary(
                    headline: line,
                    oneLine: line,
                    detail: nil,
                    tone: .allSaved,
                    needsResolveCTA: false
                )
            }
            let phrase = renderSaved(results)
            return JunoActionBatchSummary(
                headline: "Saved \(phrase)",
                oneLine: "Saved \(phrase)",
                detail: nil,
                tone: .allSaved,
                needsResolveCTA: false
            )
        }

        // ---- All failed ----------------------------------------------------
        if saved.isEmpty {
            let friction = renderFriction(failed + pending)
            // ``failed + pending`` so mixed blocked/syncing states are
            // explicit instead of collapsing pending actions into silence.
            let allBlocked = failed.allSatisfy {
                $0.status == .permissionDenied || $0.status == .blockedNoPermission || $0.status == .blockedToggleOff
            }
            let tone: JunoActionBatchTone = allBlocked ? .blocked : .failed
            let headline: String
            if results.count == 1, let first = results.first {
                headline = first.status == .timeParseFailed
                    ? "\(first.kind.descriptor.displayName) needs a time"
                    : (allBlocked ? "Permission needed" : "Couldn\u{2019}t save \(first.kind.descriptor.displayName.lowercased())")
            } else {
                headline = allBlocked ? "Permission needed" : "Couldn\u{2019}t save"
            }
            return JunoActionBatchSummary(
                headline: headline,
                oneLine: friction.isEmpty ? headline : friction.prefix(1).uppercased() + friction.dropFirst(),
                detail: friction.isEmpty ? nil : friction,
                tone: tone,
                needsResolveCTA: resolvable
            )
        }

        // ---- Partial success ------------------------------------------------
        // Saved some, failed some. Lead with the count, trail with the
        // remaining work so users know what to fix.
        let savedPhrase = renderSaved(saved)
        let friction = renderFriction(failed + pending)
        let headline = "Saved \(saved.count) of \(results.count)"
        let oneLineTail = friction.isEmpty ? "" : " \u{2014} \(friction)"
        return JunoActionBatchSummary(
            headline: headline,
            oneLine: "\(headline)\(oneLineTail)",
            detail: friction.isEmpty
                ? "Saved \(savedPhrase)."
                : "Saved \(savedPhrase). \(friction.prefix(1).uppercased() + friction.dropFirst()).",
            tone: .partial,
            needsResolveCTA: resolvable
        )
    }
}

extension JunoActionResult {
    /// JSON-ready dictionary for the
    /// `POST /api/broker/history/{uid}/actions` endpoint. The broker
    /// stores this verbatim in the ``actions_json`` column.
    func toWireDict() -> [String: Any] {
        var d: [String: Any] = [
            "juno_id": junoId,
            "operation": operation?.rawValue ?? NSNull(),
            "kind": kind.rawValue,
            "status": status.rawValue,
            "body_preview": bodyPreview,
        ]
        d["body"] = body ?? bodyPreview
        d["sink_id"] = sinkId ?? NSNull()
        d["sink_url"] = sinkUrl ?? NSNull()
        d["when_iso"] = whenIso ?? NSNull()
        d["error"] = error ?? NSNull()
        d["extras"] = extras ?? [:] as [String: String]
        return d
    }
}
