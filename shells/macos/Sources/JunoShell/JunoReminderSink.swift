// JunoReminderSink.swift
//
// EKEventStore wrapper for the Reminders sink.
//
// The framework split changed in macOS 14: ``requestFullAccessToReminders``
// is the modern entry point; ``requestAccess(to:)`` is the legacy fallback.
// We probe availability at runtime so we keep building against the older
// SDK without losing the modern UX.
//
// **Hard rule:** if the user has not granted permission, this sink must
// fail with ``JunoActionStatus.permissionDenied`` and never throw or
// crash. The dictation path that produced the action remains usable for
// other purposes; only the reminder portion is degraded.

import EventKit
import Foundation

/// Stateless wrapper. A long-lived store is used for saves; permission
/// prompts use a fresh store so macOS does not leave the UI reading a stale
/// EventKit cache immediately after the user grants access.
final class JunoReminderSink {

    static let shared = JunoReminderSink()

    private let store: EKEventStore
    private let grantLock = NSLock()
    private var optimisticGrantUntil: Date?
    private static let optimisticGrantTTL: TimeInterval = 30

    private init(store: EKEventStore = EKEventStore()) {
        self.store = store
    }

    // MARK: - Authorization

    enum Authorization {
        case granted
        case denied
        case notDetermined
        case restricted
    }

    /// Maps the platform-specific authorization status onto the simpler
    /// trinary the UI cares about. We explicitly do **not** distinguish
    /// "fullAccess" vs "writeOnly" yet — Phase 3 only writes reminders, so
    /// either grant is sufficient.
    var authorization: Authorization {
        let live = Self.authorizationStatus()
        switch live {
        case .granted:
            clearOptimisticGrant()
            JunoEventKitGrantCache.setGrant(.reminders, granted: true)
            return .granted
        case .notDetermined:
            if hasOptimisticGrant() {
                resetEventStore()
                return .granted
            }
            JunoEventKitGrantCache.setGrant(.reminders, granted: false)
            return .notDetermined
        case .denied, .restricted:
            clearOptimisticGrant()
            JunoEventKitGrantCache.setGrant(.reminders, granted: false)
            return live
        }
    }

    static func authorizationStatus() -> Authorization {
        let status = EKEventStore.authorizationStatus(for: .reminder)
        switch status {
        case .authorized:
            return .granted
        case .denied:
            return .denied
        case .restricted:
            return .restricted
        case .notDetermined:
            return .notDetermined
        case .fullAccess, .writeOnly:
            return .granted
        @unknown default:
            // Forward-compat: treat unknown future cases as notDetermined
            // so the nudge surfaces and the user gets a deterministic ask.
            return .notDetermined
        }
    }

    /// EventKit caches authorization and calendar state inside each store.
    /// If this store existed before the user granted access, reset it before
    /// the next save so the first post-grant reminder does not use stale
    /// not-determined state.
    func resetEventStore() {
        store.reset()
    }

    /// Triggers the system permission prompt the first time it is called.
    /// Subsequent calls short-circuit on the cached status. The completion
    /// runs on an arbitrary queue — UI callers should hop to main.
    func requestAccess(_ completion: @escaping (Authorization) -> Void) {
        let requestStore = EKEventStore()
        if #available(macOS 14.0, *) {
            requestStore.requestFullAccessToReminders { [requestStore, weak self] granted, _ in
                _ = requestStore
                if granted {
                    self?.rememberOptimisticGrant()
                    completion(.granted)
                } else {
                    completion(Self.authorizationStatus())
                }
            }
        } else {
            requestStore.requestAccess(to: .reminder) { [requestStore, weak self] granted, _ in
                _ = requestStore
                if granted {
                    self?.rememberOptimisticGrant()
                    completion(.granted)
                } else {
                    completion(Self.authorizationStatus())
                }
            }
        }
    }

    private func rememberOptimisticGrant() {
        grantLock.lock()
        optimisticGrantUntil = Date().addingTimeInterval(Self.optimisticGrantTTL)
        grantLock.unlock()
        JunoEventKitGrantCache.setGrant(.reminders, granted: true)
        resetEventStore()
    }

    private func hasOptimisticGrant() -> Bool {
        grantLock.lock()
        defer { grantLock.unlock() }
        guard let optimisticGrantUntil else { return false }
        return optimisticGrantUntil > Date()
    }

    private func clearOptimisticGrant() {
        grantLock.lock()
        optimisticGrantUntil = nil
        grantLock.unlock()
    }

    private func forgetGrantAfterAuthorizationFailure() {
        clearOptimisticGrant()
        JunoEventKitGrantCache.setGrant(.reminders, granted: false)
        resetEventStore()
    }

    private func permissionAwareFailure(_ error: Error) -> SinkError {
        if Self.isEventStoreNotAuthorized(error) {
            forgetGrantAfterAuthorizationFailure()
            return .permissionDenied
        }
        return .saveFailed(underlying: error)
    }

    private static func isEventStoreNotAuthorized(_ error: Error) -> Bool {
        let nsError = error as NSError
        return nsError.domain == EKError.errorDomain
            && nsError.code == EKError.Code.eventStoreNotAuthorized.rawValue
    }

    // MARK: - Create

    struct CreatedReminder {
        let id: String
        let url: URL?
    }

    struct ReminderPatch {
        let title: String?
        let dueDate: Date?
        let recurrence: JunoRecurrenceRule?
        let listName: String?
    }

    struct ReminderQueryFilter {
        let dateRange: ClosedRange<Date>?
        let text: String?
        let listName: String?
    }

    struct QueriedReminder {
        let id: String
        let title: String
        let dueDate: Date?
        let url: URL?
    }

    enum SinkError: Error, LocalizedError {
        case permissionDenied
        case notFound
        case saveFailed(underlying: Error)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Reminders access is not granted."
            case .notFound:
                return "Couldn't find that reminder."
            case .saveFailed(let underlying):
                return "Couldn't save reminder: \(underlying.localizedDescription)"
            }
        }
    }

    /// Create a single reminder. ``dueDate`` may be nil (Apple Reminders
    /// supports undated reminders natively, which is the right default
    /// when the user said "remind me to X" with no time clause).
    ///
    /// When ``recurrence`` is non-nil, the reminder is saved with an
    /// ``EKRecurrenceRule`` so Apple's Reminders engine handles the
    /// repeated firings. We never schedule N independent reminders
    /// for "every weekday at 9". ``dueDate`` should be the *first*
    /// occurrence; EventKit derives subsequent firings from the rule.
    func createReminder(
        title: String,
        notes: String? = nil,
        dueDate: Date? = nil,
        recurrence: JunoRecurrenceRule? = nil,
        listName: String? = nil,
        completion: @escaping (Result<CreatedReminder, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        let reminder = EKReminder(eventStore: store)
        reminder.title = title
        reminder.notes = notes
        reminder.calendar = reminderList(named: listName) ?? store.defaultCalendarForNewReminders()
        if let due = dueDate {
            reminder.dueDateComponents = Calendar.current.dateComponents(
                [.year, .month, .day, .hour, .minute],
                from: due
            )
            // EKAlarm so the user actually gets pinged at the time.
            reminder.addAlarm(EKAlarm(absoluteDate: due))
        }
        if let rule = recurrence,
           let ekRule = Self.ekRecurrenceRule(for: rule) {
            reminder.recurrenceRules = [ekRule]
        }
        do {
            try store.save(reminder, commit: true)
            // x-apple-reminderkit:// URLs are documented; the host segment
            // is `REMCDReminder/<calendarItemIdentifier>` per Apple's URL
            // scheme reference. The deep link opens the reminder in the
            // Reminders.app on macOS and iOS (when the chip is tapped on
            // a synced device).
            let id = reminder.calendarItemIdentifier
            let url = URL(string: "x-apple-reminderkit://REMCDReminder/\(id)")
            completion(.success(CreatedReminder(id: id, url: url)))
        } catch {
            completion(.failure(permissionAwareFailure(error)))
        }
    }

    func updateReminder(
        id: String,
        patch: ReminderPatch,
        completion: @escaping (Result<CreatedReminder, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
            completion(.failure(.notFound))
            return
        }
        if let title = patch.title?.trimmingCharacters(in: .whitespacesAndNewlines), !title.isEmpty {
            reminder.title = title
        }
        if let due = patch.dueDate {
            reminder.dueDateComponents = Calendar.current.dateComponents(
                [.year, .month, .day, .hour, .minute],
                from: due
            )
            reminder.alarms = [EKAlarm(absoluteDate: due)]
        }
        if let recurrence = patch.recurrence,
           let rule = Self.ekRecurrenceRule(for: recurrence) {
            reminder.recurrenceRules = [rule]
        }
        if let list = reminderList(named: patch.listName) {
            reminder.calendar = list
        }
        do {
            try store.save(reminder, commit: true)
            let url = URL(string: "x-apple-reminderkit://REMCDReminder/\(reminder.calendarItemIdentifier)")
            completion(.success(CreatedReminder(id: reminder.calendarItemIdentifier, url: url)))
        } catch {
            completion(.failure(permissionAwareFailure(error)))
        }
    }

    func deleteReminder(
        id: String,
        completion: @escaping (Result<Void, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
            completion(.failure(.notFound))
            return
        }
        do {
            try store.remove(reminder, commit: true)
            completion(.success(()))
        } catch {
            completion(.failure(permissionAwareFailure(error)))
        }
    }

    func completeReminder(
        id: String,
        completion: @escaping (Result<Void, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
            completion(.failure(.notFound))
            return
        }
        reminder.isCompleted = true
        reminder.completionDate = Date()
        do {
            try store.save(reminder, commit: true)
            completion(.success(()))
        } catch {
            completion(.failure(permissionAwareFailure(error)))
        }
    }

    func snoozeReminder(
        id: String,
        by offset: TimeInterval,
        completion: @escaping (Result<CreatedReminder, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
            completion(.failure(.notFound))
            return
        }
        let base = reminder.dueDateComponents?.date ?? Date()
        let due = base.addingTimeInterval(offset)
        reminder.dueDateComponents = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute],
            from: due
        )
        reminder.alarms = [EKAlarm(absoluteDate: due)]
        do {
            try store.save(reminder, commit: true)
            let url = URL(string: "x-apple-reminderkit://REMCDReminder/\(reminder.calendarItemIdentifier)")
            completion(.success(CreatedReminder(id: reminder.calendarItemIdentifier, url: url)))
        } catch {
            completion(.failure(permissionAwareFailure(error)))
        }
    }

    func queryReminders(
        filter: ReminderQueryFilter,
        completion: @escaping (Result<[QueriedReminder], SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        let predicate: NSPredicate
        if let range = filter.dateRange {
            predicate = store.predicateForIncompleteReminders(
                withDueDateStarting: range.lowerBound,
                ending: range.upperBound,
                calendars: reminderCalendars(named: filter.listName)
            )
        } else {
            predicate = store.predicateForIncompleteReminders(
                withDueDateStarting: nil,
                ending: nil,
                calendars: reminderCalendars(named: filter.listName)
            )
        }
        store.fetchReminders(matching: predicate) { reminders in
            let text = filter.text?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let out = (reminders ?? []).filter { reminder in
                guard let text, !text.isEmpty else { return true }
                return (reminder.title ?? "").lowercased().contains(text)
            }.map { reminder in
                QueriedReminder(
                    id: reminder.calendarItemIdentifier,
                    title: reminder.title ?? "Reminder",
                    dueDate: reminder.dueDateComponents?.date,
                    url: URL(string: "x-apple-reminderkit://REMCDReminder/\(reminder.calendarItemIdentifier)")
                )
            }
            completion(.success(out))
        }
    }

    func removeReminder(
        matching text: String,
        fromList listName: String?,
        completion: @escaping (Result<Void, SinkError>) -> Void
    ) {
        queryReminders(filter: ReminderQueryFilter(dateRange: nil, text: text, listName: listName)) { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let err):
                completion(.failure(err))
            case .success(let rows):
                guard let first = rows.first else {
                    completion(.failure(.notFound))
                    return
                }
                self.deleteReminder(id: first.id, completion: completion)
            }
        }
    }

    func ensureReminderList(
        named name: String,
        completion: @escaping (Result<String, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            completion(.failure(.saveFailed(underlying: Self.syntheticError("list name was empty"))))
            return
        }
        guard let list = reminderList(named: name) else {
            completion(.failure(.saveFailed(underlying: Self.syntheticError("could not create reminder list"))))
            return
        }
        completion(.success(list.calendarIdentifier))
    }

    private static func syntheticError(_ message: String) -> NSError {
        NSError(domain: "com.juno.shell.reminders", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func reminderCalendars(named name: String?) -> [EKCalendar]? {
        guard let wanted = name?.trimmingCharacters(in: .whitespacesAndNewlines), !wanted.isEmpty else {
            return nil
        }
        let existing = store.calendars(for: .reminder).filter {
            $0.title.caseInsensitiveCompare(wanted) == .orderedSame
        }
        if !existing.isEmpty { return existing }
        if let created = reminderList(named: wanted) {
            return [created]
        }
        return nil
    }

    private func reminderList(named name: String?) -> EKCalendar? {
        guard let wanted = name?.trimmingCharacters(in: .whitespacesAndNewlines), !wanted.isEmpty else {
            return nil
        }
        if let existing = store.calendars(for: .reminder).first(where: {
            $0.title.caseInsensitiveCompare(wanted) == .orderedSame
        }) {
            return existing
        }
        let calendar = EKCalendar(for: .reminder, eventStore: store)
        calendar.title = wanted
        calendar.source = store.defaultCalendarForNewReminders()?.source ?? store.sources.first
        do {
            try store.saveCalendar(calendar, commit: true)
            return calendar
        } catch {
            return nil
        }
    }

    /// Translate a v3 ``JunoRecurrenceRule`` into ``EKRecurrenceRule``.
    /// Returns nil for unsupported shapes so the caller falls back to a
    /// single-fire reminder rather than crashing.
    static func ekRecurrenceRule(for rule: JunoRecurrenceRule) -> EKRecurrenceRule? {
        let freq: EKRecurrenceFrequency
        switch rule.freq.uppercased() {
        case "DAILY":   freq = .daily
        case "WEEKLY":  freq = .weekly
        case "MONTHLY": freq = .monthly
        case "YEARLY":  freq = .yearly
        default:        return nil
        }

        let daysOfTheWeek: [EKRecurrenceDayOfWeek]? = {
            guard !rule.byDay.isEmpty else { return nil }
            return rule.byDay.compactMap { code -> EKRecurrenceDayOfWeek? in
                guard let weekday = ekWeekdayForByDay(code) else { return nil }
                return EKRecurrenceDayOfWeek(weekday)
            }
        }()

        let daysOfTheMonth: [NSNumber]? = rule.byMonthDay.isEmpty
            ? nil
            : rule.byMonthDay.map { NSNumber(value: $0) }
        let monthsOfTheYear: [NSNumber]? = rule.byMonth.isEmpty
            ? nil
            : rule.byMonth.map { NSNumber(value: $0) }

        let end: EKRecurrenceEnd?
        if let count = rule.count {
            end = EKRecurrenceEnd(occurrenceCount: count)
        } else if let untilIso = rule.untilIso,
                  let until = Self.parseISO8601(untilIso) {
            end = EKRecurrenceEnd(end: until)
        } else {
            end = nil
        }

        return EKRecurrenceRule(
            recurrenceWith: freq,
            interval: rule.interval,
            daysOfTheWeek: daysOfTheWeek,
            daysOfTheMonth: daysOfTheMonth,
            monthsOfTheYear: monthsOfTheYear,
            weeksOfTheYear: nil,
            daysOfTheYear: nil,
            setPositions: nil,
            end: end
        )
    }

    private static func ekWeekdayForByDay(_ code: String) -> EKWeekday? {
        switch code.uppercased() {
        case "SU": return .sunday
        case "MO": return .monday
        case "TU": return .tuesday
        case "WE": return .wednesday
        case "TH": return .thursday
        case "FR": return .friday
        case "SA": return .saturday
        default:   return nil
        }
    }

    private static func parseISO8601(_ iso: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: iso) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: iso)
    }
}
