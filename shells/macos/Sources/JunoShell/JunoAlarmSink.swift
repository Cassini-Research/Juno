// JunoAlarmSink.swift
//
// "Alarm" is intentionally not a new alarm app. We model alarms as
// short calendar events with an alert, written via EventKit. Juno prefers a
// dedicated **Juno Alarms** calendar and falls back to the user's default
// writable event calendar when macOS will not let us create that calendar.
// The user gets a real notification at the requested time whether Juno is
// running or not, and the OS handles delivery.
//
// **Hard rule:** never block the caller; never throw. Every method
// returns through a completion handler with a `Result`. A missing time
// is treated as a sink error (an alarm with no time isn't an alarm) —
// the executor maps that to `.timeParseFailed`.

import EventKit
import Foundation

final class JunoAlarmSink {

    static let shared = JunoAlarmSink()

    private let store: EKEventStore
    private let grantLock = NSLock()
    private var optimisticGrantUntil: Date?
    private static let optimisticGrantTTL: TimeInterval = 30

    private init(store: EKEventStore = EKEventStore()) {
        self.store = store
    }

    // MARK: - Authorization

    var authorization: JunoReminderSink.Authorization {
        let live = Self.authorizationStatus()
        switch live {
        case .granted:
            clearOptimisticGrant()
            JunoEventKitGrantCache.setGrant(.calendarEvents, granted: true)
            return .granted
        case .notDetermined:
            if hasOptimisticGrant() || JunoEventKitGrantCache.hasGrant(.calendarEvents) {
                resetEventStore()
                return .granted
            }
            return .notDetermined
        case .denied, .restricted:
            clearOptimisticGrant()
            JunoEventKitGrantCache.setGrant(.calendarEvents, granted: false)
            return live
        }
    }

    static func authorizationStatus() -> JunoReminderSink.Authorization {
        let raw = EKEventStore.authorizationStatus(for: .event)
        switch raw {
        case .authorized: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        case .fullAccess: return .granted
        case .writeOnly:
            // Write-only ("Add Events Only" in the macOS dialog) is the
            // less-invasive option Apple recommends for write-mostly
            // apps. Previously we treated it as ``.notDetermined`` so
            // the sink could keep using a dedicated "Juno Alarms"
            // calendar — but that meant a user who picked the
            // recommended option saw "Set up" and the alarm never
            // fired. Now we accept it: the sink falls back to the
            // user's default writable calendar when it can't create or
            // read the dedicated calendar.
            return .granted
        @unknown default:
            return .notDetermined
        }
    }

    /// Reset the long-lived EventKit store after a grant or external
    /// permission change. Apple documents that stores touched before a
    /// prompt may need resetting before they see newly granted data.
    func resetEventStore() {
        store.reset()
    }

    func requestAccess(_ completion: @escaping (JunoReminderSink.Authorization) -> Void) {
        let requestStore = EKEventStore()
        if #available(macOS 14.0, *) {
            requestStore.requestWriteOnlyAccessToEvents { [requestStore, weak self] granted, _ in
                _ = requestStore
                if granted {
                    self?.rememberOptimisticGrant()
                    completion(.granted)
                } else {
                    completion(Self.authorizationStatus())
                }
            }
        } else {
            requestStore.requestAccess(to: .event) { [requestStore, weak self] granted, _ in
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
        JunoEventKitGrantCache.setGrant(.calendarEvents, granted: true)
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

    // MARK: - Create

    struct CreatedAlarm {
        let id: String
        let url: URL?
    }

    struct AlarmPatch {
        let title: String?
        let fireDate: Date?
        let recurrence: JunoRecurrenceRule?
    }

    enum SinkError: Error, LocalizedError {
        case permissionDenied
        case missingTime
        case calendarUnavailable
        case notFound
        case saveFailed(underlying: Error)

        var errorDescription: String? {
            switch self {
            case .permissionDenied: return "Calendar access is not granted."
            case .missingTime: return "An alarm needs a time."
            case .calendarUnavailable: return "Couldn't find a writable Calendar destination for alarms."
            case .notFound: return "Couldn't find that alarm."
            case .saveFailed(let underlying): return "Couldn't save alarm: \(underlying.localizedDescription)"
            }
        }
    }

    /// Create one alarm. `title` becomes the calendar event name; `at`
    /// is the absolute trigger time of the *first* occurrence. Event
    /// runs for 1 minute (just to have a sane block on the user's
    /// calendar; the alert is what matters).
    ///
    /// When ``recurrence`` is non-nil, the calendar event is saved with
    /// an ``EKRecurrenceRule`` so the alert fires repeatedly on the
    /// schedule the user spoke ("every weekday at 6 am" → 5 alerts a
    /// week, not 1 calendar entry per day).
    func createAlarm(
        title: String,
        at fireDate: Date?,
        recurrence: JunoRecurrenceRule? = nil,
        completion: @escaping (Result<CreatedAlarm, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let fireDate else {
            completion(.failure(.missingTime))
            return
        }
        guard let calendar = writableAlarmCalendar() else {
            completion(.failure(.calendarUnavailable))
            return
        }

        let event = EKEvent(eventStore: store)
        event.title = title.isEmpty ? "Alarm" : title
        event.calendar = calendar
        event.startDate = fireDate
        event.endDate = fireDate.addingTimeInterval(60)
        event.addAlarm(EKAlarm(absoluteDate: fireDate))
        if let rule = recurrence,
           let ekRule = JunoReminderSink.ekRecurrenceRule(for: rule) {
            event.recurrenceRules = [ekRule]
        }

        do {
            try store.save(event, span: .thisEvent, commit: true)
            let id = event.eventIdentifier ?? event.calendarItemIdentifier
            // Calendar.app deep link — `ical://showEvent?id=` is widely
            // supported; on failure the URL just opens Calendar.
            let url = URL(string: "ical://showEvent?id=\(id)")
            completion(.success(CreatedAlarm(id: id, url: url)))
        } catch {
            completion(.failure(.saveFailed(underlying: error)))
        }
    }

    func updateAlarm(
        id: String,
        patch: AlarmPatch,
        completion: @escaping (Result<CreatedAlarm, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let event = store.event(withIdentifier: id) else {
            completion(.failure(.notFound))
            return
        }
        if let title = patch.title?.trimmingCharacters(in: .whitespacesAndNewlines), !title.isEmpty {
            event.title = title
        }
        if let fireDate = patch.fireDate {
            event.startDate = fireDate
            event.endDate = fireDate.addingTimeInterval(60)
            event.alarms = [EKAlarm(absoluteDate: fireDate)]
        }
        if let recurrence = patch.recurrence,
           let rule = JunoReminderSink.ekRecurrenceRule(for: recurrence) {
            event.recurrenceRules = [rule]
        }
        do {
            try store.save(event, span: .thisEvent, commit: true)
            let storedId = event.eventIdentifier ?? event.calendarItemIdentifier
            completion(.success(CreatedAlarm(id: storedId, url: URL(string: "ical://showEvent?id=\(storedId)"))))
        } catch {
            completion(.failure(.saveFailed(underlying: error)))
        }
    }

    func deleteAlarm(
        id: String,
        completion: @escaping (Result<Void, SinkError>) -> Void
    ) {
        guard authorization == .granted else {
            completion(.failure(.permissionDenied))
            return
        }
        guard let event = store.event(withIdentifier: id) else {
            completion(.failure(.notFound))
            return
        }
        do {
            try store.remove(event, span: .thisEvent, commit: true)
            completion(.success(()))
        } catch {
            completion(.failure(.saveFailed(underlying: error)))
        }
    }

    // MARK: - Calendar

    private func writableAlarmCalendar() -> EKCalendar? {
        let rawAuthorization = EKEventStore.authorizationStatus(for: .event)
        if #available(macOS 14.0, *),
           rawAuthorization == .writeOnly
            || (rawAuthorization == .notDetermined && JunoEventKitGrantCache.hasGrant(.calendarEvents))
        {
            // Write-only Calendar access exposes a virtual writable calendar.
            // We cannot read or create a dedicated "Juno Alarms" calendar in
            // this mode, but saving an event to the default/virtual calendar
            // is exactly the privacy-preserving path Apple provides. The
            // persisted-grant branch covers EventKit's post-grant lag where
            // the static status still reports notDetermined even though TCC
            // has accepted the grant.
            return store.defaultCalendarForNewEvents
                ?? store.calendars(for: .event).first
        }
        if let dedicated = ensureJunoAlarmsCalendar(), dedicated.allowsContentModifications {
            return dedicated
        }
        if let defaultCalendar = store.defaultCalendarForNewEvents,
           defaultCalendar.allowsContentModifications {
            return defaultCalendar
        }
        return store.calendars(for: .event).first(where: { $0.allowsContentModifications })
    }

    /// Find or create the Juno Alarms calendar in the user's default
    /// event source (iCloud preferred, falls back to local).
    private func ensureJunoAlarmsCalendar() -> EKCalendar? {
        let name = "Juno Alarms"
        if let existing = store.calendars(for: .event).first(where: { $0.title == name }) {
            return existing
        }
        let calendar = EKCalendar(for: .event, eventStore: store)
        calendar.title = name
        calendar.cgColor = NSColor(calibratedRed: 0.55, green: 0.45, blue: 0.95, alpha: 1).cgColor
        let source = preferredSource()
        guard let source else { return nil }
        calendar.source = source
        do {
            try store.saveCalendar(calendar, commit: true)
            return calendar
        } catch {
            return nil
        }
    }

    private func preferredSource() -> EKSource? {
        // iCloud first; otherwise first source that supports calendars;
        // local last.
        let sources = store.sources
        if let cloud = sources.first(where: { $0.sourceType == .calDAV && $0.title.lowercased().contains("icloud") }) {
            return cloud
        }
        if let calDAV = sources.first(where: { $0.sourceType == .calDAV }) {
            return calDAV
        }
        return sources.first(where: { $0.sourceType == .local })
            ?? sources.first
    }
}

#if canImport(AppKit)
import AppKit
#endif
