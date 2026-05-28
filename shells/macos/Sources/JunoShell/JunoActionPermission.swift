// JunoActionPermission.swift
//
// Unified permission model across every action kind. The Actions page,
// Home priority card, and Settings all read from one shape:
//
//   * `JunoActionPermissionDescriptor` — what kind of permission this
//     action needs (EventKit Reminders, Automation→Notes, …).
//   * `JunoActionPermissionStatus`     — single enum across all backends.
//   * `JunoActionPermissionStore`      — observable singleton that polls
//     each descriptor on a 1s tick while any view is observing it.
//
// Why a store and not per-card pollers? The old VoiceActionsBanner polled
// EKEventStore.authorizationStatus every 100ms while expanded — wasteful,
// and Notes Automation status was never polled at all. One store, one
// rhythm, one place to add new permissions.

import AppKit
import ApplicationServices
import Combine
import EventKit
import Foundation
import os.log

private let permLog = OSLog(subsystem: "com.juno.shell", category: "actions.perm")

// MARK: - Descriptor

enum JunoActionPermissionDescriptor: Hashable {
    /// EventKit reminders grant.
    case reminders
    /// EventKit calendar-event grant. Used by Alarm actions.
    case calendarEvents
    /// AppleScript Automation → Notes. Probed through the Apple Events
    /// permission API; actual note creation sends the AppleScript.
    case notesAutomation
}

enum JunoEventKitGrantCache {
    private static func key(_ descriptor: JunoActionPermissionDescriptor) -> String? {
        switch descriptor {
        case .reminders: return "JunoActionEverGranted.reminders"
        case .calendarEvents: return "JunoActionEverGranted.calendarEvents"
        case .notesAutomation: return nil
        }
    }

    static func hasGrant(_ descriptor: JunoActionPermissionDescriptor) -> Bool {
        guard let key = key(descriptor) else { return false }
        return UserDefaults.standard.bool(forKey: key)
    }

    static func setGrant(_ descriptor: JunoActionPermissionDescriptor, granted: Bool) {
        guard let key = key(descriptor) else { return }
        UserDefaults.standard.set(granted, forKey: key)
    }
}

extension JunoActionPermissionDescriptor {
    /// Short label used in the Actions page status row.
    var label: String {
        switch self {
        case .reminders: return "Apple Reminders access"
        case .calendarEvents: return "Apple Calendar access for alarms"
        case .notesAutomation: return "Apple Notes automation"
        }
    }

    /// The deep-link to System Settings for the denied state.
    var systemSettingsURL: URL? {
        switch self {
        case .reminders:
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Reminders")
        case .calendarEvents:
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Calendars")
        case .notesAutomation:
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation")
        }
    }
}

// MARK: - Status

enum JunoActionPermissionStatus: Equatable {
    case granted
    case denied
    case restricted
    case notDetermined
    /// The required app isn't installed (e.g. Notes.app missing on a
    /// stripped server build). UI presents an "install Notes" hint.
    case notInstalled
}

extension JunoActionPermissionStatus {
    var isGranted: Bool { self == .granted }
    var needsUserAction: Bool {
        switch self {
        case .granted: return false
        case .restricted: return true
        case .denied, .notDetermined, .notInstalled: return true
        }
    }
}

// MARK: - Store

/// Observable wrapper. Views observe `@Published` fields and the store
/// polls every active descriptor on a 1Hz tick — cheap, accurate to
/// within a second when the user grants/revokes elsewhere.
@MainActor
final class JunoActionPermissionStore: ObservableObject {

    static let shared = JunoActionPermissionStore()

    @Published private(set) var statuses: [JunoActionPermissionDescriptor: JunoActionPermissionStatus] = [:]
    @Published private(set) var requestInFlight: Set<JunoActionPermissionDescriptor> = []

    /// True while we're actively asking macOS for Notes Automation
    /// consent.
    @Published private(set) var isProbingNotesAutomation: Bool = false

    private var pollTimer: AnyCancellable?
    private var activationObserver: NSObjectProtocol?
    private var observerCount = 0
    private var notesProbeInFlight = false
    private var optimisticEventKitGrants: [JunoActionPermissionDescriptor: Date] = [:]
    private let optimisticGrantTTL: TimeInterval = 30
    private let prober = JunoNotesAutomationProber()

    private init() {
        refreshAll()
    }

    func status(for descriptor: JunoActionPermissionDescriptor) -> JunoActionPermissionStatus {
        statuses[descriptor] ?? .notDetermined
    }

    func status(for kind: JunoActionKind) -> JunoActionPermissionStatus {
        status(for: kind.descriptor.permission)
    }

    func isRequesting(_ descriptor: JunoActionPermissionDescriptor) -> Bool {
        requestInFlight.contains(descriptor)
    }

    /// Snapshot all descriptors. Cheap; no AppleScript runs while polling.
    func refreshAll(forceNotesProbe: Bool = false) {
        setStatus(reconciledEventKitStatus(for: .reminders, live: readReminders()), for: .reminders)
        setStatus(reconciledEventKitStatus(for: .calendarEvents, live: readCalendarEvents()), for: .calendarEvents)
        if forceNotesProbe {
            revalidateNotesAutomation()
        } else if let cached = prober.cachedStatusIfFresh() {
            statuses[.notesAutomation] = cached
        } else {
            refreshNotesAutomationIfNeeded()
        }
    }

    /// Hint to the prober that we want a fresh probe of Notes Automation
    /// even if the cache is warm. Call from the Actions page when the
    /// user explicitly clicks "Allow" or returns from System Settings.
    func revalidateNotesAutomation() {
        guard !notesProbeInFlight else { return }
        notesProbeInFlight = true
        requestInFlight.insert(.notesAutomation)
        prober.invalidate()
        prober.probe { [weak self] status in
            Task { @MainActor in
                self?.notesProbeInFlight = false
                self?.requestInFlight.remove(.notesAutomation)
                self?.statuses[.notesAutomation] = status
            }
        }
    }

    /// Keep Notes Automation status live without launching Notes or
    /// AppleScript. Normal polling reads the 30s prober cache; only a
    /// stale/missing cache starts one background preflight probe.
    private func refreshNotesAutomationIfNeeded() {
        guard !notesProbeInFlight else { return }
        notesProbeInFlight = true
        prober.probe { [weak self] status in
            Task { @MainActor in
                self?.notesProbeInFlight = false
                self?.statuses[.notesAutomation] = status
            }
        }
    }

    // MARK: - Active polling

    /// Start the polling tick if not already running. Mirrors how
    /// `JunoCapabilitySnapshot` and similar surfaces opt into refresh.
    func beginObserving() {
        observerCount += 1
        guard pollTimer == nil else { return }
        pollTimer = Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.refreshAll() }
        activationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.refreshAll(forceNotesProbe: true)
            }
        }
    }

    func endObserving() {
        observerCount = max(0, observerCount - 1)
        if observerCount == 0 {
            pollTimer?.cancel()
            pollTimer = nil
            if let activationObserver {
                NotificationCenter.default.removeObserver(activationObserver)
                self.activationObserver = nil
            }
        }
    }

    // MARK: - Backends

    private func readReminders() -> JunoActionPermissionStatus {
        let raw = EKEventStore.authorizationStatus(for: .reminder)
        switch raw {
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        case .authorized: return .granted
        case .fullAccess, .writeOnly: return .granted
        @unknown default:
            return .notDetermined
        }
    }

    private func readCalendarEvents() -> JunoActionPermissionStatus {
        let raw = EKEventStore.authorizationStatus(for: .event)
        switch raw {
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        case .authorized: return .granted
        case .fullAccess: return .granted
        case .writeOnly:
            // "Add Events Only" is sufficient. The sink falls back from
            // its dedicated "Juno Alarms" calendar to the user's default
            // writable calendar when the dedicated path isn't available.
            // See ``JunoAlarmSink.authorizationStatus`` for the matching
            // policy.
            return .granted
        @unknown default:
            return .notDetermined
        }
    }

    /// Trigger the EventKit access prompt. The completion runs on main.
    func requestReminders(_ completion: @escaping (JunoActionPermissionStatus) -> Void) {
        requestInFlight.insert(.reminders)
        JunoReminderSink.shared.requestAccess { auth in
            Task { @MainActor in
                var immediate = self.status(from: auth)
                if immediate == .granted {
                    self.rememberOptimisticEventKitGrant(for: .reminders)
                } else {
                    immediate = self.reconciledEventKitStatus(for: .reminders, live: immediate)
                }
                self.requestInFlight.remove(.reminders)
                self.setStatus(immediate, for: .reminders)
                completion(immediate)
                self.scheduleEventKitRefresh(for: .reminders)
            }
        }
    }

    func requestCalendarEvents(_ completion: @escaping (JunoActionPermissionStatus) -> Void) {
        requestInFlight.insert(.calendarEvents)
        JunoAlarmSink.shared.requestAccess { auth in
            Task { @MainActor in
                var immediate = self.status(from: auth)
                if immediate == .granted {
                    self.rememberOptimisticEventKitGrant(for: .calendarEvents)
                } else {
                    immediate = self.reconciledEventKitStatus(for: .calendarEvents, live: immediate)
                }
                self.requestInFlight.remove(.calendarEvents)
                self.setStatus(immediate, for: .calendarEvents)
                completion(immediate)
                self.scheduleEventKitRefresh(for: .calendarEvents)
            }
        }
    }

    private func status(from auth: JunoReminderSink.Authorization) -> JunoActionPermissionStatus {
        switch auth {
        case .granted: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        }
    }

    private func scheduleEventKitRefresh(for descriptor: JunoActionPermissionDescriptor) {
        refreshEventKitStatus(for: descriptor)
        for delay in [0.2, 0.6, 1.2, 2.5, 5.0] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                Task { @MainActor in
                    self?.refreshEventKitStatus(for: descriptor)
                }
            }
        }
    }

    private func refreshEventKitStatus(for descriptor: JunoActionPermissionDescriptor) {
        switch descriptor {
        case .reminders:
            setStatus(reconciledEventKitStatus(for: .reminders, live: readReminders()), for: .reminders)
        case .calendarEvents:
            setStatus(reconciledEventKitStatus(for: .calendarEvents, live: readCalendarEvents()), for: .calendarEvents)
        case .notesAutomation:
            revalidateNotesAutomation()
        }
    }

    private func rememberOptimisticEventKitGrant(for descriptor: JunoActionPermissionDescriptor) {
        optimisticEventKitGrants[descriptor] = Date().addingTimeInterval(optimisticGrantTTL)
        JunoEventKitGrantCache.setGrant(descriptor, granted: true)
    }

    private func reconciledEventKitStatus(
        for descriptor: JunoActionPermissionDescriptor,
        live: JunoActionPermissionStatus
    ) -> JunoActionPermissionStatus {
        switch live {
        case .granted:
            optimisticEventKitGrants[descriptor] = nil
            JunoEventKitGrantCache.setGrant(descriptor, granted: true)
            return .granted
        case .notDetermined:
            if let until = optimisticEventKitGrants[descriptor], until > Date() {
                return .granted
            }
            optimisticEventKitGrants[descriptor] = nil
            // Cold-launch quirk: after a prior grant via
            // requestWriteOnlyAccessToEvents / requestFullAccessToReminders,
            // the static EKEventStore.authorizationStatus(...) call can come
            // back .notDetermined for a while even though TCC still holds
            // the grant (this is why re-clicking Allow on the Actions page
            // succeeds without a system prompt). Trust the persisted
            // "ever granted in this profile" flag in that case so the
            // Home "Set up" card doesn't flap. The flag is cleared when
            // the OS reports a real .denied / .restricted below, so a
            // user who actually revokes in System Settings still sees the
            // accurate state.
            if JunoEventKitGrantCache.hasGrant(descriptor) {
                return .granted
            }
            return .notDetermined
        case .denied, .restricted, .notInstalled:
            optimisticEventKitGrants[descriptor] = nil
            JunoEventKitGrantCache.setGrant(descriptor, granted: false)
            return live
        }
    }

    private func setStatus(
        _ status: JunoActionPermissionStatus,
        for descriptor: JunoActionPermissionDescriptor
    ) {
        let previous = statuses[descriptor]
        if status == .granted && previous != .granted {
            resetEventKitStore(for: descriptor)
        }
        statuses[descriptor] = status
    }

    private func resetEventKitStore(for descriptor: JunoActionPermissionDescriptor) {
        switch descriptor {
        case .reminders:
            JunoReminderSink.shared.resetEventStore()
        case .calendarEvents:
            JunoAlarmSink.shared.resetEventStore()
        case .notesAutomation:
            break
        }
    }

    /// Trigger Notes Automation consent. Passive refreshes never launch
    /// Notes, but an explicit Allow click is allowed to send one harmless
    /// AppleEvent so macOS creates the Automation row and consent prompt.
    func requestNotesAutomation(_ completion: @escaping (JunoActionPermissionStatus) -> Void) {
        notesProbeInFlight = true
        isProbingNotesAutomation = true
        requestInFlight.insert(.notesAutomation)
        JunoWindowActivation.activateApp()

        prober.requestConsent { [weak self] status in
            Task { @MainActor in
                guard let self else { return }
                self.notesProbeInFlight = false
                self.isProbingNotesAutomation = false
                self.requestInFlight.remove(.notesAutomation)
                self.statuses[.notesAutomation] = status
                os_log("notes_consent result=%{public}@", log: permLog, type: .info, String(describing: status))
                completion(status)
            }
        }
    }

    /// Open System Settings → Privacy & Security → Automation. Tries the
    /// modern URL first, then the legacy URL, then a final fallback that
    /// just opens the Settings app — better than the silent no-op users
    /// hit when one URL form fails on a given macOS version.
    func openAutomationSettings() {
        JunoSystemSettingsLinks.openAutomationPrivacy()
        os_log("opened automation privacy settings", log: permLog, type: .info)
    }

    func openRemindersSettings() {
        JunoSystemSettingsLinks.openRemindersPrivacy()
    }

    func openCalendarSettings() {
        JunoSystemSettingsLinks.openCalendarsPrivacy()
    }
}

// MARK: - Notes Automation prober

/// Probes Automation permission for Notes.app. Caches the result for 30s so
/// repeat reads from the Settings/Actions surfaces don't keep touching TCC.
final class JunoNotesAutomationProber {

    private let queue = DispatchQueue(label: "juno.notes-prober", qos: .utility)
    private var cached: (status: JunoActionPermissionStatus, at: Date)?
    private let ttl: TimeInterval = 30

    /// Returns the cached value if present and fresh; otherwise nil so
    /// the store can decide whether to kick one background probe.
    func cachedStatusIfFresh() -> JunoActionPermissionStatus? {
        if let cached, Date().timeIntervalSince(cached.at) < ttl {
            return cached.status
        }
        return nil
    }

    func invalidate() {
        cached = nil
    }

    func cacheStatus(_ status: JunoActionPermissionStatus) {
        cached = (status, Date())
    }

    func requestConsent(_ completion: @escaping (JunoActionPermissionStatus) -> Void) {
        queue.async { [weak self] in
            let status = Self.runConsentRequest()
            self?.cached = (status, Date())
            completion(status)
        }
    }

    /// Trigger the macOS Automation consent flow for Notes.app.
    ///
    /// `AEDeterminePermissionToAutomateTarget(..., askUserIfNeeded: true)`
    /// is only a preflight. On some macOS releases it returns `procNotFound`
    /// or `errAEEventWouldRequireUserConsent` without creating the visible
    /// System Settings > Automation row, especially when Notes is not already
    /// running. After that preflight misses, send one harmless read-only
    /// AppleEvent. That is the event macOS uses to register Juno -> Notes in
    /// TCC and show the real consent prompt.
    static func runConsentRequest() -> JunoActionPermissionStatus {
        if NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Notes") == nil {
            return .notInstalled
        }
        let preflight = determinePermission(askUserIfNeeded: true)
        switch preflight {
        case .granted, .denied, .restricted, .notInstalled:
            return preflight
        case .notDetermined:
            return runBenignNotesAutomationRequest()
        }
    }

    private static func runBenignNotesAutomationRequest() -> JunoActionPermissionStatus {
        let source = """
        tell application id "com.apple.Notes"
            return count of accounts
        end tell
        """
        guard let script = NSAppleScript(source: source) else {
            os_log("notes_consent_script_compile_failed", log: permLog, type: .error)
            return .notDetermined
        }
        var errorInfo: NSDictionary?
        _ = script.executeAndReturnError(&errorInfo)
        if let err = errorInfo {
            let code = (err[NSAppleScript.errorNumber] as? Int) ?? 0
            let message = (err[NSAppleScript.errorMessage] as? String) ?? "AppleScript error"
            switch code {
            case Int(errAEEventNotPermitted):  // -1743
                return .denied
            case Int(errAEEventWouldRequireUserConsent), Int(procNotFound), -128:  // -1744, app unavailable, user cancelled
                return .notDetermined
            default:
                os_log("notes_consent_script_error code=%{public}d message=%{public}@", log: permLog, type: .error, code, message)
                return determinePermission(askUserIfNeeded: false)
            }
        }
        return .granted
    }

    private static func determinePermission(askUserIfNeeded: Bool) -> JunoActionPermissionStatus {
        guard let target = NSAppleEventDescriptor(bundleIdentifier: "com.apple.Notes").aeDesc?.pointee else {
            return .notDetermined
        }
        var t = target
        let err = AEDeterminePermissionToAutomateTarget(&t, typeWildCard, typeWildCard, askUserIfNeeded)
        switch err {
        case OSStatus(noErr):
            return .granted
        case OSStatus(errAEEventNotPermitted):  // -1743
            return .denied
        case OSStatus(errAEEventWouldRequireUserConsent):  // -1744
            return .notDetermined
        case OSStatus(procNotFound):  // Notes not running
            return .notDetermined
        default:
            os_log("automation_determine_unknown_err code=%{public}d ask=%{public}@", log: permLog, type: .error, Int(err), askUserIfNeeded ? "true" : "false")
            return .notDetermined
        }
    }

    /// Run the probe on a background queue and call back on an arbitrary
    /// thread with the resolved status.
    func probe(_ completion: @escaping (JunoActionPermissionStatus) -> Void) {
        queue.async { [weak self] in
            let status = Self.runProbe()
            self?.cached = (status, Date())
            completion(status)
        }
    }

    /// Passive status check. Uses the same Apple Events permission API
    /// as the explicit consent request, but with `askUserIfNeeded=false`
    /// so background polling never surfaces a TCC prompt or launches a
    /// visible Notes workflow.
    private static func runProbe() -> JunoActionPermissionStatus {
        if NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Notes") == nil {
            return .notInstalled
        }
        return determinePermission(askUserIfNeeded: false)
    }
}
