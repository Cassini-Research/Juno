// JunoActionExecutor.swift
//
// Coordinator that receives parsed ``JunoActionRequest`` items from a
// broker dictation response, dispatches each to its sink, and posts the
// resulting ``JunoActionResult`` array back to the broker history row.
//
// The executor is intentionally stateless other than caching sinks. It
// publishes a small ``@Published`` "did the user attempt an action
// without permission" flag so the Home page nudge can switch from cold
// to warm copy at the moment of intent.
//
// **Hard rule:** the executor must never crash, throw, or block the
// caller. Sinks that fail produce error results; the user's history row
// records the failure with full context so the HUD can surface a
// "couldn't save" affordance later.

import Combine
import Foundation
import os.log

private let actionLog = OSLog(subsystem: "com.juno.shell", category: "actions")

@MainActor
final class JunoActionExecutor: ObservableObject {

    static let shared = JunoActionExecutor()

    /// Set to ``true`` whenever the user dictates an action utterance and
    /// at least one action could not run because permission is missing.
    /// The Home-page nudge observes this and switches to "Heads up — that
    /// reminder didn't save." copy. The flag clears once the user has
    /// either granted access or explicitly dismissed the nudge.
    @Published private(set) var pendingPermissionAttempt: Bool = false

    /// Brief preview of the most recent unfulfilled action, for the warm
    /// nudge ("Juno heard: *call sam tomorrow at 9am*"). Cleared when the
    /// pending attempt flag clears.
    @Published private(set) var lastUnfulfilledPreview: String?

    /// Most recent batch of completed actions, paired with the originating
    /// requests so chip views can show inferred-time badges, error states,
    /// and deep-link affordances. Set when ``execute`` finishes; cleared
    /// via :meth:`clearRecent` once the HUD/Home view is done with it.
    @Published private(set) var recentBatch: ActionBatch?

    /// Snapshot of the actions currently being dispatched. Set the moment
    /// ``execute`` begins, cleared the moment its completion fires. Drives
    /// the brand-island "Saving N notes…" working state so the HUD doesn't
    /// vanish during the AppleScript round-trip — that gap previously
    /// looked like the action had silently failed.
    @Published private(set) var inFlight: InFlight?

    struct InFlight: Equatable {
        let utteranceId: String
        let kinds: [JunoActionKind]
        var completed: Int

        var total: Int { kinds.count }
        var remaining: Int { max(0, total - completed) }
    }

    /// Pairs a batch of executed actions with their results. The chip stack
    /// renders one row per request in the original order.
    struct ActionBatch: Equatable {
        let utteranceId: String
        let requests: [JunoActionRequest]
        let results: [JunoActionResult]
        let completedAt: Date

        var hasInferredTime: Bool {
            requests.contains {
                $0.when?.inferred == true
                    || $0.when?.needsConfirmation == true
                    || $0.schedule?.vague?.needsConfirmation == true
            }
        }
    }

    func clearRecent() { recentBatch = nil }

    private let reminders: JunoReminderSink
    private let notes: JunoNotesSink
    private let alarms: JunoAlarmSink

    init(
        reminders: JunoReminderSink = .shared,
        notes: JunoNotesSink = .shared,
        alarms: JunoAlarmSink = .shared
    ) {
        self.reminders = reminders
        self.notes = notes
        self.alarms = alarms
    }

    // MARK: - Entry point

    /// Run every action and post the results back to the broker.
    ///
    /// - Parameters:
    ///   - utteranceId: the broker history row to update.
    ///   - actions: the parsed actions, in order. The order is preserved
    ///     in the result array so HUD chips render in the same order the
    ///     user said them.
    ///   - postResults: dependency-injected for testability. When nil
    ///     (the default), the live broker endpoint is used.
    ///   - completion: called once with the final result array.
    func execute(
        utteranceId: String,
        actions: [JunoActionRequest],
        postResults: ((String, [JunoActionResult]) -> Void)? = nil,
        completion: @escaping ([JunoActionResult]) -> Void = { _ in }
    ) {
        guard !actions.isEmpty else {
            completion([])
            return
        }
        // Verification lane: JUNO_ACTIONS_DRY_RUN=1 exercises the full
        // shell path — HUD in-flight states, result chips, broker result
        // posting, history rows — without creating real Reminders /
        // Calendar events / Notes. Used by installed-app gates.
        if ProcessInfo.processInfo.environment["JUNO_ACTIONS_DRY_RUN"] == "1" {
            os_log(
                "execute DRY-RUN uid=%{public}@ count=%{public}d",
                log: actionLog, type: .info, utteranceId, actions.count
            )
            let results = actions.map { req in
                JunoActionResult(
                    junoId: Self.junoId(for: req),
                    kind: req.kind,
                    status: .ok,
                    bodyPreview: String(req.body.prefix(80)),
                    body: req.body,
                    whenIso: Self.primaryIso(for: req),
                    error: "dry-run (no native side effects)"
                )
            }
            let post = postResults ?? Self.postResultsToBroker(utteranceId:results:)
            post(utteranceId, results)
            completion(results)
            return
        }
        os_log(
            "execute uid=%{public}@ count=%{public}d kinds=%{public}@",
            log: actionLog, type: .info,
            utteranceId, actions.count,
            actions.map { $0.kind.rawValue }.joined(separator: ",")
        )
        let post = postResults ?? Self.postResultsToBroker(utteranceId:results:)
        inFlight = InFlight(
            utteranceId: utteranceId,
            kinds: actions.map { $0.kind },
            completed: 0
        )
        runSerially(
            actions: actions,
            accumulator: [],
            onProgress: { [weak self] _ in
                guard let self, var snapshot = self.inFlight,
                      snapshot.utteranceId == utteranceId else { return }
                snapshot.completed += 1
                self.inFlight = snapshot
            }
        ) { [weak self] results in
            guard let self else { return }
            post(utteranceId, results)
            self.refreshPendingFlag(from: results, sourceActions: actions)
            self.recentBatch = ActionBatch(
                utteranceId: utteranceId,
                requests: actions,
                results: results,
                completedAt: Date()
            )
            if self.inFlight?.utteranceId == utteranceId {
                self.inFlight = nil
            }
            os_log(
                "execute_done uid=%{public}@ statuses=%{public}@",
                log: actionLog, type: .info,
                utteranceId,
                results.map { $0.status.rawValue }.joined(separator: ",")
            )
            completion(results)
        }
    }

    // MARK: - Blocked path
    //
    // Called when actions arrived in the broker response but were not
    // dispatched (Voice Actions toggle off, no permissions, etc.). Surfaces
    // a chip per action so the user understands *why nothing happened*
    // instead of seeing dead silence — the silent-failure mode that made
    // earlier builds feel broken.
    func recordBlocked(
        utteranceId: String,
        actions: [JunoActionRequest],
        reason: BlockedReason
    ) {
        guard !actions.isEmpty else { return }
        os_log(
            "blocked uid=%{public}@ count=%{public}d reason=%{public}@",
            log: actionLog, type: .info,
            utteranceId, actions.count, reason.rawValue
        )
        let status: JunoActionStatus
        let errorCopy: String
        switch reason {
        case .toggleOff:
            status = .blockedToggleOff
            errorCopy = "Voice Actions are off — turn them on in Actions."
        case .missingPermission:
            status = .blockedNoPermission
            errorCopy = "Permission missing — open Actions to grant access."
        }
        let results = actions.map { req -> JunoActionResult in
            JunoActionResult(
                junoId: Self.junoId(for: req),
                kind: req.kind,
                status: status,
                bodyPreview: String(req.body.prefix(80)),
                body: req.body,
                whenIso: Self.primaryIso(for: req),
                error: errorCopy
            )
        }
        Self.postResultsToBroker(utteranceId: utteranceId, results: results)
        recentBatch = ActionBatch(
            utteranceId: utteranceId,
            requests: actions,
            results: results,
            completedAt: Date()
        )
        if reason == .missingPermission {
            pendingPermissionAttempt = true
            lastUnfulfilledPreview = actions.first?.body
        }
    }

    enum BlockedReason: String {
        case toggleOff = "toggle_off"
        case missingPermission = "missing_permission"
    }

    // MARK: - Pending-permission flag

    /// Called by the nudge card when the user explicitly dismisses the
    /// warm copy or after permission is granted. Either way the executor
    /// stops advertising "you tried to set a reminder and it failed."
    func clearPendingPermissionAttempt() {
        pendingPermissionAttempt = false
        lastUnfulfilledPreview = nil
    }

    private func refreshPendingFlag(
        from results: [JunoActionResult],
        sourceActions: [JunoActionRequest]
    ) {
        let firstDenied = results.firstIndex {
            $0.status == .permissionDenied || $0.status == .blockedNoPermission
        }
        if let idx = firstDenied {
            pendingPermissionAttempt = true
            // Use the original parsed body so the nudge can show what
            // Juno heard, not a sink-side preview.
            if idx < sourceActions.count {
                lastUnfulfilledPreview = sourceActions[idx].body
            }
        }
    }

    // MARK: - Dispatch

    private func runSerially(
        actions: [JunoActionRequest],
        accumulator: [JunoActionResult],
        linkResults: [String: JunoActionResult] = [:],
        onProgress: @escaping (JunoActionResult) -> Void = { _ in },
        completion: @escaping ([JunoActionResult]) -> Void
    ) {
        guard let head = actions.first else {
            completion(accumulator)
            return
        }
        let tail = Array(actions.dropFirst())
        if let link = head.linksTo?.trimmingCharacters(in: .whitespacesAndNewlines),
           !link.isEmpty {
            guard let dependency = linkResults[link], dependency.status == .ok else {
                let junoId = Self.junoId(for: head)
                let result = operationFailure(
                    head,
                    junoId: junoId,
                    message: "Linked action '\(link)' did not complete."
                )
                onProgress(result)
                runSerially(
                    actions: tail,
                    accumulator: accumulator + [result],
                    linkResults: linkResults,
                    onProgress: onProgress,
                    completion: completion
                )
                return
            }
        }
        run(head) { [weak self] result in
            var updatedLinks = linkResults
            if let link = head.linkId?.trimmingCharacters(in: .whitespacesAndNewlines),
               !link.isEmpty {
                updatedLinks[link] = result
            }
            onProgress(result)
            self?.runSerially(
                actions: tail,
                accumulator: accumulator + [result],
                linkResults: updatedLinks,
                onProgress: onProgress,
                completion: completion
            )
        }
    }

    /// Per-task watchdog. Caps the wall-clock cost of a single action so a
    /// hung sink (Notes Automation prompt left dangling, Reminders store
    /// blocked, AppleScript spinning on an iCloud account fetch) cannot
    /// stall the rest of the batch and leave the HUD frozen.
    ///
    /// Notes get a longer budget because the very first AppleScript call
    /// after a permission grant can take ~10 s while macOS warms Notes.app
    /// and seeds the Juno folder. Reminders / alarms talk to EventKit
    /// directly and finish much faster.
    private static func watchdogSeconds(for kind: JunoActionKind) -> TimeInterval {
        switch kind {
        case .note:     return 18.0
        case .reminder: return 8.0
        case .alarm:    return 8.0
        }
    }

    private static func junoId(for action: JunoActionRequest) -> String {
        if let existing = action.junoId?.trimmingCharacters(in: .whitespacesAndNewlines),
           !existing.isEmpty {
            return existing
        }
        return UUID().uuidString
    }

    private static func primaryIso(for action: JunoActionRequest) -> String? {
        action.schedule?.primaryIso ?? action.when?.iso
    }

    private func run(
        _ action: JunoActionRequest,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        let junoId = Self.junoId(for: action)
        os_log(
            "dispatch kind=%{public}@ body_len=%{public}d has_when=%{public}@",
            log: actionLog, type: .info,
            action.kind.rawValue, action.body.count,
            action.when != nil ? "yes" : "no"
        )
        // Once-only completion guard. A misbehaving sink that fires its
        // callback twice would otherwise re-enter ``runSerially`` with the
        // same tail and duplicate every remaining action; a sink that
        // never fires would stall the chain. The watchdog below covers
        // the never-fire case; this lock covers the double-fire case.
        var fired = false
        let fireLock = NSLock()
        let watchdog = DispatchWorkItem {
            fireLock.lock()
            let already = fired
            if !already { fired = true }
            fireLock.unlock()
            guard !already else { return }
            os_log(
                "watchdog kind=%{public}@ — synthesizing sinkError",
                log: actionLog, type: .error,
                action.kind.rawValue
            )
            let preview = String(action.body.prefix(80))
            let result = JunoActionResult(
                junoId: junoId,
                kind: action.kind,
                status: .sinkError,
                bodyPreview: preview,
                body: action.body,
                whenIso: Self.primaryIso(for: action),
                error: "Action timed out before \(action.kind.descriptor.displayName.lowercased()) sink replied."
            )
            DispatchQueue.main.async { completion(result) }
        }
        DispatchQueue.main.asyncAfter(
            deadline: .now() + Self.watchdogSeconds(for: action.kind),
            execute: watchdog
        )

        let wrapped: (JunoActionResult) -> Void = { result in
            fireLock.lock()
            let already = fired
            if !already { fired = true }
            fireLock.unlock()
            guard !already else {
                os_log(
                    "duplicate_completion kind=%{public}@ status=%{public}@ — dropped",
                    log: actionLog, type: .error,
                    result.kind.rawValue, result.status.rawValue
                )
                return
            }
            watchdog.cancel()
            os_log(
                "result kind=%{public}@ status=%{public}@ err=%{public}@",
                log: actionLog, type: .info,
                result.kind.rawValue, result.status.rawValue,
                result.error ?? ""
            )
            completion(result)
        }
        switch action.operation {
        case .create:
            runCreate(action, junoId: junoId, completion: wrapped)
        case .update:
            runUpdate(action, junoId: junoId, completion: wrapped)
        case .complete:
            runComplete(action, junoId: junoId, completion: wrapped)
        case .snooze:
            runSnooze(action, junoId: junoId, completion: wrapped)
        case .delete:
            runDelete(action, junoId: junoId, completion: wrapped)
        case .query:
            runQuery(action, junoId: junoId, completion: wrapped)
        case .appendTo, .removeFrom:
            runContainerOp(action, junoId: junoId, completion: wrapped)
        }
    }

    private func runCreate(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        switch action.kind {
        case .reminder:
            if action.container?.listName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
                runReminderListFanout(action, junoId: junoId, completion: completion)
            } else {
                runReminder(action, junoId: junoId, completion: completion)
            }
        case .note:
            runNote(action, junoId: junoId, completion: completion)
        case .alarm:
            runAlarm(action, junoId: junoId, completion: completion)
        }
    }

    private func runUpdate(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard let targetId = action.sinkId?.trimmingCharacters(in: .whitespacesAndNewlines),
              !targetId.isEmpty else {
            completion(operationFailure(action, junoId: junoId, message: "No target id for update."))
            return
        }
        let due = Self.primaryIso(for: action).flatMap { Self.parseISO8601($0) }
        switch action.kind {
        case .reminder:
            if due == nil,
               let offsetSeconds = action.relativeOffsetSeconds,
               offsetSeconds > 0 {
                reminders.snoozeReminder(id: targetId, by: TimeInterval(offsetSeconds)) { result in
                    Task { @MainActor in
                        completion(self.operationResult(action, junoId: junoId, sinkId: targetId, result: result))
                    }
                }
                return
            }
            let patch = JunoReminderSink.ReminderPatch(
                title: action.body.isEmpty ? nil : action.body,
                dueDate: due,
                recurrence: action.schedule?.series,
                listName: action.container?.listName
            )
            reminders.updateReminder(id: targetId, patch: patch) { result in
                Task { @MainActor in
                    completion(self.operationResult(action, junoId: junoId, sinkId: targetId, result: result))
                }
            }
        case .alarm:
            let patch = JunoAlarmSink.AlarmPatch(
                title: action.body.isEmpty ? nil : action.body,
                fireDate: due,
                recurrence: action.schedule?.series
            )
            alarms.updateAlarm(id: targetId, patch: patch) { result in
                Task { @MainActor in
                    completion(self.operationResult(action, junoId: junoId, sinkId: targetId, result: result))
                }
            }
        case .note:
            completion(operationFailure(action, junoId: junoId, message: "Updating notes is not supported yet."))
        }
    }

    private func runComplete(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard action.kind == .reminder else {
            completion(operationFailure(action, junoId: junoId, message: "Only reminders can be completed."))
            return
        }
        guard let targetId = action.sinkId?.trimmingCharacters(in: .whitespacesAndNewlines),
              !targetId.isEmpty else {
            completion(operationFailure(action, junoId: junoId, message: "No target id for completion."))
            return
        }
        reminders.completeReminder(id: targetId) { result in
            Task { @MainActor in
                switch result {
                case .success:
                    completion(self.operationSuccess(action, junoId: junoId, sinkId: targetId))
                case .failure(let err):
                    completion(self.operationFailure(action, junoId: junoId, sinkId: targetId, message: err.localizedDescription))
                }
            }
        }
    }

    private func runSnooze(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard action.kind == .reminder else {
            completion(operationFailure(action, junoId: junoId, message: "Only reminders can be snoozed."))
            return
        }
        guard let targetId = action.sinkId?.trimmingCharacters(in: .whitespacesAndNewlines),
              !targetId.isEmpty else {
            completion(operationFailure(action, junoId: junoId, message: "No target id for snooze."))
            return
        }
        let offset = TimeInterval(action.snoozeOffsetSeconds ?? action.relativeOffsetSeconds ?? 0)
        guard offset > 0 else {
            completion(operationFailure(action, junoId: junoId, sinkId: targetId, message: "No snooze offset."))
            return
        }
        reminders.snoozeReminder(id: targetId, by: offset) { result in
            Task { @MainActor in
                completion(self.operationResult(action, junoId: junoId, sinkId: targetId, result: result))
            }
        }
    }

    private func runDelete(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard let targetId = action.sinkId?.trimmingCharacters(in: .whitespacesAndNewlines),
              !targetId.isEmpty else {
            completion(operationFailure(action, junoId: junoId, message: "No target id for delete."))
            return
        }
        switch action.kind {
        case .reminder:
            reminders.deleteReminder(id: targetId) { result in
                Task { @MainActor in
                    switch result {
                    case .success:
                        completion(self.operationSuccess(action, junoId: junoId, sinkId: targetId))
                    case .failure(let err):
                        completion(self.operationFailure(action, junoId: junoId, sinkId: targetId, message: err.localizedDescription))
                    }
                }
            }
        case .alarm:
            alarms.deleteAlarm(id: targetId) { result in
                Task { @MainActor in
                    switch result {
                    case .success:
                        completion(self.operationSuccess(action, junoId: junoId, sinkId: targetId))
                    case .failure(let err):
                        completion(self.operationFailure(action, junoId: junoId, sinkId: targetId, message: err.localizedDescription))
                    }
                }
            }
        case .note:
            notes.deleteNote(id: targetId) { result in
                Task { @MainActor in
                    switch result {
                    case .success:
                        completion(self.operationSuccess(action, junoId: junoId, sinkId: targetId))
                    case .failure(let err):
                        completion(self.operationFailure(action, junoId: junoId, sinkId: targetId, message: err.localizedDescription))
                    }
                }
            }
        }
    }

    private func runQuery(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard action.kind == .reminder else {
            completion(operationFailure(action, junoId: junoId, message: "Query is only wired for reminders in this phase."))
            return
        }
        reminders.queryReminders(filter: JunoReminderSink.ReminderQueryFilter(dateRange: nil, text: action.body, listName: action.container?.listName)) { result in
            Task { @MainActor in
                switch result {
                case .success(let reminders):
                    completion(
                        JunoActionResult(
                            junoId: junoId,
                            operation: action.operation,
                            kind: action.kind,
                            status: .ok,
                            bodyPreview: "Found \(reminders.count) reminders",
                            body: action.body,
                            extras: ["count": "\(reminders.count)"]
                        )
                    )
                case .failure(let err):
                    completion(self.operationFailure(action, junoId: junoId, message: err.localizedDescription))
                }
            }
        }
    }

    private func runContainerOp(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard action.kind == .reminder else {
            completion(operationFailure(action, junoId: junoId, message: "List operations are only wired for reminders."))
            return
        }
        switch action.operation {
        case .appendTo:
            runReminderListFanout(action, junoId: junoId, completion: completion)
        case .removeFrom:
            reminders.removeReminder(matching: action.body, fromList: action.container?.listName) { result in
                Task { @MainActor in
                    switch result {
                    case .success:
                        completion(self.operationSuccess(action, junoId: junoId))
                    case .failure(let err):
                        completion(self.operationFailure(action, junoId: junoId, message: err.localizedDescription))
                    }
                }
            }
        default:
            completion(operationFailure(action, junoId: junoId, message: "Unsupported list operation."))
        }
    }

    private func runAlarm(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        let firstFireIso: String? = (action.schedule?.primaryIso) ?? action.when?.iso
        let due: Date? = firstFireIso.flatMap { Self.parseISO8601($0) }
        let recurrence: JunoRecurrenceRule? = action.schedule?.series
        let title = action.body.isEmpty ? "Alarm" : action.body
        let preview = String(title.prefix(80))

        switch alarms.authorization {
        case .granted:
            alarms.createAlarm(title: title, at: due, recurrence: recurrence) { sinkResult in
                Task { @MainActor in
                    switch sinkResult {
                    case .success(let created):
                        completion(
                            JunoActionResult(
                                junoId: junoId,
                                kind: .alarm,
                                status: .ok,
                                bodyPreview: preview,
                                body: title,
                                sinkId: created.id,
                                sinkUrl: created.url?.absoluteString,
                                whenIso: firstFireIso
                            )
                        )
                    case .failure(let err):
                        let status: JunoActionStatus = {
                            if case .missingTime = err { return .timeParseFailed }
                            if case .permissionDenied = err { return .permissionDenied }
                            return .sinkError
                        }()
                        completion(
                            JunoActionResult(
                                junoId: junoId,
                                kind: .alarm,
                                status: status,
                                bodyPreview: preview,
                                body: title,
                                whenIso: firstFireIso,
                                error: err.localizedDescription
                            )
                        )
                    }
                }
            }
        case .denied, .restricted:
            completion(
                JunoActionResult(
                    junoId: junoId,
                    kind: .alarm,
                    status: .permissionDenied,
                    bodyPreview: preview,
                    body: title,
                    whenIso: firstFireIso,
                    error: "Calendar access is off."
                )
            )
        case .notDetermined:
            completion(
                JunoActionResult(
                    junoId: junoId,
                    kind: .alarm,
                    status: .permissionDenied,
                    bodyPreview: preview,
                    body: title,
                    whenIso: firstFireIso,
                    error: "Calendar access has not been requested yet.",
                    extras: ["needs_initial_prompt": "true"]
                )
            )
        }
    }

    private func runReminder(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        // Phase 1: prefer schedule.instant or schedule.series anchor
        // for the first-occurrence due date; fall back to legacy
        // ``when`` for v2 envelopes.
        let firstFireIso: String? = (action.schedule?.primaryIso) ?? action.when?.iso
        let due: Date? = firstFireIso.flatMap { Self.parseISO8601($0) }
        let recurrence: JunoRecurrenceRule? = action.schedule?.series
        let title = action.body
        let preview = String(title.prefix(80))

        switch reminders.authorization {
        case .granted:
            reminders.createReminder(
                title: title,
                notes: nil,
                dueDate: due,
                recurrence: recurrence,
                listName: action.container?.listName
            ) { sinkResult in
                Task { @MainActor in
                    switch sinkResult {
                    case .success(let created):
                        completion(
                            JunoActionResult(
                                junoId: junoId,
                                kind: .reminder,
                                status: .ok,
                                bodyPreview: preview,
                                body: title,
                                sinkId: created.id,
                                sinkUrl: created.url?.absoluteString,
                                whenIso: firstFireIso
                            )
                        )
                    case .failure(let err):
                        completion(
                            JunoActionResult(
                                junoId: junoId,
                                kind: .reminder,
                                status: .sinkError,
                                bodyPreview: preview,
                                body: title,
                                whenIso: firstFireIso,
                                error: err.localizedDescription
                            )
                        )
                    }
                }
            }
        case .denied, .restricted:
            completion(
                JunoActionResult(
                    junoId: junoId,
                    kind: .reminder,
                    status: .permissionDenied,
                    bodyPreview: preview,
                    body: title,
                    whenIso: firstFireIso,
                    error: "Reminders access is off."
                )
            )
        case .notDetermined:
            // We do **not** auto-prompt here. The Home-page nudge owns
            // the first ask so the user sees the value prop before the
            // system dialog. Record as permissionDenied with a
            // distinguishing extras flag the nudge can read.
            completion(
                JunoActionResult(
                    junoId: junoId,
                    kind: .reminder,
                    status: .permissionDenied,
                    bodyPreview: preview,
                    body: title,
                    whenIso: firstFireIso,
                    error: "Reminders access has not been requested yet.",
                    extras: ["needs_initial_prompt": "true"]
                )
            )
        }
    }

    private func runReminderListFanout(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard let listName = action.container?.listName?.trimmingCharacters(in: .whitespacesAndNewlines),
              !listName.isEmpty else {
            runReminder(action, junoId: junoId, completion: completion)
            return
        }
        let items = Self.reminderListItems(from: action.body)
        if items.isEmpty {
            reminders.ensureReminderList(named: listName) { result in
                Task { @MainActor in
                    switch result {
                    case .success(let listId):
                        completion(
                            JunoActionResult(
                                junoId: junoId,
                                operation: action.operation,
                                kind: .reminder,
                                status: .ok,
                                bodyPreview: "Created list \(listName)",
                                body: action.body,
                                sinkId: listId,
                                extras: ["list_name": listName, "item_count": "0"]
                            )
                        )
                    case .failure(let err):
                        completion(self.operationFailure(action, junoId: junoId, message: err.localizedDescription))
                    }
                }
            }
            return
        }

        switch reminders.authorization {
        case .granted:
            createReminderItems(
                items,
                action: action,
                junoId: junoId,
                listName: listName,
                created: [],
                failures: [],
                completion: completion
            )
        case .denied, .restricted:
            completion(
                JunoActionResult(
                    junoId: junoId,
                    operation: action.operation,
                    kind: .reminder,
                    status: .permissionDenied,
                    bodyPreview: String(action.body.prefix(80)),
                    body: action.body,
                    whenIso: Self.primaryIso(for: action),
                    error: "Reminders access is off."
                )
            )
        case .notDetermined:
            completion(
                JunoActionResult(
                    junoId: junoId,
                    operation: action.operation,
                    kind: .reminder,
                    status: .permissionDenied,
                    bodyPreview: String(action.body.prefix(80)),
                    body: action.body,
                    whenIso: Self.primaryIso(for: action),
                    error: "Reminders access has not been requested yet.",
                    extras: ["needs_initial_prompt": "true", "list_name": listName]
                )
            )
        }
    }

    private func createReminderItems(
        _ remaining: [String],
        action: JunoActionRequest,
        junoId: String,
        listName: String,
        created: [JunoReminderSink.CreatedReminder],
        failures: [String],
        completion: @escaping (JunoActionResult) -> Void
    ) {
        guard let item = remaining.first else {
            let count = created.count
            let failed = failures.count
            let status: JunoActionStatus = count > 0 ? .ok : .sinkError
            var extras: [String: String] = [
                "list_name": listName,
                "item_count": "\(count)",
            ]
            if failed > 0 {
                extras["failed_count"] = "\(failed)"
            }
            completion(
                JunoActionResult(
                    junoId: junoId,
                    operation: action.operation,
                    kind: .reminder,
                    status: status,
                    bodyPreview: count == 1 ? String(action.body.prefix(80)) : "Saved \(count) items to \(listName)",
                    body: action.body,
                    sinkId: created.first?.id,
                    sinkUrl: created.first?.url?.absoluteString,
                    whenIso: Self.primaryIso(for: action),
                    error: count > 0 ? nil : failures.first,
                    extras: extras
                )
            )
            return
        }

        let firstFireIso: String? = (action.schedule?.primaryIso) ?? action.when?.iso
        let due: Date? = firstFireIso.flatMap { Self.parseISO8601($0) }
        reminders.createReminder(
            title: item,
            notes: nil,
            dueDate: due,
            recurrence: action.schedule?.series,
            listName: listName
        ) { result in
            Task { @MainActor in
                var nextCreated = created
                var nextFailures = failures
                switch result {
                case .success(let row):
                    nextCreated.append(row)
                case .failure(let err):
                    nextFailures.append(err.localizedDescription)
                }
                self.createReminderItems(
                    Array(remaining.dropFirst()),
                    action: action,
                    junoId: junoId,
                    listName: listName,
                    created: nextCreated,
                    failures: nextFailures,
                    completion: completion
                )
            }
        }
    }

    private static func reminderListItems(from body: String) -> [String] {
        let cleaned = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return [] }
        let commaParts = cleaned
            .split(whereSeparator: { ",;\n".contains($0) })
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let firstPass = commaParts.count > 1 ? commaParts : [cleaned]
        return firstPass.flatMap { part -> [String] in
            let split = part
                .components(separatedBy: " and ")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            return split.isEmpty ? [part] : split
        }
    }

    private func runNote(
        _ action: JunoActionRequest,
        junoId: String,
        completion: @escaping (JunoActionResult) -> Void
    ) {
        let body = action.body
        let preview = String(body.prefix(80))
        let appendSig = JunoUserDefaults.actionsNotesSignatureEnabled

        notes.createNote(
            body: body,
            appendSignature: appendSig,
            folderName: action.container?.folderName
        ) { sinkResult in
            Task { @MainActor in
                switch sinkResult {
                case .success(let created):
                    completion(
                        JunoActionResult(
                            junoId: junoId,
                            kind: .note,
                            status: .ok,
                            bodyPreview: preview,
                            body: body,
                            sinkId: created.id,
                            sinkUrl: created.url?.absoluteString
                        )
                    )
                case .failure(let err):
                    let status: JunoActionStatus
                    if case .automationDenied = err {
                        status = .permissionDenied
                        JunoActionPermissionStore.shared.revalidateNotesAutomation()
                    } else {
                        status = .sinkError
                    }
                    completion(
                        JunoActionResult(
                            junoId: junoId,
                            kind: .note,
                            status: status,
                            bodyPreview: preview,
                            body: body,
                            error: err.localizedDescription
                        )
                    )
                }
            }
        }
    }

    // MARK: - Helpers

    private func operationSuccess(
        _ action: JunoActionRequest,
        junoId: String,
        sinkId: String? = nil,
        sinkUrl: String? = nil
    ) -> JunoActionResult {
        JunoActionResult(
            junoId: junoId,
            operation: action.operation,
            kind: action.kind,
            status: .ok,
            bodyPreview: String(action.body.prefix(80)),
            body: action.body,
            sinkId: sinkId ?? action.sinkId,
            sinkUrl: sinkUrl,
            whenIso: Self.primaryIso(for: action)
        )
    }

    private func operationFailure(
        _ action: JunoActionRequest,
        junoId: String,
        sinkId: String? = nil,
        message: String
    ) -> JunoActionResult {
        JunoActionResult(
            junoId: junoId,
            operation: action.operation,
            kind: action.kind,
            status: .sinkError,
            bodyPreview: String(action.body.prefix(80)),
            body: action.body,
            sinkId: sinkId ?? action.sinkId,
            whenIso: Self.primaryIso(for: action),
            error: message
        )
    }

    private func operationResult(
        _ action: JunoActionRequest,
        junoId: String,
        sinkId: String,
        result: Result<JunoReminderSink.CreatedReminder, JunoReminderSink.SinkError>
    ) -> JunoActionResult {
        switch result {
        case .success(let updated):
            return operationSuccess(
                action,
                junoId: junoId,
                sinkId: updated.id,
                sinkUrl: updated.url?.absoluteString
            )
        case .failure(let err):
            return operationFailure(action, junoId: junoId, sinkId: sinkId, message: err.localizedDescription)
        }
    }

    private func operationResult(
        _ action: JunoActionRequest,
        junoId: String,
        sinkId: String,
        result: Result<JunoAlarmSink.CreatedAlarm, JunoAlarmSink.SinkError>
    ) -> JunoActionResult {
        switch result {
        case .success(let updated):
            return operationSuccess(
                action,
                junoId: junoId,
                sinkId: updated.id,
                sinkUrl: updated.url?.absoluteString
            )
        case .failure(let err):
            return operationFailure(action, junoId: junoId, sinkId: sinkId, message: err.localizedDescription)
        }
    }

    private static func parseISO8601(_ iso: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: iso) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: iso)
    }

    /// Default network impl with bounded retry.
    ///
    /// History is the user's source of truth ("did my note save?"). When
    /// this post drops on the floor, the row stays on the parsed-only
    /// shape and the History UI shows "Saving…" forever — that's the
    /// "Not yet dispatched" complaint we kept hearing. So:
    ///
    /// * Read the broker's ``ok`` flag; on failure (network blip, broker
    ///   restart, bad JSON), retry with backoff (~1s, 3s, 8s).
    /// * After retries are exhausted, log a warning and stash the payload
    ///   to disk so the next app launch can drain it. The server-side
    ///   ``update_actions`` upserts a stub row when the row doesn't yet
    ///   exist, so this post is safe to send before the pipeline finishes
    ///   writing the history row.
    nonisolated static func postResultsToBroker(
        utteranceId: String,
        results: [JunoActionResult]
    ) {
        postResultsWithRetry(
            utteranceId: utteranceId,
            results: results,
            retryIndex: 0,
            enqueueOnFailure: true
        )
    }

    nonisolated fileprivate static func postBackloggedResults(
        utteranceId: String,
        results: [JunoActionResult],
        completion: @escaping (Bool) -> Void
    ) {
        postResultsWithRetry(
            utteranceId: utteranceId,
            results: results,
            retryIndex: 0,
            enqueueOnFailure: false,
            completion: completion
        )
    }

    nonisolated private static let pendingPostRetryDelays: [TimeInterval] = [1.0, 3.0, 8.0]

    nonisolated private static func postResultsWithRetry(
        utteranceId: String,
        results: [JunoActionResult],
        retryIndex: Int,
        enqueueOnFailure: Bool,
        completion: ((Bool) -> Void)? = nil
    ) {
        let body: [String: Any] = ["actions": results.map { $0.toWireDict() }]
        JunoBroker.postJSON(
            path: "api/broker/history/\(utteranceId)/actions",
            payload: body
        ) { resp in
            let ok = (resp["ok"] as? Bool) ?? false
            if ok {
                completion?(true)
                return
            }
            let err = (resp["error"] as? String) ?? "unknown"
            if retryIndex >= pendingPostRetryDelays.count {
                os_log(
                    "post_actions_failed uid=%{public}@ err=%{public}@ — giving up",
                    log: actionLog, type: .error,
                    utteranceId, err
                )
                if enqueueOnFailure {
                    JunoActionPostBacklog.shared.enqueue(
                        utteranceId: utteranceId,
                        results: results
                    )
                }
                completion?(false)
                return
            }
            let delay = pendingPostRetryDelays[retryIndex]
            os_log(
                "post_actions_retry uid=%{public}@ err=%{public}@ in=%{public}.1fs retry=%{public}d",
                log: actionLog, type: .info,
                utteranceId, err, delay, retryIndex + 1
            )
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) {
                postResultsWithRetry(
                    utteranceId: utteranceId,
                    results: results,
                    retryIndex: retryIndex + 1,
                    enqueueOnFailure: enqueueOnFailure,
                    completion: completion
                )
            }
        }
    }
}

// MARK: - Backlog
//
// Disk-backed FIFO of action result posts that exhausted their in-memory
// retries. We drain the backlog on every app launch so a broker restart in
// the middle of a dictation doesn't leave History stuck on "Saving…".
//
// Failures are stored as a single JSON file under Application Support so
// the data survives restarts; the file is rewritten atomically and
// truncated to the most recent ~50 entries.

private struct JunoActionPostBacklogEntry: Codable {
    let utteranceId: String
    let results: [JunoActionResult]
    let queuedAt: Date
}

final class JunoActionPostBacklog {
    static let shared = JunoActionPostBacklog()

    private let queue = DispatchQueue(label: "juno.actions.backlog")
    private let storeURL: URL? = {
        guard let root = JunoSupportPaths.supportRoot() else { return nil }
        let dir = root.appendingPathComponent("ActionsBacklog", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            os_log("Action backlog directory create failed: %{public}@", log: actionLog, type: .error, error.localizedDescription)
            return nil
        }
        return dir.appendingPathComponent("post_results.json")
    }()
    private let cap = 50

    func enqueue(utteranceId: String, results: [JunoActionResult]) {
        queue.async {
            var entries = self.loadLocked()
            entries.append(JunoActionPostBacklogEntry(
                utteranceId: utteranceId,
                results: results,
                queuedAt: Date()
            ))
            if entries.count > self.cap {
                entries = Array(entries.suffix(self.cap))
            }
            self.saveLocked(entries)
        }
    }

    /// Drain the backlog. Each entry gets one fresh retry chain. Entries
    /// that succeed are removed; entries that fail again stay on disk for
    /// the next attempt.
    func drain() {
        queue.async {
            let entries = self.loadLocked()
            guard !entries.isEmpty else { return }
            for entry in entries {
                JunoActionExecutor.postBackloggedResults(
                    utteranceId: entry.utteranceId,
                    results: entry.results
                ) { ok in
                    if ok {
                        self.remove(entry)
                    }
                }
            }
        }
    }

    private func remove(_ entry: JunoActionPostBacklogEntry) {
        queue.async {
            let entries = self.loadLocked().filter {
                !($0.utteranceId == entry.utteranceId && $0.queuedAt == entry.queuedAt)
            }
            self.saveLocked(entries)
        }
    }

    private func loadLocked() -> [JunoActionPostBacklogEntry] {
        guard let url = storeURL else { return [] }
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([JunoActionPostBacklogEntry].self, from: data)
        } catch {
            os_log("Action backlog load failed: %{public}@", log: actionLog, type: .error, error.localizedDescription)
            return []
        }
    }

    private func saveLocked(_ entries: [JunoActionPostBacklogEntry]) {
        guard let url = storeURL else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        do {
            let data = try encoder.encode(entries)
            try data.write(to: url, options: .atomic)
        } catch {
            os_log("Action backlog save failed: %{public}@", log: actionLog, type: .error, error.localizedDescription)
        }
    }
}
