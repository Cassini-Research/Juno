import AppKit
import Darwin
import Foundation

/// Single source of truth for the engine's launch lifecycle. The existing
/// ``JunoLocalBrokerBootstrap`` + ``JunoEngineSupervisor`` + ``JunoSetupModel``
/// chain still owns spawning, respawn, and setup polling — this class
/// observes them, drives an ordered fast-poll for the early phases, and
/// publishes a single ``phase`` value the splash and home gate read from.
///
/// Commit 1 keeps the existing flow intact; later commits will fold the
/// supervisor's public state and the setup-model timer into this class.
@MainActor
final class JunoEngineLifecycle: ObservableObject {
    enum Phase: Equatable {
        case idle
        /// Bundled engine helper is being launched (or we are waiting for an
        /// already-running engine to come back online after a respawn).
        case spawning(attempt: Int)
        /// The UDS socket file exists on disk — engine is alive enough to
        /// bind, but ``/healthz`` has not yet returned ok.
        case socketBound
        /// ``/healthz`` returned ok at least once; models may still be loading.
        case healthOk
        /// Setup status reports ``overallReady = true``: every model the
        /// active deployment requires is present and the engine has loaded
        /// what it needs to load. Writer prewarm is fire-and-forget after
        /// this point.
        case modelsLoaded
        /// Final terminal-success phase. Writer prewarm has been kicked off
        /// (or wasn't required) and the splash is safe to dismiss.
        case ready
        /// Engine is healthy but the user must download or repair models.
        case needsModels
        /// Engine is healthy and models are ready, but TCC permissions are
        /// missing so dictation can't start.
        case needsPermissions
        /// We were ready and lost the engine — auto-recover in progress.
        case degraded(reason: DegradeReason, since: Date)
        /// Unrecoverable without explicit user action (View logs / Retry / Reset).
        case failed(FailureKind)

        var isTerminalSuccess: Bool {
            switch self {
            case .ready, .needsModels, .needsPermissions: return true
            default: return false
            }
        }

        var isFailure: Bool {
            if case .failed = self { return true }
            return false
        }

        /// True whenever the engine has been observed responding — any phase
        /// at or past ``.socketBound``. Used by the home view to decide
        /// whether to fetch broker-backed data (stats, recent dictations,
        /// editing profile) rather than falling back to local-only mode.
        /// ``.degraded`` stays reachable because we *had* a healthy engine
        /// and the supervisor is actively reconnecting.
        var isReachable: Bool {
            switch self {
            case .socketBound, .healthOk, .modelsLoaded, .ready,
                 .needsModels, .needsPermissions, .degraded:
                return true
            case .idle, .spawning, .failed:
                return false
            }
        }

        var label: String {
            switch self {
            case .idle: return "idle"
            case .spawning(let a): return "spawning(\(a))"
            case .socketBound: return "socketBound"
            case .healthOk: return "healthOk"
            case .modelsLoaded: return "modelsLoaded"
            case .ready: return "ready"
            case .needsModels: return "needsModels"
            case .needsPermissions: return "needsPermissions"
            case .degraded(let r, _): return "degraded(\(r))"
            case .failed(let k): return "failed(\(k))"
            }
        }
    }

    enum DegradeReason: String, Equatable {
        case engineExited
        case socketGone
        case healthTimeout
        case wakeFromSleep
        case appForeground
        case appReopen
    }

    enum FailureKind: Equatable, CustomStringConvertible {
        case bundleMissing(path: String)
        case execError(String)
        case socketStuck
        case diskFull
        case oomCrashLoop(attempts: Int)
        case unknown(String)

        var description: String {
            switch self {
            case .bundleMissing(let p): return "bundleMissing:\(p)"
            case .execError(let s): return "execError:\(s)"
            case .socketStuck: return "socketStuck"
            case .diskFull: return "diskFull"
            case .oomCrashLoop(let n): return "oomCrashLoop:\(n)"
            case .unknown(let s): return "unknown:\(s)"
            }
        }

        var userMessage: String {
            switch self {
            case .bundleMissing: return "The bundled voice engine is missing. Reinstall Juno to recover."
            case .execError(let detail):
                // Keep the splash subhead readable: short summary + advise
                // viewing logs for the full traceback. The full tail is on
                // the lifecycle's lastCrashLogTail and surfaces in the
                // diagnostics window.
                if detail.count > 120 {
                    return "Voice engine crashed during startup. View logs for the full traceback."
                }
                return "Voice engine crashed: \(detail)"
            case .socketStuck: return "Couldn't claim the engine socket. Try Reset to clear stale state."
            case .diskFull: return "Not enough disk space to start the voice engine."
            case .oomCrashLoop: return "The voice engine kept running out of memory. Free up RAM or switch to a cloud profile."
            case .unknown(let s): return s
            }
        }
    }

    // MARK: Singleton

    static let shared = JunoEngineLifecycle()

    // MARK: Published state

    @Published private(set) var phase: Phase = .idle
    /// Last status message rendered under the splash headline. Distinct from
    /// ``phase.label`` because failure detail messages and download progress
    /// are interesting to the user but should not change the phase enum.
    @Published private(set) var statusDetail: String = ""
    /// Determinate progress fraction the splash shows. Only set during the
    /// happy path; any failure surface uses its own UI.
    @Published private(set) var progress: Double = 0.0
    /// True for the brief window after ``boot()`` returns success — lets the
    /// AppDelegate skip duplicated post-launch warmups when commit 2 lands.
    @Published private(set) var didReachReadyOnce: Bool = false

    // MARK: Private state

    private var booted = false
    private var spawnAttempt = 0
    private var phaseEnteredAt: Date = .init()
    private var observers: [NSObjectProtocol] = []
    private var oomEvents: [Date] = []
    /// Non-OOM crash exits within the last 60 s. Three strikes during the
    /// pre-health window flip us to ``.failed(.execError)`` so the user sees
    /// the last traceback instead of the splash spinning forever while the
    /// supervisor respawns into the same crash.
    private var crashEvents: [Date] = []
    private var lastSetupSnapshot: BrokerSetupStatusResponse?
    /// Most recent crash-log tail captured at the moment we transitioned to
    /// ``.failed``. Surfaced by ``JunoDiagnosticsView`` in an expander so
    /// users can see the actual Python traceback without leaving the app.
    private(set) var lastCrashLogTail: String?

    private let socketProbeBudget: TimeInterval = 30
    private let healthProbeBudget: TimeInterval = 45
    private let setupProbeBudget: TimeInterval = 60

    private init() {}

    // MARK: Public API

    func boot() {
        guard !booted else { return }
        booted = true
        installNotificationObservers()

        Task { @MainActor in
            await runBoot()
        }
    }

    /// Cheap re-probe used on Dock-reopen and wake-from-sleep. If we were
    /// already ``.ready`` and the ping comes back ok, no-op. Otherwise
    /// transition to ``.degraded`` and let the boot ladder re-enter.
    func reprobe(reason: DegradeReason) {
        Task { @MainActor in
            // Only re-probe terminal-success phases; an in-flight boot is
            // already polling.
            guard phase.isTerminalSuccess else { return }
            let ok = await pingHealthOnce(timeoutMs: 250)
            if ok { return }
            transition(to: .degraded(reason: reason, since: Date()), note: "reprobe failed")
            // Hand recovery to the existing supervisor (already running) and
            // re-enter our own poll ladder so the splash banner gets useful
            // phase updates.
            JunoLocalBrokerBootstrap.spawnIfNotRunning()
            await pollUntilHealthy(initialPhase: .spawning(attempt: spawnAttempt + 1))
        }
    }

    /// User pressed "Retry" on the failure splash. Resets failure counters
    /// and re-enters the spawn ladder.
    func retry() {
        Task { @MainActor in
            guard case .failed = phase else { return }
            spawnAttempt = 0
            oomEvents.removeAll()
            transition(to: .spawning(attempt: 1), note: "user retry")
            JunoLocalBrokerBootstrap.spawnIfNotRunning()
            await pollUntilHealthy(initialPhase: .spawning(attempt: 1))
        }
    }

    /// Hard reset: kill any tracked engine process, unlink the UDS socket if
    /// stale, then re-enter spawn. Used to recover from "socket stuck"
    /// scenarios where a half-dead peer holds the path.
    func reset() {
        Task { @MainActor in
            if let proc = JunoShellRuntime.shared.brokerProcess, proc.isRunning {
                JunoProcessTreeTerminator.terminate(
                    process: proc,
                    reason: "lifecycle_reset",
                    graceSeconds: 2.0
                )
            }
            JunoShellRuntime.shared.brokerProcess = nil
            let socketPath = JunoBroker.engineSocketPath
            if FileManager.default.fileExists(atPath: socketPath) {
                try? FileManager.default.removeItem(atPath: socketPath)
            }
            spawnAttempt = 0
            oomEvents.removeAll()
            transition(to: .spawning(attempt: 1), note: "user reset")
            JunoLocalBrokerBootstrap.spawnIfNotRunning()
            await pollUntilHealthy(initialPhase: .spawning(attempt: 1))
        }
    }

    // MARK: Boot pipeline

    private func runBoot() async {
        // 1. Preflight — fail fast on bundle/disk problems before we ever
        //    spawn Python so the user gets an actionable error, not a silent
        //    timeout.
        if let failure = preflight() {
            transition(to: .failed(failure), note: "preflight")
            return
        }

        // 2. Kick the existing bootstrap. It is idempotent and may attach to
        //    an already-running peer instead of spawning, so we don't gate on
        //    its synchronous return.
        spawnAttempt = 1
        transition(to: .spawning(attempt: spawnAttempt), note: "ensureRunningIfPossible")
        JunoLocalBrokerBootstrap.ensureRunningIfPossible()

        await pollUntilHealthy(initialPhase: .spawning(attempt: spawnAttempt))
        guard case .healthOk = phase else { return }

        // 3. Once healthz is green, drive setup-status until we know whether
        //    the user is ready or needs to install models.
        await waitForSetup()
    }

    private func pollUntilHealthy(initialPhase: Phase) async {
        // Phase-A: socket file appears on disk. The supervisor and bootstrap
        // own the actual spawn; we just watch for the bind so the splash can
        // surface meaningful progress instead of a forever-spinner.
        if !(phase.isTerminalSuccess || phase.isFailure) {
            transition(to: initialPhase, note: "pollUntilHealthy")
        }
        let socketBound = await pollSocketBound(budget: socketProbeBudget)
        guard socketBound else {
            transition(to: .failed(.execError("socket never bound after \(Int(socketProbeBudget))s")),
                       note: "socket timeout")
            return
        }
        if !phase.isTerminalSuccess {
            transition(to: .socketBound, note: "socket on disk")
        }

        // Phase-B: /healthz returns ok. This covers the long MLX/Whisper
        // import. The supervisor's own ping loop also runs in parallel;
        // racing it is fine because both call the same endpoint.
        let healthOk = await pollHealthz(budget: healthProbeBudget)
        guard healthOk else {
            transition(to: .degraded(reason: .healthTimeout, since: Date()),
                       note: "/healthz timeout — supervisor will keep retrying")
            return
        }
        if !phase.isTerminalSuccess {
            transition(to: .healthOk, note: "/healthz ok")
        }
    }

    // MARK: Polling helpers

    private func pollSocketBound(budget: TimeInterval) async -> Bool {
        // Unix-domain socket only — the TCP fallback path doesn't put a file
        // on disk, so for that branch we just treat .spawning → .socketBound
        // as instant.
        guard JunoBroker.shouldUseUDS else { return true }

        let socketPath = JunoBroker.engineSocketPath
        let start = Date()
        var step: UInt32 = 250_000 // 250 ms
        while Date().timeIntervalSince(start) < budget {
            if FileManager.default.fileExists(atPath: socketPath) {
                return true
            }
            try? await Task.sleep(nanoseconds: UInt64(step) * 1_000)
            // 250ms × 6 → 500ms × 6 → 1s thereafter
            if step < 1_000_000 {
                step = min(1_000_000, step + 250_000)
            }
        }
        return false
    }

    private func pollHealthz(budget: TimeInterval) async -> Bool {
        let start = Date()
        var step: UInt32 = 250_000
        while Date().timeIntervalSince(start) < budget {
            if await pingHealthOnce(timeoutMs: 600) {
                return true
            }
            try? await Task.sleep(nanoseconds: UInt64(step) * 1_000)
            if step < 1_000_000 {
                step = min(1_000_000, step + 250_000)
            }
        }
        return false
    }

    private func pingHealthOnce(timeoutMs: Int) async -> Bool {
        await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            JunoBroker.callBrokerRPC(
                httpMethod: "GET",
                path: "healthz",
                timeoutSeconds: max(0.1, TimeInterval(timeoutMs) / 1000.0)
            ) { result in
                switch result {
                case .success(let out):
                    let ok = (out.object["ok"] as? Bool) ?? true
                    cont.resume(returning: ok)
                case .failure:
                    cont.resume(returning: false)
                }
            }
        }
    }

    // MARK: Setup-status wait

    private func waitForSetup() async {
        let start = Date()
        while Date().timeIntervalSince(start) < setupProbeBudget {
            let snapshot = await fetchSetupStatusOnce()
            if let snapshot {
                lastSetupSnapshot = snapshot
                applySetupSnapshot(snapshot)
            }
            switch phase {
            case .ready, .needsModels, .needsPermissions, .failed:
                return
            default:
                try? await Task.sleep(nanoseconds: 800_000_000)
            }
        }
        // Setup timed out but health is fine. Fall through to ready so the
        // user can use the app; the home view's existing setup gate will
        // surface any required action.
        if !phase.isTerminalSuccess {
            transition(to: .ready, note: "setup probe timed out — proceeding optimistically")
            promoteToReadyIfNeeded()
        }
    }

    private func fetchSetupStatusOnce() async -> BrokerSetupStatusResponse? {
        await withCheckedContinuation { (cont: CheckedContinuation<BrokerSetupStatusResponse?, Never>) in
            JunoBroker.fetchSetupStatus { result in
                switch result {
                case .success(let s): cont.resume(returning: s)
                case .failure: cont.resume(returning: nil)
                }
            }
        }
    }

    private func applySetupSnapshot(_ s: BrokerSetupStatusResponse) {
        // Don't override an in-flight pre-health phase or a hard failure.
        switch phase {
        case .idle, .spawning, .socketBound, .failed:
            return
        default:
            break
        }
        if !JunoPermissionMonitor.shared.canDictate {
            transition(to: .needsPermissions, note: "tcc missing")
            return
        }
        let install = s.installState ?? "unknown"
        if install.hasPrefix("failed") {
            transition(to: .failed(.unknown(s.error ?? install)), note: "install state failed")
            return
        }
        if install == "downloading" {
            statusDetail = "Downloading models…"
            progress = max(progress, 0.7)
            return
        }
        if install == "not_started" || install == "needs_setup" {
            transition(to: .needsModels, note: install)
            return
        }
        if s.overallReady == true {
            // Allow forward progression out of `.needsModels` and
            // `.needsPermissions` when setup completes out-of-band. The
            // existing `isTerminalSuccess` check treats those as final,
            // which froze the home gate when install finished after the
            // initial probe had already classified us as needing models.
            switch phase {
            case .needsModels, .needsPermissions:
                transition(to: .modelsLoaded, note: "setup advanced; overallReady=true")
                promoteToReadyIfNeeded()
                return
            default:
                break
            }
            if !phase.isTerminalSuccess {
                transition(to: .modelsLoaded, note: "overallReady=true")
            }
            promoteToReadyIfNeeded()
        }
    }

    private func promoteToReadyIfNeeded() {
        guard case .modelsLoaded = phase else {
            // Already promoted, or we skipped straight to .ready (setup
            // timeout fallback).
            if case .ready = phase { didReachReadyOnce = true }
            return
        }
        transition(to: .ready, note: "models loaded")
        didReachReadyOnce = true
    }

    // MARK: Notification observers

    private func installNotificationObservers() {
        let nc = NotificationCenter.default

        observers.append(nc.addObserver(
            forName: .junoBrokerBootstrapFailed, object: nil, queue: .main
        ) { [weak self] note in
            Task { @MainActor in
                self?.handleBootstrapFailure(reason: note.object as? String ?? "unknown")
            }
        })

        observers.append(nc.addObserver(
            forName: .junoEngineSupervisorStateChanged, object: nil, queue: .main
        ) { [weak self] note in
            Task { @MainActor in
                if let state = note.object as? JunoEngineSupervisor.State {
                    self?.handleSupervisorState(state)
                }
            }
        })

        // Re-apply setup snapshots after waitForSetup() has exited so the
        // phase doesn't freeze at .needsModels when install completes
        // out-of-band. applySetupSnapshot() already short-circuits in
        // pre-health and terminal-failure phases, so this is safe to
        // observe continuously.
        observers.append(nc.addObserver(
            forName: .junoSetupSnapshotUpdated, object: nil, queue: .main
        ) { [weak self] note in
            Task { @MainActor in
                guard let self else { return }
                guard let s = note.object as? BrokerSetupStatusResponse else { return }
                self.lastSetupSnapshot = s
                self.applySetupSnapshot(s)
            }
        })

        // canDictate flip — re-apply the most recent setup snapshot so
        // .needsPermissions can promote out once perms land.
        observers.append(nc.addObserver(
            forName: .junoPermissionsCanDictateChanged, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if let s = self.lastSetupSnapshot {
                    self.applySetupSnapshot(s)
                }
            }
        })

        let workspace = NSWorkspace.shared.notificationCenter
        observers.append(workspace.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.reprobe(reason: .wakeFromSleep)
            }
        })

        observers.append(nc.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.reprobe(reason: .appForeground)
            }
        })
    }

    private func handleBootstrapFailure(reason: String) {
        // Map the well-known reasons emitted by spawnBundledEngine to typed
        // failures the splash can render with proper recovery actions.
        let kind: FailureKind
        switch reason {
        case "missing_run_engine_script", "no_engine_root":
            kind = .bundleMissing(path: JunoEngineContract.bundledEngineRoot()?.path ?? "<unknown>")
        case "missing_live_script":
            kind = .execError("Repo dev launcher script not found")
        default:
            kind = .execError(reason)
        }
        // Only flip to .failed if we haven't reached health yet. If we were
        // ready and a respawn briefly fails, the supervisor will recover —
        // surface as degraded instead of erroring out.
        switch phase {
        case .idle, .spawning, .socketBound:
            transition(to: .failed(kind), note: "bootstrap failed")
        default:
            transition(to: .degraded(reason: .engineExited, since: Date()), note: "respawn failed: \(reason)")
        }
    }

    private func handleSupervisorState(_ state: JunoEngineSupervisor.State) {
        switch state {
        case .online:
            // If we drifted into degraded after being ready, the supervisor
            // recovering means we can leave the banner.
            if case .degraded = phase {
                transition(to: .ready, note: "supervisor online")
                didReachReadyOnce = true
            }
        case .offline(let reason):
            // Track OOM crashes for the failure-mode classifier. Status code
            // is encoded into the reason string by recordEngineExit:
            //   engine_exit:signal:9 → SIGKILL, treat as OOM-like
            if reason.contains(":signal:9") {
                oomEvents.append(Date())
                let recent = oomEvents.filter { Date().timeIntervalSince($0) < 60 }
                oomEvents = recent
                if recent.count >= 3 {
                    captureCrashLogTail()
                    transition(to: .failed(.oomCrashLoop(attempts: recent.count)),
                               note: "OOM crash loop")
                    return
                }
            }
            // Pre-health failures are real; post-health failures are a
            // transient blip the supervisor will retry. During the pre-health
            // window, three crash exits within 60s flip us to .failed so the
            // user can see the actual traceback instead of watching the
            // splash spin while the supervisor respawns into the same crash.
            switch phase {
            case .idle, .spawning, .socketBound:
                if reason.hasPrefix("engine_exit:") || reason.hasPrefix("spawn_failed:") {
                    crashEvents.append(Date())
                    let recent = crashEvents.filter { Date().timeIntervalSince($0) < 60 }
                    crashEvents = recent
                    if recent.count >= 3 {
                        captureCrashLogTail()
                        let summary = lastCrashLogTail.map(Self.summariseTraceback) ?? reason
                        transition(to: .failed(.execError(summary)),
                                   note: "pre-health crash loop: \(reason)")
                        return
                    }
                }
                // Stay in spawning — supervisor will respawn.
                break
            default:
                transition(to: .degraded(reason: .engineExited, since: Date()), note: reason)
            }
        case .restarting(let attempt):
            spawnAttempt = max(spawnAttempt, attempt)
            if !phase.isTerminalSuccess {
                transition(to: .spawning(attempt: spawnAttempt), note: "supervisor restarting")
            }
        case .starting:
            break
        }
    }

    // MARK: Preflight

    private func preflight() -> FailureKind? {
        let fm = FileManager.default

        // Disk-space probe — guard the runtime dir, not the repo, because the
        // engine writes its socket and crash snapshots there.
        if let runtimeDir = JunoSupportPaths.runtimeDir() {
            try? fm.createDirectory(at: runtimeDir, withIntermediateDirectories: true)
            if let attrs = try? fm.attributesOfFileSystem(forPath: runtimeDir.path),
               let free = (attrs[.systemFreeSize] as? NSNumber)?.int64Value,
               free < 500_000_000 {
                return .diskFull
            }
        }

        // Bundle script presence — only enforced for bundled installs. Repo
        // dev launches fall through to the existing live-script branch.
        if let engineRoot = JunoEngineContract.bundledEngineRoot() {
            let script = engineRoot.appendingPathComponent("run_engine.sh", isDirectory: false)
            if !fm.fileExists(atPath: script.path) {
                return .bundleMissing(path: script.path)
            }
            // Auto-recover non-exec bit before failing.
            if access(script.path, X_OK) != 0 {
                _ = chmod(script.path, 0o755)
                if access(script.path, X_OK) != 0 {
                    return .execError("run_engine.sh is not executable: \(script.path)")
                }
            }
        }

        return nil
    }

    // MARK: Crash-log capture

    /// Read the tail of the most recent ``engine-crash-*.log`` snapshot
    /// written by ``JunoEngineSupervisor.snapshotCrashLog``. The supervisor
    /// dispatches the snapshot off the main actor on
    /// ``Process.terminationHandler``, so this lookup happens slightly
    /// after the offline notification. The 250 ms wait + best-effort lookup
    /// is good enough — if the file isn't there yet we fall back to the
    /// supervisor's reason string, which still surfaces the exit code.
    private func captureCrashLogTail() {
        let fm = FileManager.default
        guard let logsDir = fm.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/Juno", isDirectory: true) else { return }
        // Tiny grace window for the supervisor's async snapshot writer.
        Thread.sleep(forTimeInterval: 0.25)
        let urls = (try? fm.contentsOfDirectory(at: logsDir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        let snapshots = urls.filter { $0.lastPathComponent.hasPrefix("engine-crash-") }
        guard let newest = snapshots.max(by: { lhs, rhs in
            let lm = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rm = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lm < rm
        }) else { return }
        if let body = try? String(contentsOf: newest, encoding: .utf8) {
            let lines = body.split(separator: "\n", omittingEmptySubsequences: false)
            let tail = lines.suffix(40).joined(separator: "\n")
            lastCrashLogTail = tail
        }
    }

    /// Pull the most informative single line out of a captured tail — Python
    /// tracebacks end with the exception type + message, which is what the
    /// user actually wants in the splash subhead. Falls back to the last
    /// non-empty line.
    private static func summariseTraceback(_ tail: String) -> String {
        let lines = tail.split(separator: "\n", omittingEmptySubsequences: true)
        // Walk from the end looking for the canonical "ErrorClass: msg" line.
        for line in lines.reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            if trimmed.hasPrefix("Traceback") || trimmed.hasPrefix("File ") { continue }
            if trimmed.contains("Error:") || trimmed.contains("Exception:") {
                return String(trimmed.prefix(180))
            }
            return String(trimmed.prefix(180))
        }
        return "Voice engine crashed during startup"
    }

    // MARK: Transition + telemetry

    private func transition(to next: Phase, note: String?) {
        if next == phase { return }
        let now = Date()
        let durMs = Int(now.timeIntervalSince(phaseEnteredAt) * 1000)
        let prev = phase
        phase = next
        phaseEnteredAt = now
        updateProgressAndDetail(for: next)

        var error: String? = nil
        if case .failed(let kind) = next {
            error = String(describing: kind)
        }
        JunoLifecycleTraceLog.shared.record(
            from: prev.label,
            to: next.label,
            durMs: durMs,
            attempt: spawnAttempt,
            error: error,
            note: note
        )
        NSLog("Juno lifecycle: \(prev.label) → \(next.label) (\(durMs)ms) — \(note ?? "")")
    }

    private func updateProgressAndDetail(for next: Phase) {
        switch next {
        case .idle:
            statusDetail = ""
            progress = 0
        case .spawning(let attempt):
            statusDetail = attempt > 1
                ? "Restarting voice engine (attempt \(attempt))…"
                : "Starting voice engine — first launch can take 10–15 seconds."
            progress = 0.15
        case .socketBound:
            statusDetail = "Engine connected. Verifying health…"
            progress = 0.4
        case .healthOk:
            statusDetail = "Engine online. Loading speech models…"
            progress = 0.6
        case .modelsLoaded:
            statusDetail = "Models ready. Warming writer…"
            progress = 0.9
        case .ready:
            statusDetail = "Ready."
            progress = 1.0
        case .needsModels:
            statusDetail = "Set up speech models to continue."
            progress = 1.0
        case .needsPermissions:
            statusDetail = "Grant permissions to continue."
            progress = 1.0
        case .degraded(let reason, _):
            statusDetail = "Reconnecting (\(reason.rawValue))…"
        case .failed(let kind):
            statusDetail = kind.userMessage
            progress = 0
        }
    }
}
