import Combine
import Foundation

// MARK: - Setup model

/// Tracks Juno's runtime/model readiness by polling the broker setup status
/// endpoint. Used by onboarding and settings to show real install state and
/// offer install/repair actions.
@MainActor
final class JunoSetupModel: ObservableObject {
    @Published private(set) var overallReady: Bool = false
    @Published private(set) var installState: String = "unknown"
    @Published private(set) var checks: [SetupCheckResult] = []
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var previewModelReady: Bool
    @Published private(set) var finalModelReady: Bool
    @Published private(set) var writerModelReady: Bool
    @Published private(set) var liveCorrectorModelReady: Bool
    @Published private(set) var finalBackend: String = ""
    @Published private(set) var writerBackend: String = "none"
    @Published private(set) var writerModelPath: String = ""
    @Published private(set) var liveCorrectorBackend: String = "none"
    @Published private(set) var liveCorrectorModelPath: String = ""
    @Published private(set) var liveCorrectorRequired: Bool = false
    @Published private(set) var liveCorrectorModelCached: Bool = false
    @Published private(set) var liveCorrectorRuntimeWarm: Bool = false
    @Published private(set) var liveCorrectorRuntimeLoaded: Bool = false
    @Published private(set) var writerRequired: Bool = false
    @Published private(set) var writerModelCached: Bool = false
    @Published private(set) var writerRuntimeWarm: Bool = false
    @Published private(set) var writerRuntimeLoaded: Bool = false
    @Published private(set) var previewRepoId: String = ""
    @Published private(set) var finalRepoId: String = ""
    @Published private(set) var writerRepoId: String = ""
    @Published private(set) var liveCorrectorRepoId: String = ""
    @Published private(set) var previewModelTitle: String = ""
    @Published private(set) var finalModelTitle: String = ""
    @Published private(set) var writerModelTitle: String = ""
    @Published private(set) var liveCorrectorModelTitle: String = ""
    /// Mirrors PR #33's ``warm.state`` field on ``/healthz``. When the
    /// broker is reachable but still downloading or loading models on
    /// first launch, this flips to ``"warming"`` so the onboarding view
    /// can show a "Setting up voice engine, ~2 min on first run"
    /// surface instead of falling back to the existing
    /// ``broker_unreachable`` copy. ``nil`` for older brokers that
    /// don't ship the field.
    @Published private(set) var engineWarmingState: String?
    /// True when ``JunoLocalBrokerBootstrap.spawnBundledEngine`` returned without launching a process
    /// (missing script, no repo root, exec failure). Distinct from ``broker_unreachable`` which is
    /// also "still warming"; this one is unrecoverable without user action.
    @Published private(set) var bootstrapFailed: Bool = false
    @Published private(set) var bootstrapFailureReason: String?
    @Published private(set) var enginePresenceUnknown: Bool = false
    /// True after the first ``fetchSetupStatus`` attempt finishes (success or failure).
    @Published private(set) var hasCompletedSetupFetch: Bool = false
    /// True once we have successfully decoded a setup/status payload (persists across later transient failures).
    @Published private(set) var receivedSuccessfulSetupPayload: Bool = false
    /// Latest ``/healthz`` reachability from ``pingHealthDetailed`` (parallel to setup/status).
    @Published private(set) var lastHealthPingReachable: Bool?
    /// Live HF download snapshot — broker publishes bytes_so_far / bytes_total /
    /// bytes_per_second / eta_seconds while a provisioning install runs. Reset
    /// to zeros when no install is active. Drives the premium download UI on
    /// the onboarding setup step (bytes-MB progress bar, speed, ETA).
    @Published private(set) var downloadBytesSoFar: Int64 = 0
    @Published private(set) var downloadBytesTotal: Int64 = 0
    @Published private(set) var downloadBytesPerSecond: Double = 0
    @Published private(set) var downloadEtaSeconds: Double? = nil
    @Published private(set) var downloadElapsedSeconds: Double = 0
    @Published private(set) var downloadActive: Bool = false
    /// Repo currently downloading ("mlx-community/…") plus x-of-y position,
    /// and the broker's short install log ("Downloading X (1 of 4)",
    /// "Loading models into memory"). Drives the per-model line and the
    /// status log on the onboarding setup card.
    @Published private(set) var downloadCurrentRepo: String? = nil
    @Published private(set) var downloadReposDone: Int = 0
    @Published private(set) var downloadReposTotal: Int = 0
    @Published private(set) var downloadLog: [String] = []

    init() {
        // Seed the per-lane readiness flags from the on-disk inventory so
        // the UI shows the right state on first render — *before* the
        // first /api/broker/setup/status round-trip lands. Without this,
        // a fresh launch with cached HF models for ~200-500ms shows the
        // Step-4 onboarding cards with download icons because all three
        // *Ready flags default to false. The first poll then flips them
        // to true and the cards re-render. The transient looks broken
        // ("Preparing voice models" with download arrows even though the
        // engine is fully warmed) and is what users report as "models
        // didn't load."
        let inv = JunoLocalModelInventory.snapshot()
        self.previewModelReady = inv.previewModelOnDisk
        self.finalModelReady = inv.finalModelOnDisk
        // Writer cache lives in the same HF cache dir but the inventory
        // probe only checks preview + final today. We default the writer
        // flag to true only when the inventory confirmed at least one of
        // the lanes is on disk (a strong signal that the HF cache root
        // exists); otherwise leave it false so the broker poll fills it.
        self.writerModelReady = inv.previewModelOnDisk || inv.finalModelOnDisk
        self.liveCorrectorModelReady = inv.previewModelOnDisk || inv.finalModelOnDisk
    }

    func clearBootstrapFailure() {
        bootstrapFailed = false
        bootstrapFailureReason = nil
    }

    private var timer: AnyCancellable?
    private var bootstrapObserver: NSObjectProtocol?
    private var fastPollTimer: AnyCancellable?
    private var refreshGeneration: UInt64 = 0

    /// When true, Home should not show the heavy **Finish setup** gate yet (warming or transient HTTP race).
    var shouldDeferFinishSetupGate: Bool {
        if engineWarmingState == "warming" { return true }
        if installState == "broker_unreachable", lastHealthPingReachable == true, !receivedSuccessfulSetupPayload {
            return true
        }
        return false
    }

    func installAndStartLaunchdEngine() {
        let fm = FileManager.default
        let scriptCandidates: [URL] = {
            var u: [URL] = []
            if let b = JunoEngineContract.bundledLaunchdInstallerURL() {
                u.append(b)
            }
            if let root = JunoRepoPaths.guessRepoRoot() {
                u.append(URL(fileURLWithPath: root).appendingPathComponent("scripts/install_juno_launchd.sh"))
            }
            return u
        }()
        guard let script = scriptCandidates.first(where: { fm.fileExists(atPath: $0.path) }) else {
            return
        }
        var args = [script.path, "install"]
        if let engineRoot = JunoEngineContract.bundledEngineRoot(),
           fm.fileExists(atPath: engineRoot.appendingPathComponent(".venv/bin/python").path) {
            args.append(contentsOf: ["--engine-bundle", engineRoot.path])
        } else if let root = JunoRepoPaths.guessRepoRoot() {
            args.append(contentsOf: ["--repo-root", root])
        } else {
            return
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = args
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        do {
            try p.run()
        } catch {
            return
        }
    }

    func startPolling() {
        timer?.cancel()
        refresh()
        timer = Timer.publish(every: 12, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.refresh() }
        if bootstrapObserver == nil {
            bootstrapObserver = NotificationCenter.default.addObserver(
                forName: .junoBrokerBootstrapFailed, object: nil, queue: .main
            ) { [weak self] note in
                Task { @MainActor [weak self] in
                    self?.bootstrapFailed = true
                    self?.bootstrapFailureReason = note.object as? String
                }
            }
        }
    }

    func stopPolling() {
        timer?.cancel()
        timer = nil
        stopFastPoll()
        if let obs = bootstrapObserver {
            NotificationCenter.default.removeObserver(obs)
            bootstrapObserver = nil
        }
    }

    func refresh() {
        refreshGeneration += 1
        let gen = refreshGeneration
        isLoading = true
        // Parallel /healthz probe so the onboarding view can distinguish
        // "broker is warming up models on first launch" from "broker is
        // genuinely unreachable". Captured into a dedicated published
        // field rather than overloading installState because installState
        // already serializes ready/downloading/error from a different
        // endpoint and we don't want to break that contract.
        JunoBroker.pingHealthDetailed { [weak self] snapshot in
            guard let self else { return }
            guard gen == self.refreshGeneration else { return }
            if let snapshot {
                self.lastHealthPingReachable = snapshot.reachable
                self.engineWarmingState = snapshot.warmState
            } else {
                self.lastHealthPingReachable = false
            }
        }
        JunoBroker.fetchSetupStatus { [weak self] result in
            guard let self else { return }
            guard gen == self.refreshGeneration else { return }
            self.isLoading = false
            self.hasCompletedSetupFetch = true
            switch result {
            case .success(let s):
                self.receivedSuccessfulSetupPayload = true
                self.bootstrapFailed = false
                self.bootstrapFailureReason = nil
                self.enginePresenceUnknown = false
                self.overallReady = s.overallReady ?? false
                self.installState = s.installState ?? "unknown"
                self.checks = s.checks ?? []
                self.previewModelReady = s.previewModelReady ?? false
                self.finalModelReady = s.finalModelReady ?? false
                self.writerModelReady = s.writerModelReady ?? false
                self.liveCorrectorModelReady = s.liveCorrectorModelReady ?? false
                self.finalBackend = s.finalBackend ?? ""
                self.writerBackend = s.writerBackend ?? "none"
                self.writerModelPath = s.writerModelPath ?? ""
                self.liveCorrectorBackend = s.liveCorrectorBackend ?? "none"
                self.liveCorrectorModelPath = s.liveCorrectorModelPath ?? ""
                self.liveCorrectorRequired = s.liveCorrectorRequired ?? false
                self.liveCorrectorModelCached = s.liveCorrectorModelCached ?? false
                self.liveCorrectorRuntimeWarm = s.liveCorrectorRuntimeWarm ?? false
                self.liveCorrectorRuntimeLoaded = s.liveCorrectorRuntimeLoaded ?? false
                self.writerRequired = s.writerRequired ?? false
                self.writerModelCached = s.writerModelCached ?? false
                self.writerRuntimeWarm = s.writerRuntimeWarm ?? false
                self.writerRuntimeLoaded = s.writerRuntimeLoaded ?? false
                self.previewRepoId = s.previewRepoId ?? ""
                self.finalRepoId = s.finalRepoId ?? ""
                self.writerRepoId = s.writerRepoId ?? ""
                self.liveCorrectorRepoId = s.liveCorrectorRepoId ?? ""
                self.previewModelTitle = s.previewModelTitle ?? ""
                self.finalModelTitle = s.finalModelTitle ?? ""
                self.writerModelTitle = s.writerModelTitle ?? ""
                self.liveCorrectorModelTitle = s.liveCorrectorModelTitle ?? ""
                self.errorMessage = s.error
                if let dp = s.downloadProgress {
                    self.downloadActive = true
                    self.downloadBytesSoFar = dp.bytesSoFar ?? 0
                    self.downloadBytesTotal = dp.bytesTotal ?? 0
                    self.downloadBytesPerSecond = dp.bytesPerSecond ?? 0
                    self.downloadEtaSeconds = dp.etaSeconds
                    self.downloadElapsedSeconds = dp.elapsedSeconds ?? 0
                    self.downloadCurrentRepo = dp.currentRepo
                    self.downloadReposDone = dp.reposDone ?? 0
                    self.downloadReposTotal = dp.repos?.count ?? 0
                    self.downloadLog = (dp.log ?? []).compactMap { $0.line }
                } else {
                    self.downloadActive = false
                    self.downloadBytesSoFar = 0
                    self.downloadBytesTotal = 0
                    self.downloadBytesPerSecond = 0
                    self.downloadEtaSeconds = nil
                    self.downloadElapsedSeconds = 0
                    self.downloadCurrentRepo = nil
                    self.downloadReposDone = 0
                    self.downloadReposTotal = 0
                    self.downloadLog = []
                }
                // Wake JunoEngineLifecycle so it can re-evaluate `phase`
                // when install advances. waitForSetup() exits at the first
                // terminal phase and never re-runs; without this notify
                // the home gate gets glued to .needsModels even after
                // overallReady flips to true. See Notification.Name.
                NotificationCenter.default.post(
                    name: .junoSetupSnapshotUpdated, object: s
                )
                // If install finished, stop fast polling
                if self.installState == "ready" || self.installState.hasPrefix("failed") {
                    self.stopFastPoll()
                }
            case .failure:
                self.installState = "broker_unreachable"
                self.overallReady = false
                self.errorMessage = "Broker not reachable"
                let inv = JunoLocalModelInventory.snapshot()
                self.previewModelReady = inv.previewModelOnDisk
                self.finalModelReady = inv.finalModelOnDisk
                self.liveCorrectorModelReady = false
                self.enginePresenceUnknown = true
                self.liveCorrectorBackend = "none"
                self.liveCorrectorModelPath = ""
                self.previewRepoId = ""
                self.finalRepoId = ""
                self.writerRepoId = ""
                self.liveCorrectorRepoId = ""
                self.previewModelTitle = ""
                self.finalModelTitle = ""
                self.writerModelTitle = ""
                self.liveCorrectorModelTitle = ""
                self.writerRequired = false
                self.writerModelCached = false
                self.writerRuntimeWarm = false
                self.writerRuntimeLoaded = false
                self.liveCorrectorRequired = false
                self.liveCorrectorModelCached = false
                self.liveCorrectorRuntimeWarm = false
                self.liveCorrectorRuntimeLoaded = false
                // Keep ``engineWarmingState`` / ping snapshot from the parallel
                // ``pingHealthDetailed`` callback — do not clear warming here.
            }
        }
    }

    func triggerInstall() {
        installState = "downloading"
        JunoBroker.postSetupInstall(repair: false) { _ in }
        startFastPoll()
    }

    func triggerRepair() {
        installState = "downloading"
        JunoBroker.postSetupInstall(repair: true) { _ in }
        startFastPoll()
    }

    var readinessLabel: String {
        switch installState {
        case "ready": return "Ready"
        case "downloading": return "Downloading…"
        case "not_started": return "Not installed"
        case "needs_setup": return "Setup required"
        case "broker_unreachable": return "Broker not running"
        case "error": return "Error"
        case "unknown": return "Checking…"
        default:
            if installState.hasPrefix("failed:") { return "Install failed" }
            return installState
        }
    }

    var canInstall: Bool {
        installState == "not_started" || installState == "needs_setup"
    }

    var canRepair: Bool {
        installState.hasPrefix("failed") || installState == "ready"
    }

    private func startFastPoll() {
        fastPollTimer?.cancel()
        fastPollTimer = Timer.publish(every: 3, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.refresh() }
    }

    private func stopFastPoll() {
        fastPollTimer?.cancel()
        fastPollTimer = nil
    }

}
