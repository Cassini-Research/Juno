import Foundation

// MARK: - JSON decoding (broker uses snake_case)

enum BrokerDecode {
    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    static func objectToData(_ obj: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: obj, options: [])
    }
}

// MARK: - Health + runtime

/// Warmup state reported by the broker (added in PR #33). Values:
///   - ``"ready"``   — engine reachable, models on disk; dictation works.
///   - ``"warming"`` — broker is still loading models (cold cache, multi-GB
///     HF download). The shell should show a "Setting up voice engine"
///     surface instead of "engine unreachable".
///   - ``"error"``   — pre-warm thread raised; first dictation will surface
///     the underlying failure.
/// Optional so the response decodes against older brokers that don't ship
/// the field. ``ready`` mirrors ``state == "ready"`` for callers that want
/// a boolean.
struct BrokerWarmStatus: Codable, Hashable {
    let ready: Bool?
    let state: String?
    let error: String?
}

struct BrokerHealthResponse: Codable {
    let ok: Bool?
    let sessionId: String?
    let warm: BrokerWarmStatus?
}

// MARK: - History

struct HistoryContext: Codable, Hashable {
    let appName: String?
    let appBundleId: String?
    let windowTitle: String?
    let appCategory: String?
}

/// Shell-side enum of recovery affordances the History detail pane is
/// allowed to surface for a row. Mirrors the closed set the broker emits
/// in ``recovery.actions`` — see ``_RECOVERY_BY_FAILURE`` in server.py.
/// Decoding is tolerant: unknown strings decode to ``.unknown`` so a
/// future broker can introduce new actions without breaking older shells.
enum RecoveryAction: String, Codable, Hashable {
    case insertAgain = "insert_again"
    case copyTranscript = "copy_transcript"
    case grantAccessibility = "grant_accessibility"
    case allowApp = "allow_app"
    case replayAudio = "replay_audio"
    case rerunInMode = "rerun_in_mode"
    case restartEngine = "restart_engine"
    case unknown

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = RecoveryAction(rawValue: raw) ?? .unknown
    }
}

/// Broker-derived recovery hints for a single history row. The shell
/// renders the status strip + inline button strictly off these fields;
/// it does not parse failure codes for user-facing copy.
struct HistoryRecoveryHints: Codable, Hashable {
    /// "info" | "warning" | "danger" — drives the strip background colour
    /// in ``HistoryDetailPane``.
    let severity: String?
    /// Loose category for analytics + diagnostics ("insertion", "permission",
    /// "capability", "capture", "engine", "ok", "unknown").
    let category: String?
    let actions: [RecoveryAction]?
    let audioPresent: Bool?
    let hasText: Bool?
    /// Raw broker code (for the diagnostics disclosure only — never user copy).
    let failureCode: String?
}

struct UtteranceHistoryEntry: Codable, Identifiable, Hashable {
    let utteranceId: String
    let tsUnixMs: Int64?
    let mode: String?
    let modelPath: String?
    let context: HistoryContext?
    let failureReason: String?
    let transcript: String?
    let rawTranscript: String?
    let sessionClass: String?
    let transformType: String?
    // Broker returns the SQLite REAL column value, e.g. `1118.0` —
    // decoding into `Int?` throws `typeMismatch` and silently fails the
    // whole `entries` array decode, leaving the History tab empty.
    let processingMs: Double?
    let words: Int?
    let replayAvailable: Bool?
    // Action records (notes / reminders). The broker stores either the
    // parsed-action shape (kind/body/when) or the executor-result shape
    // (kind/status/body_preview/...). ``JunoActionResult`` decodes both
    // tolerantly so a parsed-but-not-yet-dispatched row can never break
    // the History tab. See JunoActionDTOs.swift.
    let actions: [JunoActionResult]?
    /// Pre-computed recovery affordances for the History detail pane.
    /// Optional so older brokers (no recovery field yet) decode cleanly.
    let recovery: HistoryRecoveryHints?
    /// Server-stamped updated_at used as the pagination cursor for "load
    /// older". Optional for legacy brokers; ``tsUnixMs`` is the fallback.
    let updatedAtMs: Int64?
    /// Server-stamped created_at, exposed so diagnostics can show both
    /// timestamps when they differ (broker-on-pause refinement).
    let createdAtMs: Int64?

    var id: String { utteranceId }

    var displayAppName: String {
        context?.appName ?? "Unknown"
    }

    var displayCategory: String? {
        context?.appCategory
    }

    /// Cursor value to pass as ``before_updated_at_ms`` when paginating
    /// past this row. Falls back to ``tsUnixMs`` for older brokers.
    var paginationCursorMs: Int64? {
        if let v = updatedAtMs, v > 0 { return v }
        return tsUnixMs
    }
}

struct BrokerHistoryResponse: Codable {
    let ok: Bool?
    let sessionId: String?
    let entries: [UtteranceHistoryEntry]?
    /// Cursor for loading the next (older) page. Nil when the broker
    /// reports no more rows. Older brokers omit; the shell falls back to
    /// the last entry's ``updatedAtMs`` / ``tsUnixMs``.
    let nextCursorUpdatedAtMs: Int64?
    let hasMore: Bool?
    let pageSize: Int?
}

// MARK: - User profile

struct UserProfileResponse: Codable {
    let ok: Bool?
    let displayName: String?
    let language: String?
}

struct EditingProfileResponse: Codable {
    let ok: Bool?
    let appCategory: String?
    let editingStyle: String?
    let appName: String?
    let windowTitle: String?
    let appBundleId: String?
}

// MARK: - Setup status

struct SetupCheckResult: Codable, Identifiable {
    let name: String
    let ok: Bool
    let detail: String
    var id: String { name }
}

struct BrokerSetupStatusResponse: Codable {
    let ok: Bool?
    let overallReady: Bool?
    let installState: String?
    let brokerReachable: Bool?
    let previewModelReady: Bool?
    let finalModelReady: Bool?
    let writerModelReady: Bool?
    let liveCorrectorModelReady: Bool?
    let liveCorrectorRequired: Bool?
    let liveCorrectorModelCached: Bool?
    let liveCorrectorRuntimeWarm: Bool?
    let liveCorrectorRuntimeLoaded: Bool?
    let writerRequired: Bool?
    let writerModelCached: Bool?
    let writerRuntimeWarm: Bool?
    let writerRuntimeLoaded: Bool?
    let finalBackend: String?
    let writerBackend: String?
    let writerModelPath: String?
    let liveCorrectorBackend: String?
    let liveCorrectorModelPath: String?
    let error: String?
    let checks: [SetupCheckResult]?
    /// Repo ids and display titles from broker `setup/status` (quick_check path).
    let previewRepoId: String?
    let finalRepoId: String?
    let writerRepoId: String?
    let liveCorrectorRepoId: String?
    let previewModelTitle: String?
    let finalModelTitle: String?
    let writerModelTitle: String?
    let liveCorrectorModelTitle: String?
    /// Real-time HF download snapshot — populated by the broker while a
    /// model-provisioning install is in flight. Used by the onboarding
    /// setup step to render a bytes/total progress bar plus speed and ETA.
    let downloadProgress: SetupDownloadProgress?
}

struct SetupDownloadProgress: Codable {
    let bytesSoFar: Int64?
    let bytesTotal: Int64?
    let bytesPerSecond: Double?
    let etaSeconds: Double?
    let elapsedSeconds: Double?
    let repos: [String]?
    /// Repo id currently being provisioned (first incomplete in install
    /// order) and how many are already complete — drives the "Now:
    /// <model> (2 of 4)" line on the onboarding setup card.
    let currentRepo: String?
    let currentLane: String?
    let reposDone: Int?
    /// Short human-readable install transitions ("Downloading X (1 of 4)",
    /// "Loading models into memory"). Rendered as the setup card's status
    /// log; also a diagnostic breadcrumb for stuck installs.
    let log: [SetupInstallLogEntry]?
}

struct SetupInstallLogEntry: Codable {
    let t: Double?
    let line: String?
}

// MARK: - Stats

struct StatsAppResponse: Codable, Hashable, Identifiable {
    let name: String
    let words: Int

    var id: String { name }
}

struct StatsPeriodResponse: Codable, Hashable, Identifiable {
    let id: String
    let totalWords: Int
    let dictations: Int
    let timeSavedS: Int
    let bucketStartDates: [String]
    let bucketEndDates: [String]
    let wordsByBucket: [Int]
    let topApps: [StatsAppResponse]
}

struct StatsSummaryResponse: Codable {
    let ok: Bool?
    let wordsToday: Int?
    let wordsWeek: Int?
    let appsToday: Int?
    let timeSavedS: Int?
    let timeSavedMin: Int?
    let computedAtUnixMs: Int64?
    /// Words spoken per day for the last 7 days, oldest → today. Optional
    /// for backwards-compat with older brokers that didn't emit this field.
    let wordsByDay: [Int]?
    /// App names used today, sorted by descending frequency. Used by Home's
    /// stats foot strip to render real app icons + a "+N more" pill.
    let appsTodayTop: [String]?
    /// Top app name used today (for the "most in X" caption). Optional.
    let topAppToday: String?
    /// Additive range summaries for the dedicated Stats page. Optional so
    /// a newer shell can still show Home against an older local broker.
    let periods: [StatsPeriodResponse]?
}

// MARK: - JunoBroker typed GET / POST decode

extension JunoBroker {
    static func fetchHealth(
        completion: @escaping (Result<BrokerHealthResponse, Error>) -> Void
    ) {
        getJSON(path: "healthz") { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(BrokerHealthResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }

    static func fetchHistory(
        limit: Int = 50,
        beforeUpdatedAtMs: Int64? = nil,
        completion: @escaping (Result<BrokerHistoryResponse, Error>) -> Void
    ) {
        var path = "api/broker/history?limit=\(limit)"
        if let cursor = beforeUpdatedAtMs, cursor > 0 {
            path += "&before_updated_at_ms=\(cursor)"
        }
        getJSON(path: path) { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(BrokerHistoryResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }

    /// Insert-again recovery: ask the broker for the saved transcript on a
    /// row whose original paste failed, so the shell can re-fire paste via
    /// the existing :class:`Clipboard` capability path. Keystroke synthesis
    /// stays on the macOS side — we never paste from Python.
    static func postHistoryInsertAgain(
        utteranceId: String,
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        postJSON(
            path: "api/broker/history/insert_again",
            payload: ["utterance_id": utteranceId]
        ) { obj in
            if let ok = obj["ok"] as? Bool, ok == false {
                let msg = (obj["error"] as? String) ?? "insert_again_failed"
                completion(.failure(NSError(domain: "JunoBroker", code: -20,
                                            userInfo: [NSLocalizedDescriptionKey: msg])))
                return
            }
            completion(.success(obj))
        }
    }

    static func fetchRuntimeJSON(
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        getJSON(path: "api/runtime") { obj in
            if let ok = obj["ok"] as? Bool, ok == false {
                let msg = (obj["error"] as? String) ?? "runtime_unavailable"
                completion(.failure(NSError(domain: "JunoBroker", code: -10,
                                            userInfo: [NSLocalizedDescriptionKey: msg])))
                return
            }
            completion(.success(obj))
        }
    }

    /// POST juno-capability-shaped payload; returns editing profile for HUD.
    static func postEditingProfile(
        payload: [String: Any],
        completion: @escaping (Result<EditingProfileResponse, Error>) -> Void
    ) {
        postJSON(path: "api/broker/surface/editing_profile", payload: payload) { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(EditingProfileResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }

    static func fetchSetupStatus(
        completion: @escaping (Result<BrokerSetupStatusResponse, Error>) -> Void
    ) {
        getJSON(path: "api/broker/setup/status") { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(BrokerSetupStatusResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }

    static func fetchStatsSummary(
        completion: @escaping (Result<StatsSummaryResponse, Error>) -> Void
    ) {
        getJSON(path: "api/broker/stats/summary") { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(StatsSummaryResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }

    static func postSetupInstall(
        repair: Bool = false,
        restart: Bool = false,
        completion: @escaping (Result<[String: Any], Error>) -> Void
    ) {
        // A restart supersedes an in-flight (stuck/stalled) download instead of
        // no-opping; it uses the force/repair route and carries the restart
        // marker the broker keys its generation bump on.
        let path = (repair || restart) ? "api/broker/setup/repair" : "api/broker/setup/install"
        let payload: [String: Any] = restart ? ["restart": true] : [:]
        postJSON(path: path, payload: payload) { obj in
            completion(.success(obj))
        }
    }

    static func fetchUserProfile(
        completion: @escaping (Result<UserProfileResponse, Error>) -> Void
    ) {
        getJSON(path: "api/broker/personalization/user_profile") { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(UserProfileResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }

    static func postUserProfile(
        displayName: String?,
        language: String? = nil,
        completion: @escaping (Result<UserProfileResponse, Error>) -> Void = { _ in }
    ) {
        var payload: [String: Any] = [:]
        if let displayName { payload["display_name"] = displayName }
        if let language { payload["language"] = language }
        postJSON(path: "api/broker/personalization/user_profile", payload: payload) { obj in
            do {
                let data = try BrokerDecode.objectToData(obj)
                let v = try BrokerDecode.decoder.decode(UserProfileResponse.self, from: data)
                completion(.success(v))
            } catch {
                completion(.failure(error))
            }
        }
    }
}
