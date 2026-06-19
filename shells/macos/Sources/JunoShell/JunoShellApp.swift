import ApplicationServices
import AppKit
import AVFoundation
import Combine
import CoreAudio
import Darwin
import JunoHotkeyCore
import JunoObjCSupport
import OSLog
import SwiftUI

/// Default-input-device name (e.g. "MacBook Pro Microphone"). `nil` if the system
/// has no default input or Core Audio refuses the query — caller falls back to
/// a generic label.
private func defaultInputDeviceName() -> String? {
    var deviceID = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &deviceID
    )
    guard status == noErr, deviceID != kAudioObjectUnknown else { return nil }

    var nameAddr = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var name: Unmanaged<CFString>?
    var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let r = AudioObjectGetPropertyData(deviceID, &nameAddr, 0, nil, &nameSize, &name)
    guard r == noErr, let cf = name?.takeRetainedValue() else { return nil }
    return cf as String
}

private let junoOnboardingLog = Logger(subsystem: "com.juno.shell", category: "onboarding")

// MARK: - Optional stderr-to-file capture for support bundles

/// When `JunoUserDefaults.saveLogsToFileEnabled` is on, redirect the process's
/// stderr — where `NSLog` writes — into `~/Library/Logs/Juno/juno-app.log` so
/// the support-bundle exporter has Swift-side log lines to include. The
/// redirect is installed once during `init()` and persists for the process
/// lifetime; flipping the toggle takes effect on the next launch (the
/// "next launch" caveat is documented in the toggle's subtitle).
///
/// We use `freopen` instead of a Swift `FileHandle` because `NSLog` writes via
/// the C-runtime `stderr` `FILE*`, which only `freopen`/`dup2` can redirect.
/// A `FileHandle`-based copy would miss everything `NSLog` emits.
enum JunoLocalAppLogTee {
    static func installIfEnabled() {
        guard JunoUserDefaults.saveLogsToFileEnabled else { return }
        let fm = FileManager.default
        guard let lib = fm.urls(for: .libraryDirectory, in: .userDomainMask).first else { return }
        let dir = lib.appendingPathComponent("Logs/Juno", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let target = dir.appendingPathComponent("juno-app.log", isDirectory: false).path
        // "a" = append; tee survives across relaunches, doesn't truncate prior
        // sessions. If the user wants a clean file, they can rotate it from
        // Finder.
        _ = target.withCString { freopen($0, "a", stderr) }
        let stamp = ISO8601DateFormatter().string(from: Date())
        NSLog("Juno: app log redirected to \(target) at \(stamp)")
    }
}

// MARK: - Bundle-id-keyed support paths (production-grade revamp, Phase 0)

/// Single source of truth for Juno's per-user filesystem layout. Mirrors
/// ``juno_v2.runtime.paths`` on the Python side; both must agree on
/// where the auth token, instance lock, engine socket, and endpoint
/// metadata live or the shell and engine will see different files.
enum JunoSupportPaths {
    static let legacyDirectoryName = "Juno"

    static func bundleId() -> String {
        Bundle.main.bundleIdentifier ?? "com.juno.shell"
    }

    static func applicationSupportRoot() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
    }

    /// ``~/Library/Application Support/<bundle-id>``. Created on demand.
    static func supportRoot() -> URL? {
        guard let parent = applicationSupportRoot() else { return nil }
        let url = parent.appendingPathComponent(bundleId(), isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// ``<support-root>/runtime`` — owns lock, socket, token, endpoint json.
    /// Created with mode 0700 so only the user's UID can connect.
    static func runtimeDir() -> URL? {
        guard let root = supportRoot() else { return nil }
        let url = root.appendingPathComponent("runtime", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        return url
    }

    static func tokenURL() -> URL? {
        runtimeDir()?.appendingPathComponent("broker_local_token", isDirectory: false)
    }

    static func supportRootTokenURL() -> URL? {
        supportRoot()?.appendingPathComponent("broker_local_token", isDirectory: false)
    }

    static func engineSocketURL() -> URL? {
        if let raw = ProcessInfo.processInfo.environment["JUNO_DEV_ENGINE_SOCKET"], !raw.isEmpty {
            return URL(fileURLWithPath: raw)
        }
        if let raw = ProcessInfo.processInfo.environment["JUNO_ENGINE_SOCKET"], !raw.isEmpty {
            return URL(fileURLWithPath: raw)
        }
        return runtimeDir()?.appendingPathComponent("engine.sock", isDirectory: false)
    }

    static func legacySupportRoot() -> URL? {
        applicationSupportRoot()?.appendingPathComponent(legacyDirectoryName, isDirectory: true)
    }

    static func legacyTokenURL() -> URL? {
        legacySupportRoot()?.appendingPathComponent("broker_local_token", isDirectory: false)
    }
}

// MARK: - Local broker auth (loopback shared secret)

enum JunoLocalBrokerAuth {
    /// Reads the local auth token written by the Python engine. Tries the
    /// bundle-id-keyed location first (current convention) and falls back
    /// to the legacy ``Application Support/Juno/`` location for installs
    /// that pre-date the production-grade path migration.
    static func tokenString() -> String? {
        let candidates = [
            JunoSupportPaths.tokenURL(),
            JunoSupportPaths.supportRootTokenURL(),
            JunoSupportPaths.legacyTokenURL(),
        ].compactMap { $0 }
        for url in candidates {
            guard let data = try? Data(contentsOf: url),
                  let s = String(data: data, encoding: .utf8) else { continue }
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { return t }
        }
        return nil
    }

    static func attach(to req: inout URLRequest) {
        if let t = tokenString() {
            req.setValue(t, forHTTPHeaderField: "X-Juno-Local-Token")
        }
    }
}

// MARK: - Broker HTTP client

/// Thin macOS shell: menu-bar surface that drives the broker over HTTP
/// routes mounted on the workbench process. No models, no heavy work — this
/// file is the product *surface*, not the engine.
enum JunoBroker {
    /// Base URL for the local engine/workbench that hosts broker routes.
    ///
    /// Development note: the demo launcher will auto-pick a free port near 8765,
    /// so the shell must not assume a fixed port.
    static var baseURL: URL {
        resolvedBaseURL ?? envBaseURL() ?? URL(string: "http://127.0.0.1:8765")!
    }

    private static var resolvedBaseURL: URL?
    private static var discoveryInFlight = false

    static var shouldUseUDS: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["JUNO_WORKBENCH_URL"] == nil && env["JUNO_WORKBENCH_URL"] == nil
    }

    static var engineSocketPath: String {
        JunoSupportPaths.engineSocketURL()?.path
            ?? "\(NSHomeDirectory())/Library/Application Support/\(JunoSupportPaths.bundleId())/runtime/engine.sock"
    }

    private static func splitPathAndQuery(_ path: String) -> (String, String?) {
        if let idx = path.firstIndex(of: "?") {
            return (String(path[..<idx]), String(path[path.index(after: idx)...]))
        }
        return (path, nil)
    }

    static func callBrokerRPC(
        httpMethod: String,
        path: String,
        payload: [String: Any] = [:],
        binary: Data? = nil,
        timeoutSeconds: TimeInterval? = nil,
        completion: @escaping (Result<(object: [String: Any], binary: Data?), Error>) -> Void
    ) {
        var params: [String: Any] = [
            "http_method": httpMethod,
        ]
        let split = splitPathAndQuery(path)
        params["path"] = split.0.hasPrefix("/") ? split.0 : "/\(split.0)"
        if let query = split.1 { params["query"] = query }
        if !payload.isEmpty { params["payload"] = payload }
        if let token = JunoLocalBrokerAuth.tokenString(), !token.isEmpty {
            params["_auth"] = token
        }
        let rpcTimeout = timeoutSeconds ?? brokerRPCTimeoutSeconds(httpMethod: httpMethod, path: split.0, binary: binary)
        Task {
            do {
                let out = try await JunoEngineClient.call(
                    socketPath: engineSocketPath,
                    method: "broker.http",
                    params: params,
                    binary: binary,
                    timeoutSeconds: rpcTimeout
                )
                await MainActor.run { completion(.success((object: out.result, binary: out.binary))) }
            } catch {
                await MainActor.run { completion(.failure(error)) }
            }
        }
    }

    private static func brokerRPCTimeoutSeconds(httpMethod: String, path: String, binary: Data?) -> TimeInterval {
        let normalized = path.hasPrefix("/") ? path : "/\(path)"
        if normalized == "/healthz" {
            return 0.75
        }
        if normalized.contains("/api/broker/engine/compatibility") {
            return 1.0
        }
        if normalized.contains("/api/broker/surface/capability") {
            return 1.5
        }
        if normalized.contains("/api/broker/shell/home_greeting") {
            return 8.0
        }
        if normalized.contains("/api/broker/dictation/live_correct") {
            return 15.0
        }
        if normalized.contains("/api/broker/dictation/ingest_wav") || binary != nil {
            return 180.0
        }
        return httpMethod.uppercased() == "GET" ? 12.0 : 30.0
    }

    private static func envBaseURL() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["JUNO_WORKBENCH_URL"], let u = URL(string: raw) { return u }
        if let raw = env["JUNO_WORKBENCH_URL"], let u = URL(string: raw) { return u }
        return nil
    }

    private static func healthzOK(_ base: URL, completion: @escaping (Bool) -> Void) {
        let url = base.appendingPathComponent("healthz")
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 0.35
        cfg.timeoutIntervalForResource = 0.35
        let session = URLSession(configuration: cfg)
        session.dataTask(with: url) { _, response, _ in
            let ok = (response as? HTTPURLResponse)?.statusCode == 200
            completion(ok)
        }.resume()
    }

    /// Probe identity of whatever is listening at ``base``. Returns the
    /// engine identity payload on success, or ``nil`` if no compatible
    /// HTTP server answered. Distinguishes ``juno_runtime_service`` from
    /// any other peer (workbench standalone, random dev server, etc.).
    static func probeIdentity(at base: URL, completion: @escaping (EngineIdentity?) -> Void) {
        let url = base.appendingPathComponent("api/broker/engine/compatibility")
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 0.6
        cfg.timeoutIntervalForResource = 0.6
        URLSession(configuration: cfg).dataTask(with: url) { data, response, _ in
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  (obj["ok"] as? Bool) == true else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let role = (obj["runtime_role"] as? String) ?? ""
            let version = (obj["shell_engine_protocol_version"] as? NSNumber)?.intValue
                ?? (obj["shell_engine_protocol_version"] as? Int) ?? 0
            let identity = EngineIdentity(
                runtimeRole: role,
                instanceId: (obj["instance_id"] as? String) ?? "",
                bundleId: (obj["bundle_id"] as? String) ?? "",
                pid: (obj["pid"] as? NSNumber)?.intValue ?? (obj["pid"] as? Int) ?? 0,
                protocolVersion: version,
                deploymentProfile: deploymentProfile(from: obj)
            )
            DispatchQueue.main.async { completion(identity) }
        }.resume()
    }

    /// Allow the supervisor / bootstrap to publish the URL of an engine
    /// it just spawned, so subsequent broker calls bypass discovery.
    @MainActor static func setResolvedBaseURL(_ url: URL) {
        resolvedBaseURL = url
    }

    /// Discover the live engine URL.
    ///
    /// Production-grade revamp: the discovery contract is no longer
    /// "anything answering ``/healthz``" — that's how a developer's
    /// ``python -m juno_v2.workbench.server`` on 8765 ended up
    /// impersonating the bundled engine. Discovery now requires the peer
    /// to identify itself as ``juno_runtime_service`` via the
    /// compatibility endpoint. Anything else is treated as "no engine
    /// found" so the supervisor will start a fresh one on a different
    /// port.
    ///
    /// Strategy:
    /// 1. Env override (``JUNO_WORKBENCH_URL`` / ``JUNO_WORKBENCH_URL``).
    /// 2. Scan ``8765..8785`` for a peer reporting our runtime role.
    /// 3. Return the default URL even on miss — caller (supervisor) will
    ///    handle "engine not running" by spawning one.
    private static func discoverBaseURLIfNeeded(completion: @escaping (URL) -> Void) {
        if let resolvedBaseURL {
            completion(resolvedBaseURL)
            return
        }
        if let u = envBaseURL() {
            resolvedBaseURL = u
            completion(u)
            return
        }
        if discoveryInFlight {
            completion(URL(string: "http://127.0.0.1:8765")!)
            return
        }
        discoveryInFlight = true

        let host = "127.0.0.1"
        let candidates: [URL] = (8765...8785).compactMap { port in
            URL(string: "http://\(host):\(port)")
        }
        let fallback = URL(string: "http://\(host):8765")!

        func finish(_ u: URL) {
            DispatchQueue.main.async {
                discoveryInFlight = false
                resolvedBaseURL = u
                completion(u)
            }
        }

        let queue = DispatchQueue(label: "com.juno.shell.brokerDiscovery", qos: .userInitiated)
        queue.async {
            var found: URL?
            for u in candidates {
                if found != nil { break }
                let sema = DispatchSemaphore(value: 0)
                probeIdentity(at: u) { identity in
                    if let identity,
                       identity.runtimeRole == JunoEngineContract.requiredRuntimeRole,
                       found == nil {
                        found = u
                    }
                    sema.signal()
                }
                _ = sema.wait(timeout: .now() + 0.7)
            }
            finish(found ?? fallback)
        }
    }

    /// Build a request URL from a path that may contain a query string.
    /// `URL.appendingPathComponent` percent-encodes `?` so a path like
    /// `"api/broker/history?limit=80"` becomes `…/history%3Flimit=80`,
    /// which the broker 404s on. Use string concatenation instead so
    /// the query separator survives intact, falling back to
    /// `appendingPathComponent` only when the resulting string fails
    /// to parse as a URL (e.g. a path with embedded spaces).
    static func resolveURL(path: String) -> URL {
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        let combined = baseURL.absoluteString.hasSuffix("/")
            ? baseURL.absoluteString + trimmed
            : baseURL.absoluteString + "/" + trimmed
        if let u = URL(string: combined) { return u }
        return baseURL.appendingPathComponent(trimmed)
    }

    static func pingHealth(completion: @escaping (Bool) -> Void) {
        pingHealthDetailed { result in
            completion(result?.reachable == true)
        }
    }

    /// Health snapshot used by the onboarding view. ``reachable`` tracks
    /// whether ``/healthz`` returned 200 + the engine compatibility
    /// handshake succeeded; ``warmState`` mirrors PR #33's
    /// ``warm.state`` field so the shell can show a "Setting up voice
    /// engine" surface during cold-cache model download (60-180 s on
    /// first launch) instead of "engine unreachable".
    struct HealthSnapshot {
        let reachable: Bool
        /// "ready" | "warming" | "error" | nil (older brokers without the field).
        let warmState: String?
    }

    static func pingHealthDetailed(completion: @escaping (HealthSnapshot?) -> Void) {
        if shouldUseUDS {
            rawPingHealthDetailed { snapshot in
                guard let snapshot, snapshot.reachable else {
                    completion(snapshot)
                    return
                }
                ensureCompatible { result in
                    switch result {
                    case .success:
                        completion(snapshot)
                    case .failure:
                        completion(HealthSnapshot(reachable: false, warmState: snapshot.warmState))
                    }
                }
            }
            return
        }
        discoverBaseURLIfNeeded { _ in
            rawPingHealthDetailed { snapshot in
                guard let snapshot, snapshot.reachable else {
                    completion(snapshot)
                    return
                }
                ensureCompatible { result in
                    switch result {
                    case .success:
                        completion(snapshot)
                    case .failure:
                        completion(HealthSnapshot(reachable: false, warmState: snapshot.warmState))
                    }
                }
            }
        }
    }

    private static func rawPingHealthDetailed(completion: @escaping (HealthSnapshot?) -> Void) {
        if shouldUseUDS {
            callBrokerRPC(httpMethod: "GET", path: "healthz") { result in
                switch result {
                case .success(let out):
                    let obj = out.object
                    var warmState: String? = nil
                    if let warm = obj["warm"] as? [String: Any] {
                        warmState = warm["state"] as? String
                    }
                    completion(HealthSnapshot(reachable: (obj["ok"] as? Bool) == true, warmState: warmState))
                case .failure:
                    completion(HealthSnapshot(reachable: false, warmState: nil))
                }
            }
            return
        }
        let url = baseURL.appendingPathComponent("healthz")
        URLSession.shared.dataTask(with: url) { data, response, _ in
            let httpOK = (response as? HTTPURLResponse)?.statusCode == 200
            var warmState: String? = nil
            if httpOK, let data {
                if let parsed = try? BrokerDecode.decoder.decode(BrokerHealthResponse.self, from: data) {
                    warmState = parsed.warm?.state
                }
            }
            let snapshot = HealthSnapshot(reachable: httpOK, warmState: warmState)
            DispatchQueue.main.async { completion(snapshot) }
        }.resume()
    }

    /// Identity snapshot returned by ``/api/broker/engine/compatibility``.
    /// Cached on success so callers (UI, supervisor) can read it without
    /// refetching.
    struct DeploymentProfile {
        let previewBackend: String
        let finalBackend: String
        let writerBackend: String
        let writerResidencyPolicy: String

        var matchesBundledProductionProfile: Bool {
            previewBackend == JunoEngineContract.expectedPreviewBackend
                && finalBackend == JunoEngineContract.expectedFinalBackend
                && writerBackend == JunoEngineContract.expectedWriterBackend
                && writerResidencyPolicy == JunoEngineContract.expectedWriterResidencyPolicy
        }
    }

    struct EngineIdentity {
        let runtimeRole: String
        let instanceId: String
        let bundleId: String
        let pid: Int
        let protocolVersion: Int
        let deploymentProfile: DeploymentProfile?
    }

    private static var lastIdentity: EngineIdentity?
    static var lastKnownIdentity: EngineIdentity? { lastIdentity }

    static func ensureCompatible(completion: @escaping (Result<Void, Error>) -> Void) {
        if shouldUseUDS {
            callBrokerRPC(httpMethod: "GET", path: "api/broker/engine/compatibility") { result in
                let out: Result<Void, Error>
                switch result {
                case .failure(let error):
                    out = .failure(error)
                case .success(let response):
                    out = validateCompatibilityObject(response.object)
                }
                completion(out)
            }
            return
        }
        let url = baseURL.appendingPathComponent("api/broker/engine/compatibility")
        URLSession.shared.dataTask(with: url) { data, response, error in
            let result: Result<Void, Error>
            if let error {
                result = .failure(error)
            } else if (response as? HTTPURLResponse)?.statusCode != 200 {
                result = .failure(incompatibleEngineError("voice engine compatibility check failed"))
            } else if let data,
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      (obj["ok"] as? Bool) == true {
                result = validateCompatibilityObject(obj)
            } else {
                result = .failure(incompatibleEngineError("voice engine returned an unreadable compatibility response"))
            }
            DispatchQueue.main.async { completion(result) }
        }.resume()
    }

    private static func incompatibleEngineError(_ message: String) -> NSError {
        NSError(
            domain: "JunoBroker",
            code: -20,
            userInfo: [
                NSLocalizedDescriptionKey: message,
                "JunoErrorCode": "engine_incompatible",
            ]
        )
    }

    private static func validateCompatibilityObject(_ obj: [String: Any]) -> Result<Void, Error> {
        guard (obj["ok"] as? Bool) == true else {
            return .failure(incompatibleEngineError("voice engine returned an unreadable compatibility response"))
        }
        let version = (obj["shell_engine_protocol_version"] as? NSNumber)?.intValue
            ?? (obj["shell_engine_protocol_version"] as? Int)
        let minimum = (obj["min_shell_engine_protocol_version"] as? NSNumber)?.intValue
            ?? (obj["min_shell_engine_protocol_version"] as? Int)
            ?? version
        let expected = JunoEngineContract.shellEngineProtocolVersion
        let role = (obj["runtime_role"] as? String) ?? ""
        let allowNonRuntime = ProcessInfo.processInfo
            .environment["JUNO_DEV_ALLOW_NON_RUNTIME"]?.lowercased()
            == "1"
        guard let version, let minimum, version == expected, minimum <= expected else {
            return .failure(incompatibleEngineError("voice engine needs to be updated or restarted"))
        }
        guard role == JunoEngineContract.requiredRuntimeRole || allowNonRuntime else {
            return .failure(roleMismatchError(detected: role.isEmpty ? "unknown" : role))
        }
        if let profile = deploymentProfile(from: obj),
           !profile.matchesBundledProductionProfile {
            return .failure(profileMismatchError(profile))
        }
        let identity = EngineIdentity(
            runtimeRole: role,
            instanceId: (obj["instance_id"] as? String) ?? "",
            bundleId: (obj["bundle_id"] as? String) ?? "",
            pid: (obj["pid"] as? NSNumber)?.intValue ?? (obj["pid"] as? Int) ?? 0,
            protocolVersion: version,
            deploymentProfile: deploymentProfile(from: obj)
        )
        DispatchQueue.main.async {
            lastIdentity = identity
        }
        return .success(())
    }

    static func deploymentProfile(from obj: [String: Any]) -> DeploymentProfile? {
        guard let raw = obj["deployment_profile"] as? [String: Any] else { return nil }
        let preview = ((raw["preview_backend"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let final = ((raw["final_backend"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let writer = ((raw["writer_backend"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let residency = ((raw["writer_residency_policy"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !preview.isEmpty, !final.isEmpty, !writer.isEmpty, !residency.isEmpty else {
            return nil
        }
        return DeploymentProfile(
            previewBackend: preview,
            finalBackend: final,
            writerBackend: writer,
            writerResidencyPolicy: residency
        )
    }

    static func roleMismatchError(detected: String) -> NSError {
        // Single user-facing string for both detected variants; keeps
        // developer terms (workbench, Ctrl+C, env vars) out of release UI.
        // The detected role is preserved in userInfo for support bundles.
        var userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: JunoEngineErrorCopy.roleMismatchUserMessage,
            "JunoErrorCode": "engine_role_mismatch",
            "JunoDetectedRole": detected,
        ]
        #if DEBUG
        userInfo["JunoDeveloperHint"] = "Detected role=\(detected). Set JUNO_DEV_ALLOW_NON_RUNTIME=1 to attach anyway."
        #endif
        return NSError(
            domain: "JunoBroker",
            code: -21,
            userInfo: userInfo
        )
    }

    static func profileMismatchError(_ profile: DeploymentProfile) -> NSError {
        let detected = [
            "preview=\(profile.previewBackend)",
            "final=\(profile.finalBackend)",
            "writer=\(profile.writerBackend)",
            "writer_residency=\(profile.writerResidencyPolicy)",
        ].joined(separator: ", ")
        return NSError(
            domain: "JunoBroker",
            code: -22,
            userInfo: [
                NSLocalizedDescriptionKey: JunoEngineErrorCopy.profileMismatchUserMessage,
                "JunoErrorCode": "engine_profile_mismatch",
                "JunoDetectedProfile": detected,
            ]
        )
    }

    /// Capability probe result sent back to the caller.
    struct CapabilityResponse {
        let ok: Bool
        let reason: String
        let message: String
        let windowTitle: String?
        let appBundleId: String?
        /// AX role of the focused element from the broker probe (`report.focused_role`).
        let focusedRole: String?
        /// Whether Cmd+V is likely to land in an editable field (derived from `focusedRole`).
        let hasLikelyTextInsertionPoint: Bool
        /// Domain-specific terms derived from the current surface (identifiers, filenames,
        /// lexicon entries). Passed to the local preview/final pipeline so ASR is
        /// biased toward the user's active vocabulary.
        let recognitionHints: [String]
    }

    /// Roles where synthesized Cmd+V from ``juno-paste`` typically inserts text.
    static func axRoleSuggestsPasteSurface(_ role: String?) -> Bool {
        guard let r = role?.trimmingCharacters(in: .whitespacesAndNewlines), !r.isEmpty else {
            return false
        }
        let pasteFriendly: Set<String> = [
            "AXTextField",
            "AXTextArea",
            "AXComboBox",
            "AXSearchField",
            "AXWebArea",
        ]
        return pasteFriendly.contains(r)
    }

    static func checkCapability(completion: @escaping (CapabilityResponse) -> Void) {
        ensureCompatible { compat in
            if case .failure = compat {
                completion(CapabilityResponse(
                    ok: false,
                    reason: "engine_incompatible",
                    message: "Voice engine needs to be updated or restarted.",
                    windowTitle: nil,
                    appBundleId: nil,
                    focusedRole: nil,
                    hasLikelyTextInsertionPoint: false,
                    recognitionHints: []
                ))
                return
            }
            checkCapabilityAfterCompatibility(completion: completion)
        }
    }

    private static func checkCapabilityAfterCompatibility(completion: @escaping (CapabilityResponse) -> Void) {
        if shouldUseUDS {
            callBrokerRPC(httpMethod: "GET", path: "api/broker/surface/capability") { result in
                switch result {
                case .success(let out):
                    completion(capabilityResponseWithLocalFallback(capabilityResponse(from: out.object)))
                case .failure:
                    completion(CapabilityResponse(
                        ok: false,
                        reason: "broker_unreachable",
                        message: "Broker unreachable",
                        windowTitle: nil,
                        appBundleId: nil,
                        focusedRole: nil,
                        hasLikelyTextInsertionPoint: false,
                        recognitionHints: []
                    ))
                }
            }
            return
        }
        let url = baseURL.appendingPathComponent("api/broker/surface/capability")
        URLSession.shared.dataTask(with: url) { data, _, _ in
            let response: CapabilityResponse
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                response = capabilityResponse(from: obj)
            } else {
                response = CapabilityResponse(
                    ok: false,
                    reason: "broker_unreachable",
                    message: "Broker unreachable",
                    windowTitle: nil,
                    appBundleId: nil,
                    focusedRole: nil,
                    hasLikelyTextInsertionPoint: false,
                    recognitionHints: []
                )
            }
            DispatchQueue.main.async {
                completion(capabilityResponseWithLocalFallback(response))
            }
        }.resume()
    }

    private static func capabilityResponse(from obj: [String: Any]) -> CapabilityResponse {
        let ok = (obj["ok"] as? Bool) ?? false
        let reason = (obj["reason"] as? String) ?? "unknown"
        let message = (obj["message"] as? String) ?? ""
        let report = (obj["report"] as? [String: Any]) ?? [:]
        let windowTitle = report["window_title"] as? String
        let bundleId = (report["frontmost_app_bundle_id"] as? String)
            ?? (report["app_bundle_id"] as? String)
        let focusedRole = report["focused_role"] as? String
        let canPasteAtFocus = (report["can_paste_at_focus"] as? Bool)
            ?? (obj["can_paste_at_focus"] as? Bool)
            ?? JunoBroker.axRoleSuggestsPasteSurface(focusedRole)
        let hints = (obj["recognition_hints"] as? [String]) ?? []
        return CapabilityResponse(
            ok: ok,
            reason: reason,
            message: message,
            windowTitle: windowTitle,
            appBundleId: bundleId,
            focusedRole: focusedRole,
            hasLikelyTextInsertionPoint: canPasteAtFocus && reason != "no_text_focus",
            recognitionHints: hints
        )
    }

    private static func capabilityResponseWithLocalFallback(_ response: CapabilityResponse) -> CapabilityResponse {
        let helperFailureReasons: Set<String> = [
            "ax_permission_missing",
            "helper_not_installed",
            "helper_timeout",
        ]
        guard helperFailureReasons.contains(response.reason) else {
            return response
        }
        let local = JunoCapabilitySnapshot.capture()
        let localResponse = capabilityResponse(from: JunoLocalCapability.brokerDecisionObject(from: local))
        if localResponse.reason == "no_text_focus",
           let fallback = pasteCentricFallbackResponse(from: localResponse, original: response) {
            NSLog("Juno: allowing paste for known document app after no_text_focus fallback app=%@", fallback.appBundleId ?? "unknown")
            return fallback
        }
        if localResponse.reason != response.reason {
            NSLog("Juno: using in-process AX capability fallback after broker capability \(response.reason)")
        }
        return localResponse
    }

    private static func pasteCentricFallbackResponse(
        from local: CapabilityResponse,
        original: CapabilityResponse
    ) -> CapabilityResponse? {
        let bundleId = (local.appBundleId ?? original.appBundleId ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard knownPasteCentricFallbackBundleIds.contains(bundleId) else { return nil }
        return CapabilityResponse(
            ok: true,
            reason: "allowed",
            message: "ok",
            windowTitle: local.windowTitle ?? original.windowTitle,
            appBundleId: local.appBundleId ?? original.appBundleId,
            focusedRole: local.focusedRole ?? original.focusedRole,
            hasLikelyTextInsertionPoint: true,
            recognitionHints: local.recognitionHints.isEmpty ? original.recognitionHints : local.recognitionHints
        )
    }

    private static let knownPasteCentricFallbackBundleIds: Set<String> = [
        "com.apple.notes",
        "com.apple.textedit",
    ]

    static func post(
        path: String,
        payload: [String: Any],
        completion: @escaping (Result<Data, Error>) -> Void = { _ in }
    ) {
        ensureCompatible { compat in
            switch compat {
            case .success:
                postAfterCompatibility(path: path, payload: payload, completion: completion)
            case .failure(let err):
                completion(.failure(err))
            }
        }
    }

    private static func postAfterCompatibility(
        path: String,
        payload: [String: Any],
        completion: @escaping (Result<Data, Error>) -> Void
    ) {
        if shouldUseUDS {
            callBrokerRPC(httpMethod: "POST", path: path, payload: payload) { result in
                switch result {
                case .success(let out):
                    let data = (try? JSONSerialization.data(withJSONObject: out.object, options: [])) ?? Data()
                    completion(.success(data))
                case .failure(let error):
                    completion(.failure(error))
                }
            }
            return
        }
        let url = resolveURL(path: path)
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        JunoLocalBrokerAuth.attach(to: &req)
        req.httpBody = try? JSONSerialization.data(withJSONObject: payload, options: [])
        URLSession.shared.dataTask(with: req) { data, response, error in
            let out: Result<Data, Error>
            if let error {
                out = .failure(error)
            } else if let data {
                out = .success(data)
            } else {
                out = .failure(NSError(domain: "JunoBroker", code: -1))
            }
            DispatchQueue.main.async { completion(out) }
        }.resume()
    }

    static func fetchBinary(
        path: String,
        completion: @escaping (Result<Data, Error>) -> Void
    ) {
        ensureCompatible { compat in
            switch compat {
            case .failure(let err):
                completion(.failure(err))
            case .success:
                if shouldUseUDS {
                    callBrokerRPC(httpMethod: "GET", path: path) { result in
                        switch result {
                        case .success(let out):
                            if let data = out.binary {
                                completion(.success(data))
                            } else {
                                completion(.failure(NSError(
                                    domain: "JunoBroker",
                                    code: -40,
                                    userInfo: [NSLocalizedDescriptionKey: "missing binary payload"]
                                )))
                            }
                        case .failure(let error):
                            completion(.failure(error))
                        }
                    }
                    return
                }
                let url = baseURL.appendingPathComponent(path)
                URLSession.shared.dataTask(with: url) { data, _, error in
                    let out: Result<Data, Error>
                    if let error {
                        out = .failure(error)
                    } else if let data {
                        out = .success(data)
                    } else {
                        out = .failure(NSError(domain: "JunoBroker", code: -41))
                    }
                    DispatchQueue.main.async { completion(out) }
                }.resume()
            }
        }
    }

    /// Writer-backed home hero lines. On any failure, use ``JunoHomeGreeting.heroLines()`` in the caller.
    static func fetchHomeGreeting(
        now: Date = Date(),
        calendar: Calendar = .current,
        completion: @escaping (Result<(headline: String, subline: String), Error>) -> Void
    ) {
        let hour = calendar.component(.hour, from: now)
        let weekday = calendar.component(.weekday, from: now)
        let name = JunoHomeGreeting.displayNameForGreeting()
        let payload: [String: Any] = [
            "hour": hour,
            "weekday": weekday,
            "display_name": name,
        ]
        ensureCompatible { compat in
            switch compat {
            case .failure(let err):
                completion(.failure(err))
            case .success:
                if shouldUseUDS {
                    callBrokerRPC(httpMethod: "POST", path: "api/broker/shell/home_greeting", payload: payload) { result in
                        switch result {
                        case .failure(let error):
                            completion(.failure(error))
                        case .success(let out):
                            let obj = out.object
                            guard let ok = obj["ok"] as? Bool, ok,
                                  let hl = obj["headline"] as? String,
                                  let sl = obj["subline"] as? String,
                                  !hl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                  !sl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                                completion(.failure(NSError(
                                    domain: "JunoBroker",
                                    code: -30,
                                    userInfo: [NSLocalizedDescriptionKey: "greeting_unavailable"]
                                )))
                                return
                            }
                            completion(.success((hl.trimmingCharacters(in: .whitespacesAndNewlines), sl.trimmingCharacters(in: .whitespacesAndNewlines))))
                        }
                    }
                    return
                }
                let url = baseURL.appendingPathComponent("api/broker/shell/home_greeting")
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.timeoutInterval = 6
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                JunoLocalBrokerAuth.attach(to: &req)
                req.httpBody = try? JSONSerialization.data(withJSONObject: payload, options: [])
                URLSession.shared.dataTask(with: req) { data, _, error in
                    if let error {
                        DispatchQueue.main.async { completion(.failure(error)) }
                        return
                    }
                    guard let data,
                          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let ok = obj["ok"] as? Bool, ok,
                          let hl = obj["headline"] as? String,
                          let sl = obj["subline"] as? String,
                          !hl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                          !sl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    else {
                        let err = NSError(
                            domain: "JunoBroker",
                            code: -30,
                            userInfo: [NSLocalizedDescriptionKey: "greeting_unavailable"]
                        )
                        DispatchQueue.main.async { completion(.failure(err)) }
                        return
                    }
                    DispatchQueue.main.async { completion(.success((hl.trimmingCharacters(in: .whitespacesAndNewlines), sl.trimmingCharacters(in: .whitespacesAndNewlines)))) }
                }.resume()
            }
        }
    }

    struct TranscribeResponse {
        let transcript: String
        let rawTranscript: String?
        let utteranceId: String?
        let pasteKind: String?
        let noopReason: String?
        let recoverableTranscript: String?
        let degradedWriter: Bool
        let stage: String?
        let transcriptStage: String?
        let transcriptPatch: TranscriptPatchEnvelope?
        let metadata: [String: Any]?
        /// Parsed action requests (notes / reminders) when the utterance
        /// began with a Juno wake-word + action verb. ``nil`` for plain
        /// dictation — feature is purely additive.
        let actions: [JunoActionRequest]?
        /// Backend says this utterance was parsed as an action (paste must
        /// be suppressed because of the action, not because of some other
        /// state-mutation noop). Backwards-compatible: if the field is
        /// absent (older engine), we infer from `actions` non-nil.
        let isAction: Bool
    }

    /// Normal dictation always uses multipart with ``surface_id`` + ``host_hints`` so the broker
    /// does not fall back to ``workbench_dev`` session rules.
    static func transcribeWav(
        url fileURL: URL,
        appBundleId: String? = nil,
        windowTitle: String? = nil,
        utteranceId: String? = nil,
        frozenContext: [String: Any]? = nil,
        hostHints: [String: Any]? = nil,
        shellTimeline: [String: Any]? = nil,
        surfaceId: String = "mac_overlay",
        transcriptStage: String = "final_delivery",
        sessionContextTape: [String: Any]? = nil,
        transcriptHint: String? = nil,
        languageMode: String? = nil,
        completion: @escaping (Result<TranscribeResponse, Error>) -> Void
    ) {
        guard let wavData = try? Data(contentsOf: fileURL) else {
            let err = NSError(domain: "JunoBroker", code: -2,
                              userInfo: [NSLocalizedDescriptionKey: "could not read \(fileURL.path)"])
            DispatchQueue.main.async { completion(.failure(err)) }
            return
        }
        transcribeWav(
            wavData: wavData,
            appBundleId: appBundleId,
            windowTitle: windowTitle,
            utteranceId: utteranceId,
            frozenContext: frozenContext,
            hostHints: hostHints,
            shellTimeline: shellTimeline,
            surfaceId: surfaceId,
            transcriptStage: transcriptStage,
            sessionContextTape: sessionContextTape,
            transcriptHint: transcriptHint,
            languageMode: languageMode,
            completion: completion
        )
    }

    static func transcribeWav(
        wavData: Data,
        appBundleId: String? = nil,
        windowTitle: String? = nil,
        utteranceId: String? = nil,
        frozenContext: [String: Any]? = nil,
        hostHints: [String: Any]? = nil,
        shellTimeline: [String: Any]? = nil,
        surfaceId: String = "mac_overlay",
        transcriptStage: String = "final_delivery",
        sessionContextTape: [String: Any]? = nil,
        transcriptHint: String? = nil,
        languageMode: String? = nil,
        completion: @escaping (Result<TranscribeResponse, Error>) -> Void
    ) {
        ensureCompatible { compat in
            switch compat {
            case .success:
                transcribeWavAfterCompatibility(
                    wavData: wavData,
                    appBundleId: appBundleId,
                    windowTitle: windowTitle,
                    utteranceId: utteranceId,
                    frozenContext: frozenContext,
                    hostHints: hostHints,
                    shellTimeline: shellTimeline,
                    surfaceId: surfaceId,
                    transcriptStage: transcriptStage,
                    sessionContextTape: sessionContextTape,
                    transcriptHint: transcriptHint,
                    languageMode: languageMode,
                    completion: completion
                )
            case .failure(let err):
                completion(.failure(err))
            }
        }
    }

    static func correctLiveTranscript(
        visibleText: String,
        appBundleId: String? = nil,
        windowTitle: String? = nil,
        utteranceId: String? = nil,
        frozenContext: [String: Any]? = nil,
        hostHints: [String: Any]? = nil,
        shellTimeline: [String: Any]? = nil,
        surfaceId: String = "mac_overlay",
        sessionContextTape: [String: Any]? = nil,
        languageMode: String? = nil,
        completion: @escaping (Result<TranscribeResponse, Error>) -> Void
    ) {
        ensureCompatible { compat in
            switch compat {
            case .success:
                var payload: [String: Any] = [
                    "surface_id": surfaceId,
                    "visible_text": visibleText,
                    "transcript_hint": visibleText,
                    "pause_sensitivity_seconds": JunoUserDefaults.pauseSensitivitySeconds,
                ]
                if let appBundleId, !appBundleId.isEmpty { payload["app_bundle_id"] = appBundleId }
                if let windowTitle, !windowTitle.isEmpty { payload["window_title_hint"] = windowTitle }
                if let utteranceId, !utteranceId.isEmpty { payload["utterance_id"] = utteranceId }
                if let frozenContext, !frozenContext.isEmpty { payload["frozen_context"] = frozenContext }
                if let hostHints, !hostHints.isEmpty { payload["host_hints"] = hostHints }
                if let shellTimeline, !shellTimeline.isEmpty { payload["shell_timeline"] = shellTimeline }
                if let sessionContextTape, !sessionContextTape.isEmpty { payload["session_context_tape"] = sessionContextTape }
                if let languageMode, !languageMode.isEmpty { payload["language"] = languageMode }

                if shouldUseUDS {
                    callBrokerRPC(
                        httpMethod: "POST",
                        path: "api/broker/dictation/live_correct",
                        payload: payload
                    ) { result in
                        switch result {
                        case .failure(let error):
                            completion(.failure(error))
                        case .success(let out):
                            completion(parseTranscribeResponse(out.object))
                        }
                    }
                    return
                }

                var req = URLRequest(url: baseURL.appendingPathComponent("api/broker/dictation/live_correct"))
                req.httpMethod = "POST"
                req.timeoutInterval = 10
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                JunoLocalBrokerAuth.attach(to: &req)
                req.httpBody = try? JSONSerialization.data(withJSONObject: payload, options: [])
                URLSession.shared.dataTask(with: req) { data, _, error in
                    let result: Result<TranscribeResponse, Error>
                    if let error {
                        result = .failure(error)
                    } else if let data,
                              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        result = parseTranscribeResponse(obj)
                    } else {
                        result = .failure(NSError(domain: "JunoBroker", code: -3,
                                                  userInfo: [NSLocalizedDescriptionKey: "bad response"]))
                    }
                    DispatchQueue.main.async { completion(result) }
                }.resume()
            case .failure(let err):
                completion(.failure(err))
            }
        }
    }

    private static func transcribeWavAfterCompatibility(
        wavData: Data,
        appBundleId: String? = nil,
        windowTitle: String? = nil,
        utteranceId: String? = nil,
        frozenContext: [String: Any]? = nil,
        hostHints: [String: Any]? = nil,
        shellTimeline: [String: Any]? = nil,
        surfaceId: String = "mac_overlay",
        transcriptStage: String = "final_delivery",
        sessionContextTape: [String: Any]? = nil,
        transcriptHint: String? = nil,
        languageMode: String? = nil,
        completion: @escaping (Result<TranscribeResponse, Error>) -> Void
    ) {
        if shouldUseUDS {
            var payload: [String: Any] = [
                "surface_id": surfaceId,
            ]
            if let appBundleId, !appBundleId.isEmpty { payload["app_bundle_id"] = appBundleId }
            if let windowTitle, !windowTitle.isEmpty { payload["window_title_hint"] = windowTitle }
            if let utteranceId, !utteranceId.isEmpty { payload["utterance_id"] = utteranceId }
            if let frozenContext, !frozenContext.isEmpty { payload["frozen_context"] = frozenContext }
            if let hostHints, !hostHints.isEmpty { payload["host_hints"] = hostHints }
            if let shellTimeline, !shellTimeline.isEmpty { payload["shell_timeline"] = shellTimeline }
            if !transcriptStage.isEmpty { payload["transcript_stage"] = transcriptStage }
            if let sessionContextTape, !sessionContextTape.isEmpty { payload["session_context_tape"] = sessionContextTape }
            if let transcriptHint, !transcriptHint.isEmpty { payload["transcript_hint"] = transcriptHint }
            if let languageMode, !languageMode.isEmpty { payload["language"] = languageMode }
            payload["pause_sensitivity_seconds"] = JunoUserDefaults.pauseSensitivitySeconds
            callBrokerRPC(
                httpMethod: "POST",
                path: "api/broker/dictation/ingest_wav",
                payload: payload,
                binary: wavData
            ) { result in
                switch result {
                case .failure(let error):
                    completion(.failure(error))
                case .success(let out):
                    completion(parseTranscribeResponse(out.object))
                }
            }
            return
        }
        var components = URLComponents(
            url: baseURL.appendingPathComponent("api/broker/dictation/ingest_wav"),
            resolvingAgainstBaseURL: false
        )
        var qs: [URLQueryItem] = []
        if let appBundleId, !appBundleId.isEmpty {
            qs.append(URLQueryItem(name: "app_bundle_id", value: appBundleId))
        }
        if let windowTitle, !windowTitle.isEmpty {
            qs.append(URLQueryItem(name: "window_title_hint", value: windowTitle))
        }
        if let utteranceId, !utteranceId.isEmpty {
            qs.append(URLQueryItem(name: "utterance_id", value: utteranceId))
        }
        if !qs.isEmpty { components?.queryItems = qs }
        let finalURL = components?.url ?? baseURL.appendingPathComponent("api/broker/dictation/ingest_wav")

        var req = URLRequest(url: finalURL)
        req.httpMethod = "POST"
        JunoLocalBrokerAuth.attach(to: &req)

        let boundary = "JunoBoundary-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func appendField(name: String, value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append(value.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        func appendJSONField(name: String, json: [String: Any]) {
            guard let d = try? JSONSerialization.data(withJSONObject: json, options: []) else { return }
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: application/json\r\n\r\n".data(using: .utf8)!)
            body.append(d)
            body.append("\r\n".data(using: .utf8)!)
        }

        appendField(name: "surface_id", value: surfaceId)
        if let frozenContext, !frozenContext.isEmpty {
            appendJSONField(name: "frozen_context", json: frozenContext)
        }
        if let hostHints, !hostHints.isEmpty {
            appendJSONField(name: "host_hints", json: hostHints)
        }
        if let shellTimeline, !shellTimeline.isEmpty {
            appendJSONField(name: "shell_timeline", json: shellTimeline)
        }
        appendField(name: "transcript_stage", value: transcriptStage)
        if let sessionContextTape, !sessionContextTape.isEmpty {
            appendJSONField(name: "session_context_tape", json: sessionContextTape)
        }
        if let transcriptHint, !transcriptHint.isEmpty {
            appendField(name: "transcript_hint", value: transcriptHint)
        }
        if let languageMode, !languageMode.isEmpty {
            appendField(name: "language_mode", value: languageMode)
        }
        appendField(
            name: "pause_sensitivity_seconds",
            value: String(format: "%.2f", JunoUserDefaults.pauseSensitivitySeconds)
        )
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"wav\"; filename=\"utterance.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(wavData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        req.httpBody = body

        URLSession.shared.dataTask(with: req) { data, _, error in
            let result: Result<TranscribeResponse, Error>
            if let error {
                result = .failure(error)
            } else if let data,
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                result = parseTranscribeResponse(obj)
            } else {
                result = .failure(NSError(domain: "JunoBroker", code: -3,
                                          userInfo: [NSLocalizedDescriptionKey: "bad response"]))
            }
            DispatchQueue.main.async { completion(result) }
        }.resume()
    }

    private static func parseTranscribeResponse(_ obj: [String: Any]) -> Result<TranscribeResponse, Error> {
        if let ok = obj["ok"] as? Bool, ok == false {
            let errMsg = (obj["error"] as? String) ?? "transcribe_failed"
            let cancelled = (obj["cancelled"] as? Bool) ?? false
            let code = (obj["error_code"] as? String) ?? (cancelled ? "inference_cancelled" : "transcribe_failed")
            return .failure(NSError(domain: "JunoBroker", code: -4,
                                    userInfo: [
                                        NSLocalizedDescriptionKey: errMsg,
                                        "JunoErrorCode": code,
                                        "JunoCancelled": cancelled,
                                    ]))
        }
        if let transcript = obj["transcript"] as? String {
            // Decode the optional `actions` array. Each entry is the loose
            // dict shape produced by the Python broker; `fromBrokerDict`
            // returns nil for malformed entries so a bad row can never
            // break dictation.
            var actions: [JunoActionRequest]? = nil
            if let rawActions = obj["actions"] as? [[String: Any]], !rawActions.isEmpty {
                let decoded = rawActions.compactMap(JunoActionRequest.fromBrokerDict)
                actions = decoded.isEmpty ? nil : decoded
            }
            // Prefer the explicit is_action flag from the engine. Fall back
            // to actions-non-nil for compatibility with older engine builds
            // that don't emit the field yet.
            let backendIsAction = (obj["is_action"] as? Bool) ?? (actions != nil)
            let metadata = obj["metadata"] as? [String: Any]
            let patch = parseTranscriptPatch(from: metadata?["transcript_patch"] ?? obj["transcript_patch"])
            let stage = (obj["stage"] as? String) ?? (obj["transcript_stage"] as? String)
            return .success(TranscribeResponse(
                transcript: transcript,
                rawTranscript: obj["raw_transcript"] as? String,
                utteranceId: obj["utterance_id"] as? String,
                pasteKind: obj["paste_kind"] as? String,
                noopReason: obj["noop_reason"] as? String,
                recoverableTranscript: obj["recoverable_transcript"] as? String,
                degradedWriter: (obj["degraded_writer"] as? Bool) ?? false,
                stage: stage,
                transcriptStage: stage,
                transcriptPatch: patch,
                metadata: metadata,
                actions: actions,
                isAction: backendIsAction
            ))
        }
        return .failure(NSError(domain: "JunoBroker", code: -5,
                                userInfo: [NSLocalizedDescriptionKey: "missing transcript field"]))
    }

    private static func parseTranscriptPatch(from raw: Any?) -> TranscriptPatchEnvelope? {
        guard let raw else { return nil }
        do {
            let data: Data
            if let dict = raw as? [String: Any] {
                data = try JSONSerialization.data(withJSONObject: dict, options: [])
            } else if let str = raw as? String {
                data = Data(str.utf8)
            } else {
                return nil
            }
            return try JSONDecoder().decode(TranscriptPatchEnvelope.self, from: data)
        } catch {
            NSLog("Juno: transcript patch decode failed \(error.localizedDescription)")
            return nil
        }
    }
}

// MARK: - Helper binary resolution

enum HelperBinary {
    static func path(_ name: String) -> String? {
        if let p = bundledMacOSPeer(name) { return p }
        let exe = Bundle.main.executablePath ?? ""
        let sibling = (exe as NSString).deletingLastPathComponent + "/" + name
        if FileManager.default.isExecutableFile(atPath: sibling) {
            return sibling
        }
        if let path = ProcessInfo.processInfo.environment["PATH"] {
            for dir in path.split(separator: ":") {
                let candidate = "\(dir)/\(name)"
                if FileManager.default.isExecutableFile(atPath: candidate) {
                    return candidate
                }
            }
        }
        return nil
    }

    /// `Juno.app/Contents/MacOS/<name>` when running from a packaged app.
    private static func bundledMacOSPeer(_ name: String) -> String? {
        let bundleURL = Bundle.main.bundleURL
        let p = bundleURL.appendingPathComponent("Contents/MacOS/\(name)").path
        if FileManager.default.isExecutableFile(atPath: p) {
            return p
        }
        return nil
    }
}

// MARK: - TCP port selection

/// Picks the first free TCP port in a range by binding+closing a probe
/// socket. Used by the engine bootstrap to avoid colliding with whatever
/// other dev tooling has 8765 (the historical Juno default).
enum TCPPortPicker {
    static func firstFree(in range: ClosedRange<Int>, host: String = "127.0.0.1") -> Int? {
        for port in range {
            if isFree(port: port, host: host) { return port }
        }
        return nil
    }

    private static func isFree(port: Int, host: String) -> Bool {
        var hints = addrinfo()
        hints.ai_family = AF_INET
        hints.ai_socktype = SOCK_STREAM
        var resPtr: UnsafeMutablePointer<addrinfo>? = nil
        let rc = getaddrinfo(host, String(port), &hints, &resPtr)
        guard rc == 0, let info = resPtr else { return false }
        defer { freeaddrinfo(resPtr) }
        let fd = socket(info.pointee.ai_family, info.pointee.ai_socktype, info.pointee.ai_protocol)
        if fd < 0 { return false }
        defer { close(fd) }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        let bound = bind(fd, info.pointee.ai_addr, info.pointee.ai_addrlen) == 0
        return bound
    }
}

// MARK: - Local broker bootstrap (silent, identity-aware)

@MainActor
enum JunoLocalBrokerBootstrap {
    /// Phase 1 contract: only attach to a peer that identifies itself as
    /// the production runtime. If 8765 is taken by anything else (most
    /// commonly a developer's standalone ``python -m
    /// juno_v2.workbench.server``), pick a free port in the same range,
    /// spawn the bundled engine on it, and publish the URL so subsequent
    /// broker calls bypass discovery.
    ///
    /// Phase 2 will replace the TCP scan with a UDS socket at a known
    /// path under ``<support-root>/runtime/``; the spawn/supervision
    /// shape stays the same.
    /// Public hook for the supervisor: idempotently respawn the engine if
    /// it isn't running. Called from ``JunoEngineSupervisor`` on a watchdog
    /// timer so that any death (clean exit, signal, helper-side crash) is
    /// recovered without user intervention.
    static func spawnIfNotRunning() {
        Task { @MainActor in
            if let proc = JunoShellRuntime.shared.brokerProcess, proc.isRunning {
                return
            }
            // Drop any stale identity probe before relaunching so the
            // socket-on-disk doesn't trick us into believing an engine
            // we just buried is still alive.
            if let identity = await probeUDSIdentitySync() {
                reapStaleEngineIfNeeded(identity)
            }
            spawnBundledEngine(socketPath: JunoBroker.engineSocketPath)
        }
    }

    static func ensureRunningIfPossible() {
        Task { @MainActor in
            if JunoBroker.shouldUseUDS {
                if let identity = await probeUDSIdentitySync(),
                   identityMatchesExpectedContract(identity) {
                    NSLog("Juno: attached to existing engine socket instance=\(identity.instanceId)")
                    JunoEngineSupervisor.shared.start()
                    return
                }
                if let identity = await probeUDSIdentitySync() {
                    reapStaleEngineIfNeeded(identity)
                }
                spawnBundledEngine(socketPath: JunoBroker.engineSocketPath)
                JunoEngineSupervisor.shared.start()
                return
            }

            // Explicit HTTP dev override path: preserve old behavior only
            // when the operator asks for a TCP workbench URL.
            let host = "127.0.0.1"
            let scanRange = 8765...8785
            for port in scanRange {
                guard let url = URL(string: "http://\(host):\(port)") else { continue }
                let identity = await probeIdentitySync(at: url)
                if let identity, identityMatchesExpectedContract(identity) {
                    JunoBroker.setResolvedBaseURL(url)
                    return
                }
            }
        }
    }

    private static func identityMatchesExpectedContract(_ identity: JunoBroker.EngineIdentity) -> Bool {
        guard identity.runtimeRole == JunoEngineContract.requiredRuntimeRole else { return false }
        guard identity.protocolVersion == JunoEngineContract.shellEngineProtocolVersion else { return false }
        guard let profile = identity.deploymentProfile else { return false }
        return profile.matchesBundledProductionProfile
    }

    private static func probeUDSIdentitySync() async -> JunoBroker.EngineIdentity? {
        await withCheckedContinuation { (cont: CheckedContinuation<JunoBroker.EngineIdentity?, Never>) in
            JunoBroker.callBrokerRPC(httpMethod: "GET", path: "api/broker/engine/compatibility") { result in
                switch result {
                case .success(let out):
                    let obj = out.object
                    let version = (obj["shell_engine_protocol_version"] as? NSNumber)?.intValue
                        ?? (obj["shell_engine_protocol_version"] as? Int) ?? 0
                    cont.resume(returning: JunoBroker.EngineIdentity(
                        runtimeRole: (obj["runtime_role"] as? String) ?? "",
                        instanceId: (obj["instance_id"] as? String) ?? "",
                        bundleId: (obj["bundle_id"] as? String) ?? "",
                        pid: (obj["pid"] as? NSNumber)?.intValue ?? (obj["pid"] as? Int) ?? 0,
                        protocolVersion: version,
                        deploymentProfile: JunoBroker.deploymentProfile(from: obj)
                    ))
                case .failure:
                    cont.resume(returning: nil)
                }
            }
        }
    }

    private static func reapStaleEngineIfNeeded(_ identity: JunoBroker.EngineIdentity) {
        guard identity.pid > 1 else { return }
        guard identity.runtimeRole == JunoEngineContract.requiredRuntimeRole else { return }
        let expectedBundleId = Bundle.main.bundleIdentifier ?? JunoEngineContract.defaultBundleId
        guard identity.bundleId == expectedBundleId else { return }
        let mismatchedContract = !identityMatchesExpectedContract(identity)
        guard mismatchedContract || !engineProcessExists(pid: identity.pid) else { return }

        NSLog("Juno: terminating stale engine pid=%d instance=%@ bundle=%@", identity.pid, identity.instanceId, identity.bundleId)
        _ = Darwin.kill(pid_t(identity.pid), SIGTERM)
        waitForEngineExit(pid: identity.pid, socketPath: JunoBroker.engineSocketPath)
    }

    private static func engineProcessExists(pid: Int) -> Bool {
        guard pid > 1 else { return false }
        return Darwin.kill(pid_t(pid), 0) == 0 || errno == EPERM
    }

    private static func waitForEngineExit(pid: Int, socketPath: String) {
        let fm = FileManager.default
        for _ in 0..<20 {
            if !engineProcessExists(pid: pid) {
                break
            }
            usleep(100_000)
        }
        if fm.fileExists(atPath: socketPath) && !engineProcessExists(pid: pid) {
            try? fm.removeItem(atPath: socketPath)
        }
    }

    private static func probeIdentitySync(at url: URL) async -> JunoBroker.EngineIdentity? {
        await withCheckedContinuation { (cont: CheckedContinuation<JunoBroker.EngineIdentity?, Never>) in
            JunoBroker.probeIdentity(at: url) { identity in
                cont.resume(returning: identity)
            }
        }
    }

    private static func bootstrapLogURL() -> URL {
        let fm = FileManager.default
        if let lib = fm.urls(for: .libraryDirectory, in: .userDomainMask).first {
            let dir = lib.appendingPathComponent("Logs/Juno", isDirectory: true)
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir.appendingPathComponent("juno-bootstrap-engine.log", isDirectory: false)
        }
        return URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("juno-engine.log")
    }

    /// Spawn the bundled engine helper on the given port. The
    /// ``juno_v2.workbench.server`` repo-fallback that earlier versions
    /// used has been **removed** — that path silently shipped a strict
    /// subset of the engine and was the root cause of the post-PR-28
    /// "models missing / network connection lost" symptom class. The
    /// only acceptable startup paths are now:
    /// - bundled engine inside ``Juno.app/Contents/Resources/engine/``
    /// - repo dev script (``scripts/run_live_fast_v2.sh``) which itself runs
    ///   ``juno_v2.runtime.service`` (the full live stack).
    private static func spawnBundledEngine(socketPath: String) {
        if let p = JunoShellRuntime.shared.brokerProcess, p.isRunning { return }

        let fm = FileManager.default
        let logURL = bootstrapLogURL()
        let proc = Process()
        var env = ProcessInfo.processInfo.environment
        env["JUNO_V2_LOG_LATENCY"] = env["JUNO_V2_LOG_LATENCY"] ?? "0"
        env["JUNO_REQUIRE_LOCAL_BROKER_AUTH"] = env["JUNO_REQUIRE_LOCAL_BROKER_AUTH"] ?? "1"
        env["JUNO_BUNDLE_ID"] = Bundle.main.bundleIdentifier ?? JunoEngineContract.defaultBundleId
        env["JUNO_ENGINE_SOCKET"] = socketPath
        let previewEligibility = JunoPreviewEligibility.current
        env["JUNO_V2_LIVE_CAPTION_ALLOWED"] = previewEligibility.isEligible ? "1" : "0"
        env["JUNO_V2_LIVE_CAPTION_START_ENABLED"] = JunoUserDefaults.hudLiveTranscriptionsEnabled ? "1" : "0"

        if let engineRoot = JunoEngineContract.bundledEngineRoot() {
            let script = engineRoot.appendingPathComponent("run_engine.sh", isDirectory: false)
            guard fm.fileExists(atPath: script.path) else {
                NSLog("Juno: bundled run_engine.sh missing at \(script.path)")
                NotificationCenter.default.post(name: .junoBrokerBootstrapFailed, object: "missing_run_engine_script")
                return
            }
            proc.executableURL = URL(fileURLWithPath: "/bin/bash")
            proc.arguments = [script.path]
            proc.currentDirectoryURL = engineRoot
            proc.environment = env
        } else if let root = JunoRepoPaths.guessRepoRoot() {
            let rootURL = URL(fileURLWithPath: root)
            // Dev-only fallback. The bash launcher (``run_live_*_v2.sh``)
            // computes its own ``ROOT`` from ``BASH_SOURCE`` and ``cd``s
            // there before doing anything, so the spawned process does not
            // need ``proc.currentDirectoryURL`` to be the repo root. We
            // deliberately keep cwd off the repo path here: when the repo
            // lives under ``~/Documents`` (as it commonly does), anchoring
            // a child process there causes macOS to fire the "would like
            // to access files in your Documents folder" TCC prompt with
            // no corresponding permission button in the Juno UI. Shipped
            // builds run from ``Juno.app/Contents/Resources/engine`` and
            // never hit this branch, so this only affects dev.
            proc.currentDirectoryURL = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            let fastScript = rootURL.appendingPathComponent("scripts/run_live_fast_v2.sh").path
            let liveScript = fastScript
            guard fm.isExecutableFile(atPath: liveScript) else {
                NSLog("Juno: no live-engine script in repo at \(root) — refusing to fall back to a non-runtime workbench")
                NotificationCenter.default.post(name: .junoBrokerBootstrapFailed, object: "missing_live_script")
                return
            }
            proc.executableURL = URL(fileURLWithPath: "/bin/bash")
            proc.arguments = [
                liveScript,
                "--engine-socket", socketPath,
            ]
            let existing = env["PYTHONPATH"] ?? ""
            env["PYTHONPATH"] = existing.isEmpty ? root : "\(root):\(existing)"
            proc.environment = env
        } else {
            NotificationCenter.default.post(name: .junoBrokerBootstrapFailed, object: "no_engine_root")
            return
        }

        do {
            fm.createFile(atPath: logURL.path, contents: nil)
            let fh = try FileHandle(forWritingTo: logURL)
            proc.standardOutput = fh
            proc.standardError = fh
        } catch {
            // best-effort log capture
        }

        // The supervisor needs to know when the engine dies so it can
        // respawn promptly and capture a crash log. Without this the
        // shell silently flat-lines: history fails to decode, capability
        // checks return broker_unreachable, every paste hard-blocks.
        // Process.terminationHandler is invoked off the main actor, so
        // hop back to MainActor before touching the supervisor (which is
        // @MainActor) or the runtime singleton's Process slot.
        proc.terminationHandler = { finished in
            let exitStatus = Int(finished.terminationStatus)
            let reasonCode = finished.terminationReason
            let reasonLabel: String
            switch reasonCode {
            case .exit: reasonLabel = "exit"
            case .uncaughtSignal: reasonLabel = "signal"
            @unknown default: reasonLabel = "unknown"
            }
            NSLog("Juno: bundled engine terminated reason=\(reasonLabel) status=\(exitStatus)")
            Task { @MainActor in
                JunoEngineSupervisor.shared.recordEngineExit(
                    reason: reasonLabel,
                    status: exitStatus,
                    bootstrapLog: logURL
                )
            }
        }

        do {
            try proc.run()
            JunoShellRuntime.shared.brokerProcess = proc
            NSLog("Juno: started local voice engine at \(socketPath)")
            Task { @MainActor in
                JunoEngineSupervisor.shared.recordEngineLaunched()
            }
        } catch {
            NSLog("Juno: failed to start local voice engine: \(error.localizedDescription)")
            NotificationCenter.default.post(name: .junoBrokerBootstrapFailed, object: error.localizedDescription)
            let reason = error.localizedDescription
            Task { @MainActor in
                JunoEngineSupervisor.shared.recordSpawnFailed(reason: reason)
            }
        }
    }
}

// MARK: - Engine supervisor

/// Watches the bundled engine and respawns it if it dies. Without this the
/// app silently flat-lines after any engine crash: the History view shows
/// "data couldn't be read because it is missing" (decode of an empty
/// response), every dictation hard-blocks on capability check, and paste
/// never fires. The supervisor:
///
/// 1. Pings the UDS ``/healthz`` route every ~3s.
/// 2. If 3 consecutive pings fail (or ``Process.terminationHandler`` says
///    the engine exited) it tears down any stale socket and respawns,
///    with exponential backoff capped at 30 s so a permanently-broken
///    engine doesn't pin the CPU.
/// 3. Publishes ``State`` so the UI can swap raw decode errors for
///    "Voice engine offline — reconnecting".
/// 4. On non-zero exit, snapshots the tail of ``bundled-engine.log`` into
///    a per-crash file under ``~/Library/Logs/Juno/`` so the next time the
///    engine dies during a feature like reminders we have a real trace.
@MainActor
final class JunoEngineSupervisor {
    enum State: Equatable {
        case starting
        case online
        case offline(reason: String)
        case restarting(attempt: Int)
    }

    static let shared = JunoEngineSupervisor()

    private(set) var state: State = .starting {
        didSet {
            guard oldValue != state else { return }
            NSLog("Juno: engine supervisor state \(oldValue) -> \(state)")
            NotificationCenter.default.post(
                name: .junoEngineSupervisorStateChanged,
                object: state
            )
            // Warm preview ASR the first time the engine reaches online.
            // Qwen writer stays on-demand; normal dictation commits should
            // not pay or keep its memory unless an explicit LLM rewrite/action
            // needs it.
            if case .online = state, !Self.didWarmPreview {
                Self.didWarmPreview = true
                JunoBroker.getJSON(path: "api/broker/preview/warm") { _ in
                    NSLog("Juno: preview warm requested on engine-online")
                }
            }
            // Drain any action-result posts that exhausted their retry
            // budget while the broker was unreachable. Each entry gets a
            // fresh 3-attempt chain; entries that still fail roll back
            // onto disk for the next online tick.
            if case .online = state {
                JunoActionPostBacklog.shared.drain()
            }
        }
    }
    private static var didWarmPreview = false

    private var pingTimer: Timer?
    private var consecutiveFailures = 0
    private var nextRespawnDelaySeconds: TimeInterval = 0
    private var respawnInFlight = false
    private var lastSpawnAt: Date?
    private var attemptCount = 0
    private let pingIntervalSeconds: TimeInterval = 3.0
    // Cold launch must never get trapped in a respawn loop while Python/MLX
    // binds the final ASR model or background prewarm queues. The engine now
    // publishes health before live-corrector/writer warmup, but keep a wider
    // window for slow first launches and cache verification.
    private let failureThreshold = 20
    private let maxBackoffSeconds: TimeInterval = 30.0

    private init() {}

    func start() {
        guard pingTimer == nil else { return }
        pingTimer = Timer.scheduledTimer(withTimeInterval: pingIntervalSeconds, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        // Run an immediate tick so we don't wait 3 s for the first probe.
        Task { @MainActor in self.tick() }
    }

    /// Invalidate the ping timer so it stops firing. Called from
    /// ``applicationWillTerminate`` to give the supervisor a clean
    /// shutdown — without this the timer kept firing during app
    /// teardown, opening one last UDS socket against an engine that
    /// was about to be SIGTERM'd. Safe to call multiple times.
    func stop() {
        pingTimer?.invalidate()
        pingTimer = nil
    }

    var isOnline: Bool {
        if case .online = state { return true }
        return false
    }

    func recordEngineLaunched() {
        lastSpawnAt = Date()
        attemptCount += 1
        respawnInFlight = false
        if case .online = state {
            // already healthy — nothing to advertise
        } else {
            state = .starting
        }
    }

    func recordEngineExit(reason: String, status: Int, bootstrapLog: URL) {
        // Snapshot the tail of the engine log so the next dictation regression
        // has a concrete trace to read instead of a buried needle in the
        // 4 000-line steady-state log.
        snapshotCrashLog(reason: reason, status: status, bootstrapLog: bootstrapLog)
        JunoShellRuntime.shared.brokerProcess = nil
        if case .restarting = state {
            // already mid-respawn
        } else {
            state = .offline(reason: "engine_exit:\(reason):\(status)")
        }
        scheduleRespawn(immediate: status != 0 && reason == "exit")
    }

    func recordSpawnFailed(reason: String) {
        state = .offline(reason: "spawn_failed:\(reason)")
        scheduleRespawn(immediate: false)
    }

    private func tick() {
        // The process slot can legitimately be nil — at app launch we
        // attach to an externally-running engine (e.g. an orphan from a
        // prior Juno session that we successfully reaped, or a dev
        // ``run_live`` script). In that case ``brokerProcess`` stays
        // nil even though the engine is healthy. Always confirm via a
        // socket health ping before deciding to respawn — otherwise we
        // race a perfectly good engine and end up with two processes
        // bound to the same UDS path, which causes intermittent empty
        // responses (observed: ``/api/broker/history`` returning ``{}``
        // when both engines were alive).
        let procRunning = JunoShellRuntime.shared.brokerProcess?.isRunning == true
        pingHealth { [weak self] ok in
            guard let self else { return }
            if ok {
                self.consecutiveFailures = 0
                self.nextRespawnDelaySeconds = 0
                if self.state != .online {
                    self.state = .online
                }
                return
            }
            self.consecutiveFailures += 1
            // Two ways to reach respawn:
            //   * we own the process slot AND health flat-lined for the
            //     full failure window;
            //   * we never owned the slot AND health is still failing —
            //     no engine to attach to, spawn one.
            let exhaustedFailures = self.consecutiveFailures >= self.failureThreshold
            let noEngineToAttach = !procRunning
            if exhaustedFailures || (noEngineToAttach && self.consecutiveFailures >= 2) {
                NSLog("Juno: engine health failed \(self.consecutiveFailures) times (procRunning=\(procRunning)) — respawning")
                if case .online = self.state {
                    self.state = .offline(reason: procRunning ? "health_timeout" : "no_engine")
                }
                if procRunning {
                    self.forceRespawn()
                } else {
                    self.scheduleRespawn(immediate: true)
                }
            }
        }
    }

    private func pingHealth(completion: @escaping (Bool) -> Void) {
        // Use the same RPC path the rest of the app uses so we test the
        // exact code path that history/capability rely on, not just a
        // socket connect.
        JunoBroker.callBrokerRPC(httpMethod: "GET", path: "healthz") { result in
            switch result {
            case .success(let out):
                let ok = (out.object["ok"] as? Bool) ?? true
                completion(ok)
            case .failure:
                completion(false)
            }
        }
    }

    private func forceRespawn() {
        if let proc = JunoShellRuntime.shared.brokerProcess, proc.isRunning {
            // Engine still has a process but isn't answering — nuke it so
            // the terminationHandler fires and we can respawn cleanly.
            proc.terminate()
            // Give it a moment, then SIGKILL if it didn't exit.
            DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) {
                if proc.isRunning {
                    kill(proc.processIdentifier, SIGKILL)
                }
            }
        }
        JunoShellRuntime.shared.brokerProcess = nil
        scheduleRespawn(immediate: true)
    }

    private func scheduleRespawn(immediate: Bool) {
        guard !respawnInFlight else { return }
        respawnInFlight = true
        let delay = immediate ? max(0.2, min(nextRespawnDelaySeconds, 1.0)) : nextRespawnDelaySeconds
        let attempt = attemptCount + 1
        state = .restarting(attempt: attempt)
        nextRespawnDelaySeconds = min(maxBackoffSeconds, max(2.0, nextRespawnDelaySeconds * 2))
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            // Reap stale socket if it's still on disk.
            let socketPath = JunoBroker.engineSocketPath
            if FileManager.default.fileExists(atPath: socketPath),
               JunoShellRuntime.shared.brokerProcess?.isRunning != true {
                try? FileManager.default.removeItem(atPath: socketPath)
            }
            self.consecutiveFailures = 0
            JunoLocalBrokerBootstrap.spawnIfNotRunning()
            // Clear the in-flight flag after spawn returns; recordEngineLaunched
            // / recordSpawnFailed will reset state appropriately.
            self.respawnInFlight = false
        }
    }

    private func snapshotCrashLog(reason: String, status: Int, bootstrapLog: URL) {
        let fm = FileManager.default
        guard let logsDir = fm.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/Juno", isDirectory: true) else { return }
        try? fm.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let dest = logsDir.appendingPathComponent("engine-crash-\(ts).log", isDirectory: false)

        // Tail bundled-engine.log (the file proc.standardOutput points at)
        // so we can see what the engine printed leading up to its death.
        let tail = (try? String(contentsOf: bootstrapLog, encoding: .utf8))
            .map { Self.tailLines($0, max: 400) } ?? ""

        let header = """
        Juno engine crash snapshot
        timestamp: \(ts)
        reason: \(reason)
        exit_status: \(status)
        bootstrap_log: \(bootstrapLog.path)


        """
        try? (header + tail).write(to: dest, atomically: true, encoding: .utf8)
    }

    private static func tailLines(_ s: String, max n: Int) -> String {
        let lines = s.split(separator: "\n", omittingEmptySubsequences: false)
        if lines.count <= n { return s }
        return lines.suffix(n).joined(separator: "\n")
    }
}

// MARK: - Clipboard + paste

struct PasteboardSnapshot {
    let items: [[String: Data]]

    static func capture() -> PasteboardSnapshot {
        let pb = NSPasteboard.general
        var snap: [[String: Data]] = []
        for item in pb.pasteboardItems ?? [] {
            var payload: [String: Data] = [:]
            for type in item.types {
                if let data = item.data(forType: type) {
                    payload[type.rawValue] = data
                }
            }
            if !payload.isEmpty {
                snap.append(payload)
            }
        }
        return PasteboardSnapshot(items: snap)
    }

    func restore() {
        let pb = NSPasteboard.general
        pb.clearContents()
        let restored: [NSPasteboardItem] = items.map { dict in
            let item = NSPasteboardItem()
            for (type, data) in dict {
                item.setData(data, forType: NSPasteboard.PasteboardType(type))
            }
            return item
        }
        if !restored.isEmpty {
            pb.writeObjects(restored)
        }
    }
}

enum Clipboard {
    /// PID of the frontmost application captured immediately before
    /// the most recent ``juno-paste`` invocation. ``finalizeDictationSession``
    /// reads this and compares it against the recorded ``targetPid`` so
    /// the broker's ``insertion_committed`` event can flag focus drift —
    /// the case where the user pressed the dictation hotkey in app A
    /// but switched to app B before the paste fired, sending Cmd+V to
    /// the wrong destination.
    ///
    /// Race-safe enough for a diagnostic: ``Clipboard.pasteAtCursor`` is
    /// invoked from the main queue (or the broker-on-pause partial paste
    /// flow which serializes via ``DispatchQueue.main``); concurrent
    /// pastes from genuinely overlapping flows would be a different bug
    /// and don't matter for this drift signal.
    static var lastPasteFrontmostPid: pid_t?

    static func writeString(_ s: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(s, forType: .string)
    }

    @discardableResult
    static func pasteAtCursor() -> Bool {
        if JunoLocalCapability.processHasAccessibilityTrust() {
            Self.lastPasteFrontmostPid = NSWorkspace.shared.frontmostApplication?.processIdentifier
            return pasteAtCursorInProcess()
        }

        guard let bin = HelperBinary.path("juno-paste") else {
            NSLog("Juno: juno-paste binary not found on PATH")
            return pasteAtCursorInProcess()
        }
        // Capture the frontmost PID *just before* posting Cmd+V so the
        // diagnostic reflects the actual receiver, not the original
        // hotkey-press target. juno-paste posts via cgSessionEventTap,
        // which delivers to the frontmost app at the time of the
        // CGEvent.post — same moment we're sampling here.
        Self.lastPasteFrontmostPid = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: bin)
        do {
            try proc.run()
            proc.waitUntilExit()
            if proc.terminationStatus == 0 {
                return true
            }
            NSLog("Juno: juno-paste exited \(proc.terminationStatus); trying capability paste fallback")
            if pasteViaCapabilityHelper() {
                return true
            }
            NSLog("Juno: capability paste fallback failed; trying in-process paste fallback")
            return pasteAtCursorInProcess()
        } catch {
            NSLog("Juno: juno-paste launch failed: \(error.localizedDescription)")
            if pasteViaCapabilityHelper() {
                return true
            }
            return pasteAtCursorInProcess()
        }
    }

    private static func pasteViaCapabilityHelper() -> Bool {
        guard let bin = HelperBinary.path("juno-capability") else {
            NSLog("Juno: juno-capability binary not found for paste fallback")
            return false
        }
        Self.lastPasteFrontmostPid = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: bin)
        proc.arguments = ["--paste"]
        let err = Pipe()
        proc.standardError = err
        do {
            try proc.run()
            proc.waitUntilExit()
        } catch {
            NSLog("Juno: capability paste fallback launch failed: \(error.localizedDescription)")
            return false
        }
        if proc.terminationStatus == 0 {
            NSLog("Juno: capability paste fallback succeeded")
            return true
        }
        let data = err.fileHandleForReading.readDataToEndOfFile()
        let msg = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        NSLog("Juno: capability paste fallback exited \(proc.terminationStatus)%@", msg.isEmpty ? "" : ": \(msg)")
        return false
    }

    private static func pasteAtCursorInProcess() -> Bool {
        guard JunoLocalCapability.processHasAccessibilityTrust() else {
            NSLog("Juno: in-process paste fallback blocked; Accessibility is not trusted")
            return false
        }
        let vKey: CGKeyCode = 0x09
        let source = CGEventSource(stateID: .privateState)
        guard
            let down = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: true),
            let up = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: false)
        else {
            NSLog("Juno: in-process paste fallback could not create CGEvents")
            return false
        }
        down.flags = .maskCommand
        up.flags = .maskCommand

        down.post(tap: .cgSessionEventTap)
        usleep(8_000)
        up.post(tap: .cgSessionEventTap)
        // Return "posted", not "landed". ``observedUndoSafePaste`` owns
        // read-back verification where available and deliberately assumes
        // success for custom/Electron fields that expose no reliable AX value.
        usleep(20_000)
        return true
    }

    @discardableResult
    static func undoSafePaste(_ transcript: String, restoreAfterMs: Int = 400) -> Bool {
        let snapshot = PasteboardSnapshot.capture()
        writeString(transcript)
        let ok = pasteAtCursor()
        let delay = DispatchTime.now() + .milliseconds(restoreAfterMs)
        DispatchQueue.main.asyncAfter(deadline: delay) {
            snapshot.restore()
        }
        return ok
    }

    /// Deletes the last ``n`` characters in the focused field by synthesizing
    /// Backspace key events. Retained for rare non-dictation flows; broker-on-pause
    /// avoids mid-session bulk deletes (see ``DictationController.applyBrokerTranscript``).
    ///
    /// Caveats:
    ///   * Counts grapheme clusters (Swift's ``String.count``); macOS
    ///     Backspace also operates on grapheme clusters, so the two
    ///     align for emoji + composed characters.
    ///   * If the user typed or moved the cursor between the partial
    ///     pastes and the broker response, the wrong characters get
    ///     deleted. Callers must validate the target span first.
    ///   * Spread across the call so the focused app processes events
    ///     in order. 1.5 ms per event keeps total time under ~750 ms
    ///     for the cap of 500 chars.
    ///   * Uses ``CGEventSource(stateID: .privateState)`` so the
    ///     synthetic Backspaces' modifier state is INDEPENDENT of the
    ///     physical keyboard. Critical for broker-on-pause: the user
    ///     is typically still holding the dictation hotkey (often
    ///     Option) when these events post. With the default
    ///     ``combinedSessionState`` source, the OS merged the physical
    ///     Option with each Backspace and the receiver saw
    ///     ``Option+Backspace`` — which on macOS deletes a whole
    ///     **word** rather than a character. ``deleteLastNCharacters(150)``
    ///     became "delete 150 words" which ate far past the pasted text
    ///     into surrounding content. Same modifier-leak as
    ///     ``juno-paste`` (fixed in commit fbd703d).
    static func deleteLastNCharacters(_ n: Int) {
        guard n > 0 else { return }
        let src = CGEventSource(stateID: .privateState)
        let backspaceKey: CGKeyCode = 0x33
        for _ in 0..<n {
            if let down = CGEvent(keyboardEventSource: src, virtualKey: backspaceKey, keyDown: true) {
                down.flags = []  // explicitly clear any modifier — only Backspace
                down.post(tap: .cghidEventTap)
            }
            if let up = CGEvent(keyboardEventSource: src, virtualKey: backspaceKey, keyDown: false) {
                up.flags = []
                up.post(tap: .cghidEventTap)
            }
            usleep(1500)
        }
    }
}

// MARK: - Mic capture (press-to-talk)

/// Single-pole IIR high-pass that removes DC offset and low-frequency
/// content from the captured PCM before it reaches RMS measurement, the
/// live preview streamer, and the final-WAV accumulator.
///
/// Cutoff ≈ (1 − R)·sampleRate/(2π), so R = 0.969 at 16 kHz gives ~80 Hz.
/// This is the industry-standard speech high-pass cutoff (telephony, ASR
/// preprocessing): low enough to leave the typical adult voice fundamental
/// range (~85 Hz male, ~165 Hz female) and all the speech formants (300–
/// 3000 Hz) intact, high enough to attenuate mains hum (50/60 Hz), HVAC
/// drone, and the mic-handling thumps that push the adaptive ambient-noise
/// estimate up and starve quiet real speech of the RMS threshold it needs
/// to register. Rolloff is 6 dB/octave so even a deep bass voice with a
/// fundamental near 80 Hz loses no more than a few dB at its lowest tone
/// while the energy above 160 Hz (where intelligibility lives) is
/// effectively untouched.
///
/// One multiply-add per sample. Runs in the audio thread without
/// allocation; state is two floats per recorder instance.
///
/// Skipped when Apple's ``setVoiceProcessingEnabled(true)`` AU is on —
/// that AU already includes noise suppression and AGC, double-filtering
/// only adds artifacts.
struct AudioHighPassFilter {
    static let coefficient: Float = 0.969
    private var prevIn: Float = 0
    private var prevOut: Float = 0

    mutating func reset() {
        prevIn = 0
        prevOut = 0
    }

    mutating func process(_ samples: UnsafeMutablePointer<Float>, frameCount: Int) {
        guard frameCount > 0 else { return }
        var pIn = prevIn
        var pOut = prevOut
        let r = AudioHighPassFilter.coefficient
        for i in 0..<frameCount {
            let x = samples[i]
            let y = x - pIn + r * pOut
            samples[i] = y
            pIn = x
            pOut = y
        }
        prevIn = pIn
        prevOut = pOut
    }
}

final class DictationRecorder {
    private var engine: AVAudioEngine?
    private var file: AVAudioFile?
    private var destURL: URL?
    private var voiceProcessingEnabled: Bool = false
    private var diagTapCount: Int = 0
    /// ``AVAudioConverter`` is not thread-safe; the tap runs on the audio thread.
    private let converterLock = NSLock()
    /// High-pass filter applied to converted 16 kHz mono float audio before
    /// it's handed to the WAV file writer, the silent-PCM diagnostic, and the
    /// per-buffer callback. Mutated only on the tap thread; the lock above
    /// already covers concurrent access to the converter.
    private var highPass = AudioHighPassFilter()
    /// Silent-PCM diagnostic: counts consecutive zero-amplitude samples
    /// since session start, used to detect the macOS audio-HAL silence
    /// failure mode where ``setVoiceProcessingEnabled(true)`` (or a
    /// muted/wrong input device) produces all-zero PCM frames. We've
    /// hit this twice — once on a Mac14,5 in our own QA, once in
    /// the example user's 2026-04-29 support bundle — and the resulting whisper
    /// hallucinations were the user-visible symptom both times. Logging
    /// the silence directly makes future support bundles diagnose this
    /// instantly without round-tripping a WAV through the broker.
    /// Reset to zero per ``start`` call.
    private var silentFrameRun: Int = 0
    private var silenceLogged: Bool = false
    /// One full second of all-zero PCM @ 16 kHz mono. Anything shorter
    /// could be an inter-utterance pause; one continuous second since
    /// session start is decisively "no audio at all".
    private static let silenceFrameThreshold: Int = 16_000

    /// Start recording. The optional ``bufferCallback`` receives **16 kHz mono
    /// float** buffers in the same shape used by the local preview/final ASR path.
    func start(bufferCallback: ((AVAudioPCMBuffer) -> Void)? = nil) throws -> URL {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
            "juno-\(UUID().uuidString).wav"
        )
        destURL = tmp

        let engine = AVAudioEngine()
        let input = engine.inputNode
        voiceProcessingEnabled = false
        silentFrameRun = 0
        silenceLogged = false
        highPass.reset()
        do {
            // Best-practice mic front-end: noise suppression + AGC (and echo cancel when relevant).
            // Must be toggled while the engine is stopped (fresh engine is stopped here).
            if JunoUserDefaults.micVoiceProcessingEnabled {
                try input.setVoiceProcessingEnabled(true)
                voiceProcessingEnabled = input.isVoiceProcessingEnabled
            }
        } catch {
            NSLog("Juno: voice processing unavailable (\(error.localizedDescription)) — continuing without it")
        }
        // Hardware format (e.g. 44.1 kHz / 48 kHz, stereo, float32).
        let hwFmt = input.outputFormat(forBus: 0)

        // The input node can report a degenerate format (0 Hz / 0 channels)
        // when there is no usable input device — e.g. right after sleep/wake,
        // during a device hot-swap race, or when the default input was yanked.
        // Installing a tap with such a format makes AVFoundation raise an
        // *Objective-C* `NSException` from deep inside `installTapOnBus`, which
        // Swift's `do`/`catch` cannot intercept — it tears straight down to
        // `abort()` (see the 1.0.6 crash on a Mac16,11). Reject it up front and
        // surface a recoverable Swift error the caller already handles.
        guard hwFmt.sampleRate > 0, hwFmt.channelCount > 0 else {
            throw NSError(
                domain: "JunoDictationRecorder",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No microphone is available right now. Check System Settings → Sound → Input, then try again."]
            )
        }

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: tmp, settings: settings)
        // processingFormat is float32 @ 16 kHz, mono — what write(from:) expects.
        let wFmt  = file.processingFormat
        let ratio = wFmt.sampleRate / hwFmt.sampleRate
        // AVAudioConverter handles sample-rate conversion + channel downmix.
        guard let converter = AVAudioConverter(from: hwFmt, to: wFmt) else {
            throw NSError(
                domain: "JunoDictationRecorder",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Could not create audio converter for this microphone format."]
            )
        }

        // Wrap the tap install in the ObjC-exception guard: even with a
        // validated format above, AVFoundation can still raise (e.g. the device
        // changes between the format read and the install). Convert any such
        // NSException into a Swift error so we degrade instead of aborting.
        if let installError = JunoCatchNSException({
        input.installTap(onBus: 0, bufferSize: 4096, format: hwFmt) { [weak file, weak self] buf, _ in
            guard let file, let recorder = self else { return }
            let outLen = AVAudioFrameCount((Double(buf.frameLength) * ratio).rounded(.up))
            guard let out = AVAudioPCMBuffer(pcmFormat: wFmt, frameCapacity: outLen) else { return }

            recorder.converterLock.lock()
            defer { recorder.converterLock.unlock() }

            var converterError: NSError?
            var inputConsumedForThisTap = false

            // AVAudioConverter may call its input supplier multiple times for a single
            // ``convert(to:)``. Each hardware tap must behave like a single finite input packet:
            // supply .haveData once, then .noData until the next tap.
            let status = converter.convert(to: out, error: &converterError) { _, outStatus in
                if inputConsumedForThisTap {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                inputConsumedForThisTap = true
                outStatus.pointee = .haveData
                return buf
            }

            if let err = converterError {
                // Swallowing errors here makes the HUD look "alive" while recording silence.
                // Log once per session burst so operators can correlate with broken waveforms.
                recorder.diagTapCount &+= 1
                if recorder.diagTapCount <= 3 {
                    NSLog(
                        "Juno: AVAudioConverter error (status=%ld): %@",
                        status.rawValue,
                        err.localizedDescription
                    )
                }
                return
            }

            if out.frameLength == 0 {
                recorder.diagTapCount &+= 1
                if recorder.diagTapCount <= 3 {
                    NSLog(
                        "Juno: AVAudioConverter produced 0 frames (in=%u outCap=%u hw=%.0f/%u → wFmt=%.0f/%u)",
                        buf.frameLength,
                        out.frameCapacity,
                        hwFmt.sampleRate,
                        hwFmt.channelCount,
                        wFmt.sampleRate,
                        wFmt.channelCount
                    )
                }
                return
            }

            // Classical DSP at capture: removes DC and ~80 Hz-and-below
            // rumble in place on the converted float buffer. Skipped when
            // Apple's voice-processing AU is enabled (it provides its own
            // noise suppression + AGC; chaining both adds artifacts).
            if !recorder.voiceProcessingEnabled, let chData = out.floatChannelData {
                recorder.highPass.process(chData[0], frameCount: Int(out.frameLength))
            }

            do {
                try file.write(from: out)
            } catch {
                recorder.diagTapCount &+= 1
                if recorder.diagTapCount <= 3 {
                    NSLog("Juno: AVAudioFile write failed: %@", error.localizedDescription)
                }
                return
            }

            // Silent-PCM diagnostic. Probe every converted buffer for
            // any non-zero sample; track the longest run of consecutive
            // all-zero samples since session start. Crossing one full
            // second of dead air is a strong signal that the macOS
            // audio HAL is feeding us silence (voice processing
            // misconfigured, mic muted, wrong input device) — the
            // failure mode whisper hallucinates on. Log once per
            // session to keep the support bundle scan clean.
            let frameCount = Int(out.frameLength)
            if frameCount > 0, let chData = out.floatChannelData {
                let samples = chData[0]
                var bufMaxAbs: Float = 0
                for i in 0..<frameCount {
                    let v = abs(samples[i])
                    if v > bufMaxAbs {
                        bufMaxAbs = v
                        if bufMaxAbs > 1e-6 { break }
                    }
                }
                if bufMaxAbs <= 1e-6 {
                    recorder.silentFrameRun &+= frameCount
                    if !recorder.silenceLogged
                        && recorder.silentFrameRun >= DictationRecorder.silenceFrameThreshold {
                        recorder.silenceLogged = true
                        NSLog(
                            "Juno: %.1fs of all-zero PCM captured (voice_processing=%@, hw_input=%@) — likely macOS audio HAL silence. Try Settings → Audio → Mic processing OFF or check System Settings → Sound → Input.",
                            Double(recorder.silentFrameRun) / 16_000.0,
                            recorder.voiceProcessingEnabled ? "on" : "off",
                            input.inputFormat(forBus: 0).description
                        )
                    }
                } else {
                    recorder.silentFrameRun = 0
                }
            }

            // Live Speech must see 16 kHz mono float — not raw hardware buffers.
            bufferCallback?(out)
        }
        }) {
            throw NSError(
                domain: "JunoDictationRecorder",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not start microphone capture (\(installError.localizedDescription))."]
            )
        }

        // `engine.start()` can also raise an ObjC exception when the audio HAL
        // is wedged. Guard it too and tear down the tap before propagating.
        var startSwiftError: Error?
        if let startException = JunoCatchNSException({
            do {
                try engine.start()
            } catch {
                startSwiftError = error
            }
        }) {
            input.removeTap(onBus: 0)
            throw NSError(
                domain: "JunoDictationRecorder",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Could not start the audio engine (\(startException.localizedDescription))."]
            )
        }
        if let startSwiftError {
            input.removeTap(onBus: 0)
            throw startSwiftError
        }
        self.engine = engine
        self.file = file
        if voiceProcessingEnabled {
            NSLog("Juno: voice processing enabled for mic capture")
        }
        return tmp
    }

    func stop() -> URL? {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        file = nil
        return destURL
    }

    /// Returns the URL of the WAV currently being written, or nil when not
    /// recording. The file is open and being appended to; readers must
    /// tolerate a partial WAV — the header's data-size field is only
    /// updated when AVAudioFile is closed (on ``stop()``), so a mid-stream
    /// snapshot reports a smaller size than the actual data section. mlx
    /// whisper / faster_whisper compute size from the file length anyway,
    /// so this works in practice. If a caller needs a strictly-correct
    /// header it should ``stop()`` first.
    var currentWAVURL: URL? { destURL }
}

// MARK: - Dock visibility

enum JunoDockVisibility {
    /// Apply the current `showInDock` preference.
    @MainActor
    static func applyCurrent() {
        let show = JunoUserDefaults.showInDock
        // When hidden: menu-bar-only behavior (`MenuBarExtra` remains),
        // but the app disappears from Dock / Cmd-Tab.
        NSApp.setActivationPolicy(show ? .regular : .accessory)
    }
}

final class JunoTargetApplicationTracker {
    static let shared = JunoTargetApplicationTracker()

    private static let ignoredTargetBundleIds: Set<String> = [
        "com.apple.usernotificationcenter",
        "com.apple.notificationcenterui",
        "com.apple.controlcenter",
        "com.apple.systemuiserver",
    ]
    private static let ignoredTargetNames: Set<String> = [
        "usernotificationcenter",
        "notification center",
        "control center",
    ]

    private var observer: NSObjectProtocol?
    private(set) var lastNonJunoApplication: NSRunningApplication?

    private init() {}

    func start() {
        guard observer == nil else { return }
        if let frontmost = NSWorkspace.shared.frontmostApplication {
            rememberIfExternal(frontmost)
        }
        observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else {
                return
            }
            self?.rememberIfExternal(app)
        }
    }

    func preferredTarget(current frontmost: NSRunningApplication?) -> NSRunningApplication? {
        if let frontmost, isExternal(frontmost) {
            rememberIfExternal(frontmost)
            return frontmost
        }
        guard let remembered = lastNonJunoApplication, !remembered.isTerminated else {
            return frontmost
        }
        return remembered
    }

    private func rememberIfExternal(_ app: NSRunningApplication) {
        guard isExternal(app) else { return }
        lastNonJunoApplication = app
    }

    private func isExternal(_ app: NSRunningApplication) -> Bool {
        guard !app.isTerminated else { return false }
        guard let bundleId = app.bundleIdentifier, !bundleId.isEmpty else { return false }
        let own = Bundle.main.bundleIdentifier ?? JunoEngineContract.defaultBundleId
        return bundleId != own
            && !bundleId.hasPrefix("\(own).helper.")
            && !Self.isIgnoredSystemSurface(bundleId: bundleId, name: app.localizedName)
    }

    static func isIgnoredSystemSurface(bundleId: String?, name: String?) -> Bool {
        let bid = (bundleId ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let own = (Bundle.main.bundleIdentifier ?? JunoEngineContract.defaultBundleId).lowercased()
        if !bid.isEmpty, bid == own || bid.hasPrefix("\(own).helper.") {
            return true
        }
        if !bid.isEmpty, ignoredTargetBundleIds.contains(bid) {
            return true
        }
        let appName = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !appName.isEmpty && ignoredTargetNames.contains(appName)
    }

    /// System panes that surface focusable (but effectively non-editable for
    /// dictation) elements, so a posted Cmd+V is accepted by the event system
    /// yet inserts nothing. Kept separate from ``isIgnoredSystemSurface``
    /// (which also drives target tracking / context capture).
    static let nonEditableSystemPasteBundleIds: Set<String> = [
        "com.apple.systempreferences",   // System Settings (and legacy System Preferences)
    ]
    static let nonEditableSystemPasteNames: Set<String> = [
        "system settings",
        "system preferences",
    ]
    static func isNonEditableSystemPasteSurface(bundleId: String?, name: String?) -> Bool {
        let bid = (bundleId ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !bid.isEmpty, nonEditableSystemPasteBundleIds.contains(bid) { return true }
        let appName = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !appName.isEmpty && nonEditableSystemPasteNames.contains(appName)
    }

    /// How a frontmost app sampled at a paste boundary should affect the paste
    /// target. Pure + testable.
    enum PasteTargetDisposition: Equatable {
        /// A real external app — resolve it as the paste target.
        case adopt
        /// Juno's own HUD / notification / control center transiently frontmost.
        /// Keep the target captured at dictation start and do NOT gate the paste
        /// off — Juno's floating HUD panel can be momentarily frontmost at the
        /// finalize sample even while the user is dictating into another app, and
        /// gating off here mis-files the result into the copy-ready overlay.
        case preservePrior
        /// A genuinely-focused non-editable system pane (System Settings): a
        /// posted Cmd+V is accepted but inserts nothing, so fall back to copy.
        case copyOnly
    }

    static func pasteTargetDisposition(bundleId: String?, name: String?) -> PasteTargetDisposition {
        if isIgnoredSystemSurface(bundleId: bundleId, name: name) { return .preservePrior }
        if isNonEditableSystemPasteSurface(bundleId: bundleId, name: name) { return .copyOnly }
        return .adopt
    }
}

// MARK: - Shortcut preference

/// Persisted shortcut choice. The juno-hotkey binary emits events for all
/// supported keys; this preference tells the HotkeyBridge which to listen for.
enum JunoShortcutPreference: String, CaseIterable {
    case fn = "fn"
    case rightCommand = "rightCommand"
    case rightOption = "rightOption"
    case optionSpace = "optionSpace"
    case controlSpace = "controlSpace"

    private static let shortcutKey = "JunoShortcutKey"
    private static let legacyShortcutKey = "JunoShortcutKey"
    static let defaultShortcut: JunoShortcutPreference = .rightOption

    static var stored: JunoShortcutPreference {
        get {
            let ud = UserDefaults.standard
            if ud.string(forKey: shortcutKey) == nil,
               let legacy = ud.string(forKey: legacyShortcutKey) {
                ud.set(legacy, forKey: shortcutKey)
                ud.removeObject(forKey: legacyShortcutKey)
            }
            let raw = ud.string(forKey: shortcutKey) ?? Self.defaultShortcut.rawValue
            return JunoShortcutPreference(rawValue: raw) ?? Self.defaultShortcut
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: shortcutKey) }
    }

    var displayName: String {
        switch self {
        case .fn: return "Fn / Globe key"
        case .rightCommand: return "Right Command"
        case .rightOption: return "Right Option"
        case .optionSpace: return "Option + Space"
        case .controlSpace: return "Control + Space"
        }
    }
}

// MARK: - Audio buffer RMS helper

private extension AVAudioPCMBuffer {
    /// Returns root-mean-square amplitude across all frames on channel 0.
    func rms() -> Float? {
        let count = Int(frameLength)
        guard count > 0 else { return nil }
        switch format.commonFormat {
        case .pcmFormatFloat32:
            guard let data = floatChannelData else { return nil }
            let channel = data[0]
            var sum: Float = 0
            for i in 0..<count { let s = channel[i]; sum += s * s }
            return (sum / Float(count)).squareRoot()
        case .pcmFormatInt16:
            guard let data = int16ChannelData else { return nil }
            let channel = data[0]
            // Normalize int16 to [-1, 1] float-equivalent RMS.
            let scale = 1.0 / Float(Int16.max)
            var sum: Float = 0
            for i in 0..<count {
                let s = Float(channel[i]) * scale
                sum += s * s
            }
            return (sum / Float(count)).squareRoot()
        default:
            // Unknown layout — prefer returning nil over lying with zeros.
            return nil
        }
    }

    /// Linear PCM mono payload suitable for building a 16 kHz WAV (LE int16).
    func int16MonoInterleavedData() -> Data? {
        let n = Int(frameLength)
        guard n > 0 else { return nil }
        switch format.commonFormat {
        case .pcmFormatInt16:
            guard let ch = int16ChannelData?[0] else { return nil }
            return Data(bytes: ch, count: n * MemoryLayout<Int16>.size)
        case .pcmFormatFloat32:
            guard let ch = floatChannelData?[0] else { return nil }
            var out = Data(count: n * MemoryLayout<Int16>.size)
            out.withUnsafeMutableBytes { raw in
                let dst = raw.bindMemory(to: Int16.self).baseAddress!
                for i in 0..<n {
                    let f = max(-1, min(1, ch[i]))
                    dst[i] = Int16((f * Float(Int16.max)).rounded())
                }
            }
            return out
        default:
            return nil
        }
    }
}

// MARK: - Session WAV (in-memory PCM → standalone WAV)

private enum JunoSessionWAVBuilder {
    static let sampleRate: Double = 16_000

    /// Minimum PCM payload (~0.12 s @ 16 kHz mono int16) before we bother the broker.
    static let minPCMBytesForBroker = 3840

    static func wavData(fromInt16MonoLittleEndian pcm: Data) -> Data {
        let dataSize = UInt32(pcm.count)
        let riffChunkSize = UInt32(36) + dataSize
        var out = Data()
        out.append(contentsOf: "RIFF".utf8)
        var riffLE = riffChunkSize.littleEndian
        out.append(Data(bytes: &riffLE, count: 4))
        out.append(contentsOf: "WAVE".utf8)
        out.append(contentsOf: "fmt ".utf8)
        var sub1: UInt32 = 16
        out.append(Data(bytes: &sub1, count: 4))
        var audioFormat: UInt16 = 1 // PCM
        out.append(Data(bytes: &audioFormat, count: 2))
        var ch: UInt16 = 1
        out.append(Data(bytes: &ch, count: 2))
        var sr = UInt32(sampleRate).littleEndian
        out.append(Data(bytes: &sr, count: 4))
        var byteRate = UInt32(sampleRate * 2).littleEndian
        out.append(Data(bytes: &byteRate, count: 4))
        var blockAlign: UInt16 = 2
        out.append(Data(bytes: &blockAlign, count: 2))
        var bps: UInt16 = 16
        out.append(Data(bytes: &bps, count: 2))
        out.append(contentsOf: "data".utf8)
        var ds = dataSize.littleEndian
        out.append(Data(bytes: &ds, count: 4))
        out.append(pcm)
        return out
    }
}

// MARK: - Dictation orchestrator

struct JunoActionHUDResult: Equatable {
    let title: String
    let subtitle: String
    let symbolName: String
    let kind: JunoActionKind?
    let isFailure: Bool
}

struct HUDTranscriptSpan: Identifiable, Equatable {
    enum Origin: String {
        /// Algorithm-committed (LocalAgreement-2 agreed-prefix). Append-only,
        /// rendered at full opacity.
        case committed
        /// Volatile tail past the agreement boundary. May change wholesale
        /// between Whisper passes; rendered dimmed.
        case tail
        /// Final Qwen-adjudicated text. Replaces committed/tail on final.
        case corrected
        /// Legacy: pre-LocalAgreement live preview path.
        case draft
        /// Legacy: pending typing state. Used outside the live transcript path.
        case pending
    }

    let id: String
    let text: String
    let origin: Origin
    let revision: Int
    let changed: Bool
}

final class DictationController: ObservableObject {
    /// Included in broker ``insertion/committed`` as ``trigger_source``.
    var insertionTriggerSource: String = "hotkey"

    @Published private(set) var state: String = "idle"
    /// Typed view of ``state``. Read sites should compare against this enum
    /// instead of the raw wire string (Issue #8). The producer continues to
    /// write ``state`` as a wire-format string for backwards compatibility
    /// with logs, the broker, and the support bundle.
    var hudState: HUDState { HUDState.from(wireString: state) }
    @Published private(set) var livePartialText: String = ""
    /// Full draft for the HUD: live partial, or accumulated partials + current partial.
    @Published private(set) var liveDisplayTranscript: String = ""
    @Published private(set) var liveTranscriptSpans: [HUDTranscriptSpan] = []
    /// Monotonic counter incremented every time a final-stage correction
    /// replaces the live preview text. The HUD watches this to fire the
    /// one-shot "magical replace" shimmer (see ``JunoBrandIslandStack``).
    /// Distinct from ``liveAdjudicationRevision`` — that ticks on every
    /// preview/patch update; this only ticks on the moment of correction.
    @Published private(set) var correctionGeneration: Int = 0
    private let hudTranscriptStore = HUDTranscriptStore()
    private var hudCommittedRevealWork: DispatchWorkItem?
    private var hudCommittedRevealGeneration: UInt64 = 0
    private let hudCommittedRevealInterval: TimeInterval = 0.016

    // MARK: - Live caption source state machine
    //
    // HUD captions come from Juno's local preview ASR via
    // ``JunoPreviewStreamer``. Apple Speech is intentionally not used in
    // production dictation because it creates a separate macOS permission,
    // weaker homophone behavior, and a privacy prompt that conflicts with
    // Juno's local-first product contract.
    enum LiveSource: String { case none, listening, engine }
    @Published private(set) var liveSource: LiveSource = .none
    /// Buffer for engine preview partials. Mirrored to ``livePartialText``
    /// the moment we commit to ``.engine``.
    private var enginePreviewPartialText: String = ""
    /// Streamer that owns the per-utterance preview HTTP loop.
    let previewStreamer = JunoPreviewStreamer()
    /// Shell-side timing markers for the current utterance. The broker writes
    /// these into `runtime/utterance_lifecycle/<uid>.json` so long-dictation
    /// stop-to-paste latency can be decomposed without guessing.
    private var utteranceTimelineMs: [String: Int64] = [:]
    /// First-word budget after speech is actually detected. Starting this at
    /// hotkey-down made normal "press, breathe, speak" sessions show fallback
    /// before the preview backend had any speech to decode.
    private var engineFirstWordBudget: TimeInterval = 1.6
    private var engineFirstWordTimer: DispatchWorkItem?
    /// Shown when live partials are unavailable but recording still works.
    @Published private(set) var liveSpeechHint: String?
    /// Brief “+N words” island after a successful paste (brand kit “done” state).
    @Published private(set) var transientDoneWordCount: Int?
    /// Brief action outcome in the floating HUD after a reminder/note/alarm
    /// command is consumed without pasting literal command text.
    @Published private(set) var transientActionHUDResult: JunoActionHUDResult?
    /// Subtle flash on the mark after text lands (brand kit “draft flash”).
    @Published private(set) var draftFlashActive: Bool = false
    /// Wall-clock start for the listening timer in the HUD.
    @Published private(set) var dictationStartedAt: Date?
    /// Wall-clock start while waiting on broker ASR (`refining` state).
    @Published private(set) var refiningStartedAt: Date?
    /// Transcribed text that couldn't be pasted — shown in overlay so user can Copy.
    @Published private(set) var copyableTranscript: String? = nil
    /// Brief confirmation toast for copy actions in the HUD (e.g. "Copied").
    @Published private(set) var transientCopyToast: String? = nil
    /// Set after a paste when the broker reports `degraded_writer == true`
    /// (writer LLM failed to load — Qwen3 OOM, missing weights, mlx_lm import
    /// error, etc.). Fix V-4 (audit A2): surface a quiet HUD line so the user
    /// has a signal that polish modes are running in basic mode. Auto-clears
    /// on the next dictation start.
    @Published private(set) var writerDegradedNotice: Bool = false
    /// Brief shimmer sweep in the HUD after paste/copy (visual delight).
    @Published private(set) var delightSweepActive: Bool = false
    /// Live RMS amplitude (0..~1) for the HUD meter.
    @Published private(set) var currentRMS: Float = 0
    /// Best-effort writing mode label for the HUD.
    @Published private(set) var currentModeLabel: String? = nil
    /// Default-input device name (e.g. "MacBook Pro Microphone") shown in the HUD footer.
    @Published private(set) var currentInputDeviceName: String? = nil
    @Published var targetApp: TargetApp? = nil

    struct TargetApp {
        let name: String
        let bundleIdentifier: String?
        let icon: NSImage?
    }
    /// Latest workbench/session revision observed from the engine-owned UI state.
    @Published private(set) var engineSessionRevision: Int = -1

    private var copyToastWorkItem: DispatchWorkItem?
    private var copyDismissWorkItem: DispatchWorkItem?
    private var clearActionHUDWorkItem: DispatchWorkItem?
    private var workbenchStateTimer: Timer?
    private var engineSessionPartialText: String = ""
    private var engineSessionFinalCandidateText: String = ""
    private var engineSessionActiveUtteranceId: String?
    private var lastNonEmptyHUDTranscript: String = ""
    /// Monotonic session token so late async callbacks from an older dictation
    /// session cannot mutate the HUD after a newer session has started.
    private var dictationSessionGeneration: UInt64 = 0
    private var capabilityCheckInFlight = false
    /// Workspace observer that keeps ``targetApp`` (the HUD focus pill) in
    /// sync with whatever the user clicks into mid-dictation. Registered
    /// lazily on the first ``beginPushToTalk`` and never removed — the
    /// controller is a process singleton.
    private var liveTargetAppObserver: NSObjectProtocol?

    private func ensureLiveTargetAppObserver() {
        guard liveTargetAppObserver == nil else { return }
        liveTargetAppObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let self else { return }
            // Only follow focus while a dictation is in flight. Once we go
            // idle, the pill should stay frozen on the last target so the
            // user can still see where their text landed.
            guard self.hudState != .idle else { return }
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            else { return }
            // Ignore Juno itself (HUD/Brand window briefly reactivating).
            if app.bundleIdentifier == "com.juno.shell" { return }
            self.targetApp = TargetApp(
                name: app.localizedName ?? "App",
                bundleIdentifier: app.bundleIdentifier,
                icon: app.icon
            )
            self.sessionContextTape.capture(reason: "app_switch")
        }
    }

    func clearCopyableTranscript() {
        copyableTranscript = nil
    }

    func copyCopyableTranscriptToClipboard() {
        guard let text = copyableTranscript, !text.isEmpty else { return }
        Clipboard.writeString(text)
        // No standalone copy sound — sound contract is "one on HUD open, one
        // on HUD close." When `copyableTranscript` clears below, the HUD
        // overlay coordinator fades out and emits the single close cue.
        showCopyToast("Copied")
        let dismiss = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.copyableTranscript = nil
        }
        copyDismissWorkItem?.cancel()
        copyDismissWorkItem = dismiss
        // Tighter dismiss window so the HUD doesn't linger after a successful
        // copy. Was 0.82s — shortened so the user can return to their flow
        // immediately. The toast text remains visible long enough to read.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.55, execute: dismiss)
    }

    /// User pressed Esc while the HUD was visible. Tear down audio + recognition
    /// without pasting; mirrors the early-cancel branch in ``toggleDictation`` but
    /// also covers mid-recording cancel and refining cancel where the broker call
    /// is allowed to complete in the background but its result is dropped.
    func cancelDictation() {
        switch hudState {
        case .idle:
            // Idle but copy-ready: dismissing the copy panel counts as cancel.
            // Idle with NOTHING to cancel must be a no-op for history:
            // ``juno-hotkey`` forwards every global Esc press system-wide, so
            // an Esc in another app after a successful paste was retroactively
            // stamping failure_reason="user_cancelled_hud" onto a row that
            // pasted fine (production 2026-06-11).
            if copyableTranscript != nil {
                persistHudCancelDraftIfNeeded()
                copyableTranscript = nil
                transientDoneWordCount = nil
            }
            transientActionHUDResult = nil
            clearActionHUDWorkItem?.cancel()
            clearActionHUDWorkItem = nil
            return
        case .refining:
            // Recognition already torn down; let the broker call resolve but discard
            // its result on arrival via the cancel marker so it can't paste.
            // (``persistHudCancelDraftIfNeeded`` skips refining by design —
            // the pipeline owns history for in-flight broker calls.)
            cancelInFlightBrokerInsertion = true
            goIdleOnMain()
            return
        default:
            break
        }
        // Mid-recording cancel: persist what the user said before Esc as a
        // "draft" history row. The broker upserts this verbatim with
        // failure_reason="user_cancelled_hud" and paste_kind="none" so the
        // user can find it later in History → Issues. Fire-and-forget.
        persistHudCancelDraftIfNeeded()
        teardownRecognition()
        micWatchdog?.cancel()
        micWatchdog = nil
        noSpeechWatchdog?.cancel()
        noSpeechWatchdog = nil
        stopWorkbenchStatePolling(clear: true)
        cancelEnginePreviewStreaming(reason: "user_cancel")
        liveAdjudicationInFlight = false
        pendingLiveAdjudicationReason = nil
        pendingLiveAdjudicationSnapshot = nil
        livePartialText = ""
        resetHUDTranscriptStore()
        liveSpeechHint = nil
        copyableTranscript = nil
        transientDoneWordCount = nil
        accumulatedText = ""
        cancelInFlightBrokerInsertion = true
        goIdleOnMain()
    }

    /// Best-effort POST to ``/api/broker/history/cancel_draft`` with the
    /// in-flight transcript so the History row records what the user
    /// said before they pressed Esc. Skipped while a broker snapshot is
    /// in flight (the pipeline owns history for those paths) and during
    /// ``refining`` (same reason). Idempotent on the broker side.
    private func persistHudCancelDraftIfNeeded() {
        guard !brokerSnapshotInFlight else { return }
        guard hudState != .refining else { return }

        let uid = pendingUtteranceId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !uid.isEmpty else { return }

        // Resolve a draft transcript: prefer the live HUD text, fall back
        // to the copy-ready buffer so dismissing the copy panel still
        // captures something.
        var draft = (liveDisplayTranscript.isEmpty ? livePartialText : liveDisplayTranscript)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if draft.isEmpty, hudState == .idle,
           let c = copyableTranscript?.trimmingCharacters(in: .whitespacesAndNewlines),
           !c.isEmpty {
            draft = c
        }
        guard !draft.isEmpty else { return }

        var payload: [String: Any] = [
            "utterance_id": uid,
            "transcript": draft,
            "raw_transcript": draft,
            "language_mode": JunoUserDefaults.languageMode,
        ]
        if !targetAppBundleId.isEmpty { payload["app_bundle_id"] = targetAppBundleId }
        if !targetWindowTitle.isEmpty { payload["window_title_hint"] = targetWindowTitle }
        if let fc = utteranceFrozenContext, !fc.isEmpty {
            payload["frozen_context"] = fc
        }
        JunoBroker.postJSON(path: "api/broker/history/cancel_draft", payload: payload) { _ in
            // Silent — the only failure mode worth surfacing would be
            // "broker is down", and the HUD has already been dismissed
            // by the time this resolves. The user will retry naturally.
        }
    }

    /// Set when ``cancelDictation`` fires so any late-arriving broker insertion
    /// result is dropped instead of pasted.
    fileprivate var cancelInFlightBrokerInsertion: Bool = false

    private let recorder = DictationRecorder()
    private var targetPid: pid_t = 0
    private var targetAppBundleId: String = ""
    private var targetWindowTitle: String = ""
    private var pendingUtteranceId: String = ""
    /// Surface-specific terms from the last capability probe, passed to the
    /// local preview/final pipeline for vocabulary bias.
    private var surfaceRecognitionHints: [String] = []
    private var textmonTask: Process?
    private var textmonStdout: FileHandle?

    // MARK: Pause-based partial insertion

    // lastSoundTime is written on the audio-engine thread and read on main.
    // Guard with NSLock for a cheap, safe 64-bit read/write.
    private let soundTimeLock = NSLock()
    private var _lastSoundTime: TimeInterval = 0

    private var silenceTimer: Timer?
    private var isPartialMode: Bool = false        // at least one partial pasted this session
    private var accumulatedText: String = ""       // all text pasted in partials
    private var partialInsertFailed: Bool = false  // last partial paste failed
    private var hasInsertedTextThisDictation: Bool = false

    // Broker-on-pause state. See the
    // ``reconcileWithBrokerOnRelease`` doc comment above for the full
    // design rationale.
    private var brokerSnapshotInFlight: Bool = false
    private var pendingBrokerSnapshot: Bool = false
    private struct LiveAdjudicationSnapshot {
        let visibleText: String
        let reason: String
        let createdAt: TimeInterval
        let speechAt: TimeInterval
        let wordCount: Int
        let charCount: Int
    }
    private var liveAdjudicationInFlight: Bool = false
    private var pendingLiveAdjudicationReason: String?
    private var pendingLiveAdjudicationSnapshot: LiveAdjudicationSnapshot?
    private var lastLiveAdjudicationRequestedAt: TimeInterval = 0
    private var lastLiveAdjudicationSpeechAt: TimeInterval = 0
    private var lastLiveAdjudicationWordCount: Int = 0
    private var lastLiveAdjudicationCharCount: Int = 0
    private var lastLiveAdjudicationVisibleText: String = ""
    private var liveAdjudicationSlowCount: Int = 0
    private var liveAdjudicationRevision: Int = 0
    private var liveAdjudicationBackpressureUntil: TimeInterval = 0
    private var lastLiveHUDTextChangeAt: TimeInterval = 0
    private var lastLiveHUDTextChangePCMBytes: Int = 0
    private var lastPreviewSegmentRollSpeechAt: TimeInterval = 0
    private var liveAudioCheckpointInFlight: Bool = false
    private var lastLiveAudioCheckpointRequestedAt: TimeInterval = 0
    private var lastLiveAudioCheckpointPCMBytes: Int = 0
    private var liveAudioCheckpointBackpressureUntil: TimeInterval = 0
    private var sessionContextTape = JunoSessionContextTape()
    private var lastPastedFromBroker: String = ""
    private var pendingFinalBrokerPasteKind: String?
    private var pendingFinalReplaceTarget: String?
    private var pendingFinalReplaceTargetChars: Int = 0
    private var recorderStopped: Bool = false
    private var brokerSnapshotFailedThisSession: Bool = false
    private var brokerSnapshotLowSignalThisSession: Bool = false
    /// ``juno-capability`` JSON captured when recording starts (hotkey time).
    private var utteranceFrozenContext: [String: Any]?
    /// When the frozen snapshot had a non-empty selection, skip pause-based
    /// partial pastes so the selection survives for the final replace.
    private var suppressPartialPasteForSelectionEditing: Bool = false
    /// Char count of the selection captured at recording start. Drives the
    /// HUD "Editing N chars" chip so the user sees Juno locked onto their
    /// selection. 0 when no selection. Resets when dictation ends.
    @Published private(set) var editingSelectionCharCount: Int = 0
    /// Whether ``juno-textmon`` should verify a replace-selection paste landed.
    private var textMonExpectsReplacePaste: Bool = false
    /// Transcript we last pasted (for textmon INITIAL vs replace verification).
    private var pendingTextMonExpectedPaste: String = ""
    /// Focused-field value observed immediately after paste, when AX exposes it.
    private var textMonInitialSnapshot: String?
    /// From capability probe: focused AX role looks like a text insertion target.
    private var likelyPasteDestination: Bool = true
    /// TOCTOU pin for ``JunoSecureFieldPolicy``. Captured at dictation
    /// start and refreshed when the in-process AX snapshot is
    /// authoritative (i.e. AX-trusted + ok). Read by every privacy gate
    /// (paste / learn-from-corrections / history / audio upload) for
    /// the duration of the in-flight utterance, so a mid-utterance
    /// focus change to a non-secure field can no longer un-strip the
    /// already-decided privacy posture. Reset at session reset.
    fileprivate var lastSecureFlag: Bool = false
    private var sessionStartTime: TimeInterval = 0
    private var clearDoneWorkItem: DispatchWorkItem?

    // In-memory session PCM (16 kHz mono int16 LE) — broker snapshots never read the actively written WAV.
    private let pcmLock = NSLock()
    private var sessionPCMData = Data()
    private var framesReceivedThisSession: Int64 = 0
    private var firstAudioFrameAt: TimeInterval = 0
    private var lastSpeechEnergyAt: TimeInterval = 0
    private var lastPartialSnapshotSpeechAt: TimeInterval = 0
    private var hasEverDetectedSpeech: Bool = false
    private var writerWarmRequestedThisSession: Bool = false
    private var speechDetectedLoggedThisSession: Bool = false
    private var ambientNoiseRMS: Float = 0.004
    private var micWatchdog: DispatchWorkItem?
    private var noSpeechWatchdog: DispatchWorkItem?
    /// Broker returned a full transcript that no longer extends our safe pasted prefix — do not
    /// delete mid-session; reconcile on finalize / copy-ready (repair doc P7).
    private var pendingRevisionFullTranscript: String?

    private static let silenceRMSThreshold: Float = 0.003
    private static let pauseSecondsDefault: TimeInterval = 1.4
    // Grace period: don't trigger a pause commit in the first second of a
    // session (avoids firing before the user even begins speaking).
    private static let graceSecondsDefault: TimeInterval = 1.0
    /// If the tap runs but no audio buffers arrive, surface a mic wiring / permission error.
    private static let micNoFrameTimeoutSeconds: TimeInterval = 2.5
    /// Tap-to-toggle can miss the second tap if the app is backgrounded or the helper loses an event.
    /// Do not let a no-speech capture keep the preview service and HUD awake indefinitely.
    private static let noSpeechAutoCancelSeconds: TimeInterval = 60.0
    private static let ambientNoiseAlpha: Float = 0.06
    private static let previewSegmentPauseSeconds: TimeInterval = 1.75
    private static let previewSegmentMinPauseRollSeconds: TimeInterval = 8.0
    private static let previewSegmentMaxSeconds: TimeInterval = 12.0
    private static let liveAudioCheckpointMinAudioSeconds: TimeInterval = 8.0
    private static let liveAudioCheckpointStaleAfter: TimeInterval = 2.2
    private static let liveAudioCheckpointMinInterval: TimeInterval = 4.0
    private static let liveAudioCheckpointMinNewAudioSeconds: TimeInterval = 3.0
    private static let liveAudioCheckpointSpeechFreshAfter: TimeInterval = 3.2
    private static let liveAudioCheckpointStaleSpeechBypassAfter: TimeInterval = 6.0
    private static let liveAudioCheckpointSlowResponseThreshold: TimeInterval = 4.0
    private static let liveAudioCheckpointMaxBackpressure: TimeInterval = 8.0

    // Broker-on-pause architecture. Each pause triggers a broker snapshot
    // built from the in-memory PCM buffer (no active-file WAV race). The
    // broker returns a writer-processed utterance; we paste that utterance
    // once and then trim the consumed PCM so the HUD can keep listening for
    // the next utterance in the same session.
    //
    // Coalescing: when a pause fires while a broker call is still in
    // flight, we set ``pendingBrokerSnapshot``; the completion handler
    // fires one more snapshot. At most two calls are outstanding.
    private var effectivePauseSeconds: TimeInterval {
        // Honor the user-facing "Pause sensitivity" slider in Settings.
        // ``JunoUserDefaults.pauseSensitivitySeconds`` clamps to the same
        // 0.8–3.0s range as the slider; when never set it returns the
        // historical 1.4s default that ``pauseSecondsDefault`` also encodes.
        TimeInterval(JunoUserDefaults.pauseSensitivitySeconds)
    }

    private var effectiveGraceSeconds: TimeInterval {
        Self.graceSecondsDefault
    }

    @discardableResult
    private func beginNewDictationGeneration() -> UInt64 {
        dictationSessionGeneration &+= 1
        return dictationSessionGeneration
    }

    private func matchesCurrentDictationGeneration(_ generation: UInt64) -> Bool {
        generation == dictationSessionGeneration
    }

    private func playHUDOpenSound() {
        guard JunoUserDefaults.hudOpenSoundEnabled else { return }
        JunoHUDSound.playOpen()
    }

    private func syncLiveCaptionSettingToBroker() {
        JunoBroker.postJSON(
            path: "api/broker/settings/live_caption",
            payload: ["enabled": JunoUserDefaults.hudLiveTranscriptionsEnabled]
        ) { _ in }
    }

    private func requestWriterWarmForActiveDictation(generation: UInt64, reason: String) {
        guard matchesCurrentDictationGeneration(generation) else { return }
        guard !writerWarmRequestedThisSession else { return }
        writerWarmRequestedThisSession = true
        JunoBroker.getJSON(path: "api/broker/writer/warm") { [weak self] obj in
            guard let self else { return }
            guard self.matchesCurrentDictationGeneration(generation) else { return }
            if (obj["ok"] as? Bool) == true {
                NSLog("Juno: writer warm requested during active dictation reason=%@", reason)
            } else if let error = obj["error"] as? String, !error.isEmpty {
                NSLog("Juno: writer warm skipped during active dictation reason=%@ error=%@", reason, error)
            }
        }
    }

    private func goIdleOnMain() {
        DispatchQueue.main.async {
            self.micWatchdog?.cancel()
            self.micWatchdog = nil
            self.noSpeechWatchdog?.cancel()
            self.noSpeechWatchdog = nil
            self.pcmLock.lock()
            self.sessionPCMData.removeAll(keepingCapacity: false)
            self.framesReceivedThisSession = 0
            self.pcmLock.unlock()
            self.pendingRevisionFullTranscript = nil
            self.hasInsertedTextThisDictation = false
            self.hasEverDetectedSpeech = false
            self.writerWarmRequestedThisSession = false
            self.firstAudioFrameAt = 0
            self.lastSpeechEnergyAt = 0
            self.stopWorkbenchStatePolling(clear: true)
            self.lastPartialSnapshotSpeechAt = 0
            self.speechDetectedLoggedThisSession = false
            self.state = "idle"
            self.dictationStartedAt = nil
            self.refiningStartedAt = nil
            self.utteranceFrozenContext = nil
            self.pendingFinalBrokerPasteKind = nil
            self.pendingFinalReplaceTarget = nil
            self.pendingFinalReplaceTargetChars = 0
            self.suppressPartialPasteForSelectionEditing = false
            self.editingSelectionCharCount = 0
            self.textMonExpectsReplacePaste = false
            self.currentRMS = 0
            self.currentModeLabel = nil
            self.targetApp = nil
            self.livePartialText = ""
            self.resetHUDTranscriptStore()
            self.liveSource = .none
            self.enginePreviewPartialText = ""
            self.transientCopyToast = nil
            self.delightSweepActive = false
            self.brokerSnapshotFailedThisSession = false
            self.brokerSnapshotLowSignalThisSession = false
            self.copyToastWorkItem?.cancel()
            self.copyToastWorkItem = nil
            self.copyDismissWorkItem?.cancel()
            self.copyDismissWorkItem = nil
            self.liveAdjudicationInFlight = false
            self.pendingLiveAdjudicationReason = nil
            self.pendingLiveAdjudicationSnapshot = nil
            self.liveAdjudicationSlowCount = 0
            self.lastLiveAdjudicationVisibleText = ""
            self.lastLiveHUDTextChangeAt = 0
            self.lastLiveHUDTextChangePCMBytes = 0
            self.lastPreviewSegmentRollSpeechAt = 0
            self.liveAudioCheckpointInFlight = false
            self.lastLiveAudioCheckpointRequestedAt = 0
            self.lastLiveAudioCheckpointPCMBytes = 0
            self.liveAudioCheckpointBackpressureUntil = 0
        }
    }

    private func syncLiveDisplayTranscript(allowFrozen: Bool = true) {
        // Live preview is engine-driven via JunoPreviewStreamer.onPartial →
        // hudTranscriptStore.applyPreviewRevision. This helper resolves the
        // store back into the HUD model after final correction/copy states.
        let resolved = resolvedHUDTranscript()
        if !resolved.isEmpty {
            _ = hudTranscriptStore.applyCommittedPrefix(resolved)
            syncHUDFromTranscriptStore()
            lastNonEmptyHUDTranscript = resolved
            return
        }
        if allowFrozen, hudState != .idle, !lastNonEmptyHUDTranscript.isEmpty {
            _ = hudTranscriptStore.applyCommittedPrefix(lastNonEmptyHUDTranscript)
            syncHUDFromTranscriptStore()
            return
        }
        resetHUDTranscriptStore()
    }

    private func applyLiveAdjudicatedTranscript(_ response: JunoBroker.TranscribeResponse) {
        cancelHUDCommittedReveal()
        if let patch = response.transcriptPatch,
           hudTranscriptStore.applyPatchEnvelope(patch) {
            engineSessionFinalCandidateText = hudTranscriptStore.text
            livePartialText = hudTranscriptStore.text
            syncHUDFromTranscriptStore()
            // Live correction patches arrive mid-utterance from the
            // adjudicator (e.g. punctuation/casing fixes). Fire the
            // shimmer here too so the user sees the same magical
            // replace whether the patch is mid-stream or final.
            bumpCorrectionGeneration()
            return
        }
        let stage = (response.transcriptStage ?? response.stage ?? "").lowercased()
        guard stage == "final_delivery" || stage == "final" else { return }
        let trimmed = normalizedHUDText(response.transcript)
        guard !trimmed.isEmpty else { return }
        let priorText = hudTranscriptStore.text
        hudTranscriptStore.applyFinalText(trimmed)
        engineSessionFinalCandidateText = trimmed
        livePartialText = trimmed
        syncHUDFromTranscriptStore()
        // Final corrections that actually change visible text trigger the
        // shimmer; if final equals the live preview verbatim we skip the
        // pulse so the magic stays reserved for the *replace* moment.
        if priorText != trimmed {
            bumpCorrectionGeneration()
        }
    }

    private func bumpCorrectionGeneration() {
        correctionGeneration &+= 1
    }

    private func updateLiveTranscriptSpans(text: String, origin: HUDTranscriptSpan.Origin) {
        let trimmed = normalizedHUDText(text)
        guard !trimmed.isEmpty else {
            liveTranscriptSpans = []
            return
        }
        let oldWords = liveTranscriptSpans.map(\.text)
        let words = trimmed.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
        liveAdjudicationRevision += 1
        liveTranscriptSpans = words.enumerated().map { idx, word in
            let prior = idx < oldWords.count ? oldWords[idx] : nil
            let changed = prior != word && origin == .corrected
            let stableToken = word.lowercased().filter { $0.isLetter || $0.isNumber }
            return HUDTranscriptSpan(
                id: "\(idx)-\(stableToken)",
                text: word,
                origin: changed ? .corrected : origin,
                revision: liveAdjudicationRevision,
                changed: changed
            )
        }
    }

    private func syncHUDFromTranscriptStore() {
        let previous = liveDisplayTranscript
        liveDisplayTranscript = hudTranscriptStore.text
        liveTranscriptSpans = hudTranscriptStore.spans
        if !hudTranscriptStore.text.isEmpty {
            lastNonEmptyHUDTranscript = hudTranscriptStore.text
        }
        liveAdjudicationRevision = hudTranscriptStore.revision
        if normalizedHUDText(previous) != normalizedHUDText(liveDisplayTranscript) {
            lastLiveHUDTextChangeAt = Date.timeIntervalSinceReferenceDate
            lastLiveHUDTextChangePCMBytes = sessionPCMByteCount()
        }
    }

    private func cancelHUDCommittedReveal() {
        hudCommittedRevealGeneration &+= 1
        hudCommittedRevealWork?.cancel()
        hudCommittedRevealWork = nil
    }

    private func applyEnginePreviewChunkToHUD(committed: String, tail: String) {
        _ = tail
        let targetCommitted = committed.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetTail = ""
        let steps = HUDCommittedTextReveal.steps(
            current: hudTranscriptStore.committedText,
            target: targetCommitted
        )
        guard steps.count > 1 else {
            cancelHUDCommittedReveal()
            _ = hudTranscriptStore.applyPreviewRevision(committed: targetCommitted, tail: targetTail)
            enginePreviewPartialText = hudTranscriptStore.rawText
            if liveSource == .engine {
                livePartialText = hudTranscriptStore.text
                syncHUDFromTranscriptStore()
            }
            return
        }

        cancelHUDCommittedReveal()
        let generation = hudCommittedRevealGeneration
        revealCommittedStep(steps, tail: targetTail, index: 0, generation: generation)
    }

    private func revealCommittedStep(_ steps: [String], tail: String, index: Int, generation: UInt64) {
        guard generation == hudCommittedRevealGeneration else { return }
        guard index < steps.count else { return }
        let isLast = index == steps.count - 1
        _ = hudTranscriptStore.applyPreviewRevision(
            committed: steps[index],
            tail: isLast ? tail : ""
        )
        enginePreviewPartialText = hudTranscriptStore.rawText
        if liveSource == .engine {
            livePartialText = hudTranscriptStore.text
            syncHUDFromTranscriptStore()
        }
        guard !isLast else {
            hudCommittedRevealWork = nil
            return
        }
        let work = DispatchWorkItem { [weak self] in
            self?.revealCommittedStep(steps, tail: tail, index: index + 1, generation: generation)
        }
        hudCommittedRevealWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + hudCommittedRevealInterval, execute: work)
    }

    private func resetHUDTranscriptStore() {
        cancelHUDCommittedReveal()
        hudTranscriptStore.reset()
        liveDisplayTranscript = ""
        liveTranscriptSpans = []
        lastLiveHUDTextChangeAt = 0
        lastLiveHUDTextChangePCMBytes = sessionPCMByteCount()
    }

    private func resolvedHUDTranscript() -> String {
        let candidates = [
            normalizedHUDText(engineSessionFinalCandidateText),
            normalizedHUDText(engineSessionPartialText),
            normalizedHUDText(livePartialText),
            normalizedHUDText(pendingRevisionFullTranscript),
            normalizedHUDText(lastPastedFromBroker),
            normalizedHUDText(accumulatedText),
        ]
        return candidates.first(where: { !$0.isEmpty }) ?? ""
    }

    private func normalizedHUDText(_ raw: String?) -> String {
        repairLiveCaptionBoundaries(raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func rawLivePreviewTextForCorrection() -> String {
        let rawStoreText = normalizedHUDText(hudTranscriptStore.rawText)
        if !rawStoreText.isEmpty { return rawStoreText }
        for candidate in [enginePreviewPartialText, livePartialText, liveDisplayTranscript] {
            let normalized = normalizedHUDText(candidate)
            if !normalized.isEmpty { return normalized }
        }
        return ""
    }

    private func repairLiveCaptionBoundaries(_ raw: String) -> String {
        // Apple live captions sometimes glue a trailing word to the pronoun
        // "I" across a pause ("workI don't know"). That artifact is visible
        // in the HUD and also poisons protected-term extraction downstream.
        let ns = raw as NSString
        return ns.replacingOccurrences(
            of: #"([a-z])I(?=\s|$|['’]|[.,!?;:])"#,
            with: "$1 I",
            options: .regularExpression,
            range: NSRange(location: 0, length: ns.length)
        )
    }

    private func filteredEnginePreviewText(_ raw: String) -> String {
        let incoming = normalizedHUDText(raw)
        guard !incoming.isEmpty else { return incoming }
        let previous = normalizedHUDText(enginePreviewPartialText)
        let quietTail = hasEverDetectedSpeech
            && lastSpeechEnergyAt > sessionStartTime
            && Date.timeIntervalSinceReferenceDate - lastSpeechEnergyAt > 0.9
        guard quietTail else { return incoming }
        if let stripped = stripAppendedPreviewOutro(incoming: incoming, previous: previous) {
            if stripped != incoming {
                NSLog("Juno: stripped preview outro hallucination incoming_chars=%d previous_chars=%d", incoming.count, previous.count)
            }
            return stripped
        }
        return incoming
    }

    private func stripAppendedPreviewOutro(incoming: String, previous: String) -> String? {
        let stock = [
            "thank you",
            "thanks",
            "thanks for watching",
            "thank you for watching",
            "thanks for listening",
            "please subscribe",
            "subscribe",
            "bye",
            "goodbye",
        ]
        let foldedIncoming = Self.foldPreviewPhrase(incoming)
        if stock.contains(foldedIncoming) {
            return previous.isEmpty ? "" : previous
        }
        guard !previous.isEmpty else { return nil }
        let foldedPrevious = Self.foldPreviewPhrase(previous)
        guard foldedIncoming.hasPrefix(foldedPrevious) else { return nil }
        let suffix = foldedIncoming.dropFirst(foldedPrevious.count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if stock.contains(String(suffix)) {
            return previous
        }
        return nil
    }

    private static func foldPreviewPhrase(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: #"[\s\.\!\?,;:'"\-–—…]+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func currentWallClockMs() -> Int64 {
        Int64((Date().timeIntervalSince1970 * 1000.0).rounded())
    }

    private func resetUtteranceTimeline(startedAtMs: Int64 = DictationController.currentWallClockMs()) {
        utteranceTimelineMs = [
            "hotkey_start_pressed_ms": startedAtMs,
        ]
    }

    private func markUtteranceTimeline(_ key: String, at ms: Int64 = DictationController.currentWallClockMs()) {
        utteranceTimelineMs[key] = ms
    }

    private func utteranceTimelinePayload(extra: [String: Int64] = [:]) -> [String: Any] {
        var payload = utteranceTimelineMs.reduce(into: [String: Any]()) { out, item in
            out[item.key] = item.value
        }
        for (key, value) in extra {
            payload[key] = value
        }
        if !pendingUtteranceId.isEmpty {
            payload["utterance_id"] = pendingUtteranceId
        }
        return payload
    }

    private func observedUndoSafePaste(_ text: String) -> Bool {
        markUtteranceTimeline("paste_attempt_started_ms")
        // Paste "success" historically meant only "the Cmd+V keystroke was
        // posted", not "the text landed" — so a non-editable target that
        // accepts the keystroke but inserts nothing (System Settings,
        // read-only views) reported a false success. Read the focused field's
        // value BEFORE the paste so we can verify a real change AFTER.
        let verifyEnabled = (UserDefaults.standard.object(forKey: "JunoPasteVerificationEnabled") as? Bool) ?? true
        let before = verifyEnabled ? JunoLocalCapability.focusedValueSignature() : nil
        let posted = Clipboard.undoSafePaste(text)
        markUtteranceTimeline("paste_attempt_finished_ms")
        // Couldn't even post the keystroke → definite failure.
        guard posted else { return false }
        // Verification off, can't reach the field, or the field exposes no
        // readable AXValue → keep the historical behaviour (assume success).
        // Deliberate: Chromium/Electron, web fields, Terminal and many custom
        // views don't expose a readable value, and a read-back there would
        // turn real pastes into false failures.
        guard verifyEnabled, let before, before.readable else { return true }
        let landed = pasteLandedByReadback(beforeValue: before.value)
        if !landed {
            NSLog("Juno: paste read-back saw no field change — treating as failed (non-editable target, e.g. System Settings)")
        }
        return landed
    }

    /// Poll the focused field's AX value briefly after a Cmd+V. We just posted
    /// the keystroke, so ANY change to the value within the window is strong
    /// evidence the paste landed (returns true as soon as it's seen). If the
    /// value never changes before the deadline, the paste did not take —
    /// the target accepted the keystroke but inserted nothing. Erring toward
    /// "changed = success" keeps us from false-failing real pastes.
    private func pasteLandedByReadback(beforeValue: String, deadlineMs: Int = 350, stepMs: Int = 40) -> Bool {
        let deadline = Date().addingTimeInterval(Double(deadlineMs) / 1000.0)
        while Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(Double(stepMs) / 1000.0))
            guard let after = JunoLocalCapability.focusedValueSignature() else { return true }
            guard after.readable else { return true } // became unverifiable → don't disprove
            if after.value != beforeValue { return true }
        }
        return false
    }

    private func startWorkbenchStatePolling() {
        if workbenchStateTimer != nil { return }
        let timer = Timer.scheduledTimer(withTimeInterval: 0.18, repeats: true) { [weak self] _ in
            self?.pollWorkbenchState()
        }
        workbenchStateTimer = timer
        pollWorkbenchState()
    }

    private func stopWorkbenchStatePolling(clear: Bool) {
        workbenchStateTimer?.invalidate()
        workbenchStateTimer = nil
        if clear {
            engineSessionRevision = -1
            engineSessionActiveUtteranceId = nil
            engineSessionPartialText = ""
            engineSessionFinalCandidateText = ""
            lastNonEmptyHUDTranscript = ""
        }
    }

    private func pollWorkbenchState() {
        JunoBroker.getJSON(path: "api/state") { [weak self] obj in
            guard let self else { return }
            self.applyWorkbenchStateSnapshot(obj)
        }
    }

    private func applyWorkbenchStateSnapshot(_ obj: [String: Any]) {
        let activeUtteranceId = obj["active_utterance_id"] as? String
        let shouldUseSnapshot: Bool
        if let activeUtteranceId, !activeUtteranceId.isEmpty {
            shouldUseSnapshot = activeUtteranceId == pendingUtteranceId
        } else {
            let hasInFlightUiText =
                !(normalizedHUDText(obj["partial_text"] as? String).isEmpty
                    && normalizedHUDText(obj["final_candidate_text"] as? String).isEmpty)
            let pendingCommit = (obj["pending_commit"] as? Bool) ?? false
            shouldUseSnapshot = (hudState == .partialCommit || hudState == .refining || brokerSnapshotInFlight || pendingCommit)
                && hasInFlightUiText
        }
        guard shouldUseSnapshot else { return }

        engineSessionRevision = (obj["revision"] as? Int) ?? engineSessionRevision
        engineSessionActiveUtteranceId = activeUtteranceId
        engineSessionPartialText = (obj["partial_text"] as? String) ?? ""
        engineSessionFinalCandidateText = (obj["final_candidate_text"] as? String) ?? ""
        livePartialText = engineSessionPartialText

        if let hint = obj["commit_conflict"] as? String, !hint.isEmpty {
            liveSpeechHint = hint.replacingOccurrences(of: "_", with: " ").capitalized
        }

        syncLiveDisplayTranscript()
    }

    private func showCopyToast(_ message: String) {
        transientCopyToast = message
        delightSweepActive = true
        copyToastWorkItem?.cancel()
        let clear = DispatchWorkItem { [weak self] in
            self?.transientCopyToast = nil
            self?.delightSweepActive = false
        }
        copyToastWorkItem = clear
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: clear)
    }

    // MARK: - Tap-to-toggle entry point

    /// Single public entry point for the hotkey bridge.
    /// First tap → start; second tap → stop; tap while checking → cancel.
    func toggleDictation() {
        switch hudState {
        case .idle:
            beginPushToTalk()
        case .listening, .partialCommit, .checkingMic, .waitingSpeech:
            endPushToTalkAndDictate()
        case .checkingCapability:
            // User cancelled before recording even started.
            teardownRecognition()
            goIdleOnMain()
        default:
            // If stuck in error or blocked state, next tap resets to idle
            // so the tap after that can start a fresh recording session.
            if hudState.isErrorOrBlocked {
                micWatchdog?.cancel()
                micWatchdog = nil
                state = "idle"
                dictationStartedAt = nil
                refiningStartedAt = nil
                currentRMS = 0
                currentModeLabel = nil
                targetApp = nil
            }
            // refining — ignore (let broker call finish)
        }
    }

    // MARK: - Push-to-talk entry (internal)

    func beginPushToTalk() {
        guard hudState == .idle else { return }
        guard JunoUserDefaults.onboardingCompleted else {
            NSLog("Juno: dictation deferred — onboarding not completed")
            Task { @MainActor in
                JunoOnboardingWindow.showIfNeeded()
                JunoWindowActivation.activateApp()
                NSSound.beep()
            }
            return
        }
        guard JunoLocalCapability.processHasAccessibilityTrust() else {
            NSLog("Juno: dictation deferred — Accessibility is not trusted for this app build")
            state = "blocked:ax_permission_missing"
            liveSpeechHint = "Accessibility permission needed"
            promptAccessibilityPermission()
            NSSound.beep()
            return
        }
        NSLog("Juno: dictation start requested")
        playHUDOpenSound()
        syncLiveCaptionSettingToBroker()
        let generation = beginNewDictationGeneration()
        resetUtteranceTimeline()
        stopTextMon()
        // HotkeyBridge already invokes this on the main queue; keep all audio + UI state here.
        liveSpeechHint = nil
        transientDoneWordCount = nil
        transientActionHUDResult = nil
        clearActionHUDWorkItem?.cancel()
        clearActionHUDWorkItem = nil
        copyableTranscript = nil   // clear any previous copy-ready transcript
        writerDegradedNotice = false  // V-4 / A2: cleared on each new dictation
        dictationStartedAt = nil
        currentModeLabel = nil
        currentInputDeviceName = defaultInputDeviceName()
        let frontmost = NSWorkspace.shared.frontmostApplication
        let preferredTarget = JunoTargetApplicationTracker.shared.preferredTarget(current: frontmost)
        if let target = preferredTarget {
            targetPid = target.processIdentifier
            targetAppBundleId = target.bundleIdentifier ?? ""
            targetApp = TargetApp(name: target.localizedName ?? "App", bundleIdentifier: target.bundleIdentifier, icon: target.icon)
        } else {
            targetPid = 0
            targetAppBundleId = ""
            targetApp = nil
        }
        // Keep the HUD's focus pill live while the user dictates. Without
        // this, switching from app A to app B mid-dictation leaves the
        // pill stuck on A even though the paste will land in B.
        ensureLiveTargetAppObserver()
        cancelInFlightBrokerInsertion = false
        // **Do not pre-decide where the paste will land.** A user who taps
        // the dictation shortcut, then clicks into a different text field,
        // expects their dictation to land in the *new* field — not in
        // whatever was frontmost when they started. So we record the
        // preferred external target at start for HUD/context bookkeeping,
        // but we do **not** yank focus back to it. Paste-target resolution
        // is deferred to ``refreshPasteTargetFromCurrentFocus`` immediately
        // before each paste call.
        targetWindowTitle = ""
        pendingUtteranceId = "macshell-\(UUID().uuidString.prefix(16))"
        if JunoUserDefaults.hudLiveTranscriptionsEnabled {
            // Begin engine preview streaming for this utterance. We start
            // immediately on hotkey-down so the local preview receives the
            // first PCM buffer as early as the recorder does.
            beginEnginePreviewStreaming(uid: pendingUtteranceId)
        }
        isPartialMode = false
        accumulatedText = ""
        partialInsertFailed = false
        hasInsertedTextThisDictation = false
        brokerSnapshotInFlight = false
        pendingBrokerSnapshot = false
        liveAdjudicationInFlight = false
        pendingLiveAdjudicationReason = nil
        pendingLiveAdjudicationSnapshot = nil
        lastLiveAdjudicationRequestedAt = 0
        lastLiveAdjudicationSpeechAt = 0
        lastLiveAdjudicationWordCount = 0
        lastLiveAdjudicationCharCount = 0
        lastLiveAdjudicationVisibleText = ""
        liveAdjudicationSlowCount = 0
        liveAdjudicationRevision = 0
        liveAdjudicationBackpressureUntil = 0
        lastPastedFromBroker = ""
        pendingFinalBrokerPasteKind = nil
        pendingFinalReplaceTarget = nil
        pendingFinalReplaceTargetChars = 0
        recorderStopped = false
        writerWarmRequestedThisSession = false
        brokerSnapshotFailedThisSession = false
        brokerSnapshotLowSignalThisSession = false
        livePartialText = ""
        resetHUDTranscriptStore()
        lastNonEmptyHUDTranscript = ""
        likelyPasteDestination = true
        // Reset privacy TOCTOU pin. Will be set authoritatively by the
        // capability probe / in-process AX snapshot before any paste,
        // learn, history, or audio-upload gate fires for this session.
        lastSecureFlag = false
        refiningStartedAt = nil
        utteranceFrozenContext = nil
        sessionContextTape.reset()
        let startContextSnapshot = sessionContextTape.captureDictationStart(reason: "start")
        utteranceFrozenContext = startContextSnapshot.isEmpty ? nil : frozenContextForBroker(from: startContextSnapshot)
        suppressPartialPasteForSelectionEditing = false
        editingSelectionCharCount = 0
        textMonExpectsReplacePaste = false
        pendingRevisionFullTranscript = nil
        hasEverDetectedSpeech = false
        firstAudioFrameAt = 0
        lastSpeechEnergyAt = 0
        lastPartialSnapshotSpeechAt = 0
        speechDetectedLoggedThisSession = false
        noSpeechWatchdog?.cancel()
        noSpeechWatchdog = nil
        ambientNoiseRMS = 0.004
        pcmLock.lock()
        sessionPCMData.removeAll(keepingCapacity: false)
        framesReceivedThisSession = 0
        pcmLock.unlock()
        lastLiveHUDTextChangeAt = 0
        lastLiveHUDTextChangePCMBytes = 0
        lastPreviewSegmentRollSpeechAt = 0
        liveAudioCheckpointInFlight = false
        lastLiveAudioCheckpointRequestedAt = 0
        lastLiveAudioCheckpointPCMBytes = 0
        liveAudioCheckpointBackpressureUntil = 0
        stopWorkbenchStatePolling(clear: true)
        surfaceRecognitionHints = []

        capabilityCheckInFlight = true
        if !startRecorderSession(generation: generation) {
            capabilityCheckInFlight = false
            return
        }
        scheduleNoSpeechWatchdog(generation: generation)
        let runCapabilityCheck = { [weak self] in
            JunoBroker.checkCapability { [weak self] cap in
            guard let self else { return }
            guard self.matchesCurrentDictationGeneration(generation) else { return }
            guard self.capabilityCheckInFlight else { return }
            self.capabilityCheckInFlight = false
            if !cap.ok {
                // Hard-block: can't transcribe at all, or privacy/security concern.
                let hardBlock: Set<String> = [
                    "ax_permission_missing", "broker_unreachable",
                    "helper_not_installed", "helper_timeout",
                    "secure_field", "app_blocked", "window_title_blocked",
                ]
                if hardBlock.contains(cap.reason) {
                    NSLog("Juno: capability gate blocked — \(cap.reason): \(cap.message)")
                    if cap.reason == "ax_permission_missing" {
                        self.promptAccessibilityPermission()
                    }
                    self.cancelNoSpeechWatchdogIfNeeded()
                    self.teardownRecognition()
                    _ = self.recorder.stop()
                    self.state = "blocked:\(cap.reason)"
                    return
                }
                // Soft-block: allow recording, but route the result through
                // the copy-ready overlay because there is no focused text field.
                NSLog("Juno: no paste destination (\(cap.reason)) — will offer copy overlay")
            }
            NSLog(
                "Juno: capability ok=%@ reason=%@ app=%@ text_target=%@",
                cap.ok ? "true" : "false",
                cap.reason,
                cap.appBundleId ?? "unknown",
                cap.hasLikelyTextInsertionPoint ? "true" : "false"
            )

            if self.targetAppBundleId.isEmpty,
               let id = cap.appBundleId,
               !JunoTargetApplicationTracker.isIgnoredSystemSurface(bundleId: id, name: nil) {
                self.targetAppBundleId = id
            }
            if let wt = cap.windowTitle { self.targetWindowTitle = wt }
            self.surfaceRecognitionHints = cap.recognitionHints

            // Best-effort mode label for the HUD (never blocks recording).
            JunoBroker.getJSON(path: "api/broker/modes/current") { obj in
                guard self.matchesCurrentDictationGeneration(generation) else { return }
                let sel = obj["selection"] as? [String: Any] ?? [:]
                let name = (sel["mode_name"] as? String) ?? (sel["mode"] as? String) ?? ""
                let title = JunoUserFacingCopy.builtinModeTitle(id: name)
                self.currentModeLabel = title.isEmpty ? nil : title
            }

            let capabilityTargetIsJunoOrSystem = JunoTargetApplicationTracker.isIgnoredSystemSurface(
                bundleId: cap.appBundleId,
                name: nil
            )
            if capabilityTargetIsJunoOrSystem && !self.targetAppBundleId.isEmpty {
                NSLog(
                    "Juno: preserving existing paste target after capability focused Juno/system surface app=%@",
                    cap.appBundleId ?? "unknown"
                )
            } else {
                self.likelyPasteDestination = cap.hasLikelyTextInsertionPoint
            }

            var snap = self.frozenContextForBroker(from: JunoCapabilitySnapshot.capture())
            snap = JunoSessionContextTape.preservingStartSelectionIfNeeded(
                start: self.utteranceFrozenContext,
                current: snap
            )
            self.utteranceFrozenContext = snap.isEmpty ? nil : snap
            // TOCTOU pin: latch the secure flag from the dictation-start
            // capability snapshot. ``refreshPasteTargetFromCurrentFocus``
            // also latches if a later in-process AX snapshot reports
            // secure; combined, the flag is a sticky one-way close.
            if (snap["focused_is_secure"] as? Bool) == true {
                self.lastSecureFlag = true
            }
            self.sessionContextTape.capture(reason: "capability", base: snap)
            let sel = (snap["selected_text"] as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            self.suppressPartialPasteForSelectionEditing = !sel.isEmpty
            self.editingSelectionCharCount = sel.count
            self.startWorkbenchStatePolling()
        }
        }
        // No session-start activation any more (see comment in the
        // target-resolution block above), so the capability probe can run
        // immediately — no 150ms delay needed for the activated app to
        // settle.
        runCapabilityCheck()
    }

    private func promptAccessibilityPermission() {
        // Dictation hard-block path. Use the system trust sheet (once per
        // session) — that sheet has its own "Open System Settings"
        // button so we don't open Settings ourselves and risk stealing
        // focus mid-paste.
        Task { @MainActor in
            JunoPermissionMonitor.shared.nudgeAccessibilityPrompt()
        }
    }

    /// Bring the recorded paste target to the front before posting Cmd+V.
    ///
    /// The previous implementation called ``activate`` and slept a flat 60 ms
    /// with no verification. Many apps need longer than that to win focus —
    /// any process under load, any app that's just being launched, and most
    /// non-native widget toolkits. When the target hasn't actually come
    /// forward by the time we post Cmd+V, the keystroke lands in whatever is
    /// still frontmost (``paste_frontmost_drifted`` telemetry fires here).
    ///
    /// New behavior: re-check ``frontmostApplication`` after each wait and
    /// keep nudging until the target wins focus or we exhaust the budget.
    /// Total worst-case wait is ~250 ms (60 + 90 + 100), well below the
    /// paste latency the user already tolerates and orders of magnitude
    /// cheaper than the alternative (silent paste to wrong app, user has to
    /// re-dictate).
    ///
    /// Returns ``true`` when the target is frontmost by the end of the call,
    /// ``false`` if the call gave up or the target process disappeared.
    @discardableResult
    private func activateTargetForPasteIfNeeded() -> Bool {
        guard targetPid > 0 else { return false }
        if NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPid {
            return true
        }
        guard let app = NSRunningApplication(processIdentifier: targetPid), !app.isTerminated else {
            return false
        }
        app.activate(options: .activateIgnoringOtherApps)
        // Three checks with increasing waits: covers fast apps (60 ms),
        // typical apps (150 ms cumulative), slow Electron / cold launches
        // (250 ms cumulative). Earlier return as soon as activation lands.
        let waitsMicros: [UInt32] = [60_000, 90_000, 100_000]
        for wait in waitsMicros {
            usleep(wait)
            if NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPid {
                return true
            }
        }
        return NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPid
    }

    /// Re-resolve the paste target against whatever is focused **right now**
    /// using the in-process AX snapshot. Called immediately before each
    /// paste call (and at finalize) so dictation lands wherever the user's
    /// caret currently is — not where it was when they started talking.
    ///
    /// Updates ``targetPid``, ``targetAppBundleId``, ``targetWindowTitle``,
    /// and ``likelyPasteDestination``. Cheap (sub-millisecond AX query);
    /// safe to call from the main thread on the paste path.
    ///
    /// **Why not poll continuously while dictating?** That would create a
    /// race with the broker capability check that captures
    /// ``utteranceFrozenContext`` (the ASR's view of the user's environment
    /// at the moment they started speaking). The frozen context must
    /// match what the user saw when they started dictating — only the
    /// *paste target* needs to track focus changes. So we refresh paste
    /// state exclusively at paste boundaries.
    private func refreshPasteTargetFromCurrentFocus() {
        let snap = JunoLocalCapability.snapshot()
        let snapBundleId = (snap["frontmost_app_bundle_id"] as? String)
            ?? (snap["app_bundle_id"] as? String)
        let snapName = (snap["frontmost_app_name"] as? String)
            ?? (snap["app_name"] as? String)
        switch JunoTargetApplicationTracker.pasteTargetDisposition(bundleId: snapBundleId, name: snapName) {
        case .preservePrior:
            // Juno's own floating HUD panel (canBecomeKey while visible) can be
            // momentarily frontmost at this finalize sample even though the user
            // is dictating into another app (e.g. Brave). Preserve the target
            // captured at dictation start and let activateTargetForPasteIfNeeded
            // re-front it before the Cmd+V — do NOT gate the paste off. Gating
            // off here (regressed in PR #19) mis-filed the result into the
            // copy-ready overlay instead of pasting into the real target.
            return
        case .copyOnly:
            // A genuinely-focused non-editable system pane (System Settings)
            // accepts the Cmd+V keystroke but inserts nothing; fall back to copy.
            likelyPasteDestination = false
            return
        case .adopt:
            break
        }
        // PID/app/window: safe to refresh from NSWorkspace data even
        // without AX trust — these fields land in the snapshot before the
        // AX gate at JunoLocalCapability.snapshot():47.
        if let pidInt = snap["frontmost_pid"] as? Int, pidInt > 0 {
            targetPid = pid_t(pidInt)
        } else if let bid = (snap["frontmost_app_bundle_id"] as? String)
                    ?? (snap["app_bundle_id"] as? String),
                  let app = NSRunningApplication.runningApplications(withBundleIdentifier: bid).first {
            targetPid = app.processIdentifier
        }
        if let bid = (snap["frontmost_app_bundle_id"] as? String)
            ?? (snap["app_bundle_id"] as? String),
           !bid.isEmpty {
            targetAppBundleId = bid
        }
        if let wt = snap["window_title"] as? String, !wt.isEmpty {
            targetWindowTitle = wt
        }
        // ``likelyPasteDestination`` is only safe to override when this
        // snapshot is **authoritative** — i.e. in-process AX is trusted
        // *and* the snapshot actually inspected the focused element.
        // Otherwise ``can_paste_at_focus`` is missing/false-by-default
        // and we'd silently downgrade what the broker already decided
        // via ``juno-capability`` (a separately-signed helper with its
        // own TCC grant). That asymmetry was the root cause of pastes
        // turning into copy-fallback for users who'd granted AX to
        // Juno.app but not to the helper, or vice versa. Preserve the
        // broker's verdict in that case; only refine it when our local
        // probe can speak with authority.
        let hasTrust = (snap["has_ax_trust"] as? Bool) ?? false
        let snapshotOk = (snap["ok"] as? Bool) ?? false
        guard hasTrust && snapshotOk else {
            return
        }
        let secure = (snap["focused_is_secure"] as? Bool) == true
        let canPaste = (snap["can_paste_at_focus"] as? Bool) ?? false
        likelyPasteDestination = !secure && canPaste
        // Pin the secure flag for every privacy gate downstream (paste,
        // learn, history, audio upload). One-way latch within an
        // utterance: if any authoritative AX snapshot reports secure,
        // the gate stays closed even if focus later returns to a
        // non-secure field — preserves the at-start privacy posture
        // and defends against TOCTOU races.
        if secure {
            lastSecureFlag = true
        }
    }

    // MARK: - Push-to-talk exit

    func endPushToTalkAndDictate() {
        NSLog("Juno: dictation stop requested state=%@", state)
        markUtteranceTimeline("hotkey_stop_pressed_ms")
        capabilityCheckInFlight = false
        cancelNoSpeechWatchdogIfNeeded()
        if hudState == .checkingCapability || hudState.isErrorOrBlocked {
            goIdleOnMain()
            teardownRecognition()
            return
        }
        // Accept active capture states (user may stop while still priming the mic).
        guard hudState == .listening
            || hudState == .partialCommit
            || hudState == .checkingMic
            || hudState == .waitingSpeech
        else {
            NSLog("Juno: dictation stop ignored state=%@", state)
            return
        }

        cancelMicWatchdogIfNeeded()
        stopSilenceTimer()
        transientDoneWordCount = nil
        teardownRecognition()

        // Stop the recorder so its WAV is closed cleanly (header
        // finalised). The broker-on-pause flow reads the closed WAV for
        // the final snapshot — see ``fireBrokerSnapshot`` /
        // ``finalizeDictationSession``.
        guard recorder.stop() != nil else {
            NSLog(
                "Juno: recorder stop returned nil frames=%lld pcm_bytes=%d",
                framesReceivedThisSession,
                sessionPCMByteCount()
            )
            goIdleOnMain()
            return
        }
        recorderStopped = true
        NSLog(
            "Juno: recorder stopped frames=%lld pcm_bytes=%d speech_detected=%@",
            framesReceivedThisSession,
            sessionPCMByteCount(),
            hasEverDetectedSpeech ? "true" : "false"
        )
        // We used to drop the session here when the shell-side RMS gate never
        // tripped. That gate uses `max(0.003, ambientNoiseRMS * 2.1)` — in a
        // noisy room it climbs above normal speaking volume and silently kills
        // real dictation. The broker pipeline runs Whisper on the full PCM and
        // returns `low_signal_audio` for true silence, which the finalize path
        // surfaces via `state = "error:low_signal_audio"`. Letting the broker
        // decide is honest; the local gate was guessing.
        refreshFrozenContextAtStop()

        // Flush the trailing PCM chunk to the engine preview before entering
        // final processing. The HUD should show the user's complete spoken
        // text while Whisper/Qwen produce the final answer, otherwise a long
        // utterance looks like words were dropped at stop.
        endEnginePreviewStreaming { [weak self] deliveredFinal in
            guard let self else { return }
            if deliveredFinal {
                self.markUtteranceTimeline("final_preview_flush_received_ms")
            }
            self.syncLiveDisplayTranscript()
            self.refiningStartedAt = Date()
            self.state = "refining"
            // Two paths:
            //   A. We're already coalescing a broker call — let it finish
            //      and queue the final-WAV snapshot via pendingBrokerSnapshot.
            //   B. No call in flight — fire the final snapshot directly.
            // Either way, finalizeDictationSession runs in the broker-call
            // completion handler once the final transcript is applied.
            if self.brokerSnapshotInFlight {
                self.pendingBrokerSnapshot = true
            } else {
                self.fireBrokerSnapshot()
            }
        }
    }

    // MARK: - Mic priming + in-memory PCM

    @discardableResult
    private func startRecorderSession(generation: UInt64) -> Bool {
        do {
            let now = Date().timeIntervalSinceReferenceDate
            sessionStartTime = now
            state = "checking_mic"

            _ = try recorder.start(bufferCallback: { [weak self] buffer in
                guard let self else { return }
                guard self.matchesCurrentDictationGeneration(generation) else { return }
                let totalFrames = self.appendSessionPCMSample(from: buffer)
                let rms = buffer.rms()
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    guard self.matchesCurrentDictationGeneration(generation) else { return }
                    self.recorderMainDidReceiveBuffer(totalFrames: totalFrames, rms: rms)
                }
            })
            dictationStartedAt = Date()
            NSLog("Juno: recorder started input=%@", currentInputDeviceName ?? "unknown")
            scheduleMicWatchdog()
            startSilenceTimer()
            return true
        } catch {
            NSLog("Juno: recorder start failed: \(error.localizedDescription)")
            cancelEnginePreviewStreaming(reason: "recorder_start_failed")
            teardownRecognition()
            state = "error:\(error.localizedDescription)"
            return false
        }
    }

    @discardableResult
    private func appendSessionPCMSample(from buffer: AVAudioPCMBuffer) -> Int64 {
        guard let chunk = buffer.int16MonoInterleavedData() else { return framesReceivedThisSession }
        pcmLock.lock()
        sessionPCMData.append(chunk)
        framesReceivedThisSession += Int64(buffer.frameLength)
        let t = framesReceivedThisSession
        pcmLock.unlock()
        // Push the same PCM frame to the engine preview streamer only when
        // the user wants live transcripts. Final ASR still receives the
        // complete PCM buffer on stop.
        if JunoUserDefaults.hudLiveTranscriptionsEnabled {
            previewStreamer.enqueue(pcm: chunk)
        }
        return t
    }

    // MARK: - Live caption source orchestration

    /// Start engine preview streaming for a fresh utterance. Wires the
    /// streamer's callbacks to update HUD state.
    private func beginEnginePreviewStreaming(uid: String) {
        // Reset the preview buffer; HUD enters the .listening placeholder
        // until the local engine commits text.
        enginePreviewPartialText = ""
        livePartialText = ""
        liveSource = .listening
        lastLiveHUDTextChangeAt = 0
        lastLiveHUDTextChangePCMBytes = sessionPCMByteCount()
        lastPreviewSegmentRollSpeechAt = 0
        liveAudioCheckpointInFlight = false
        lastLiveAudioCheckpointRequestedAt = 0
        lastLiveAudioCheckpointPCMBytes = 0
        liveAudioCheckpointBackpressureUntil = 0
        if JunoScreenContextAccess.isEnabledAndGranted {
            JunoScreenTermHarvester.shared.activate()
        }
        previewStreamer.start(utteranceId: uid)
        previewStreamer.visibleTextHint = { [weak self] in
            guard let self else { return "" }
            return self.rawLivePreviewTextForCorrection()
        }
        previewStreamer.candidateEntities = { [weak self] in
            guard let self else { return [] }
            var hints = self.surfaceRecognitionHints
            var seen = Set(hints.map { $0.lowercased() })
            if JunoScreenContextAccess.isEnabledAndGranted {
                for term in JunoScreenTermHarvester.shared.currentTerms() where !seen.contains(term.lowercased()) {
                    hints.append(term)
                    seen.insert(term.lowercased())
                }
            }
            return hints
        }
        previewStreamer.sessionContextTape = { [weak self] in
            guard let self else { return [:] }
            return self.sessionContextTape.payload(liveTranscript: nil)
        }
        previewStreamer.onPartial = { [weak self] reportedUid, committed, tail, isFinal in
            guard let self else { return }
            guard reportedUid == self.pendingUtteranceId else { return }
            let hasAnyText = !committed.isEmpty || !tail.isEmpty
            if hasAnyText, self.liveSource != .engine {
                self.liveSource = .engine
                self.engineFirstWordTimer?.cancel()
                self.engineFirstWordTimer = nil
            }
            // Root-level preview update. Segment-level LocalAgreement remains
            // append-only; the store accepts only bounded suffix revisions when
            // the root utterance gains better evidence.
            self.applyEnginePreviewChunkToHUD(committed: committed, tail: tail)
            if self.liveSource == .engine {
                let generation = self.dictationSessionGeneration
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) { [weak self] in
                    self?.requestWriterWarmForActiveDictation(
                        generation: generation,
                        reason: "preview_text_visible"
                    )
                }
            }
            _ = isFinal  // Informational; recorder.stop drives finalization.
        }
        previewStreamer.onGiveUp = { [weak self] reportedUid, reason in
            guard let self else { return }
            guard reportedUid == self.pendingUtteranceId else { return }
            // If local preview fails, the HUD stays in its listening state and
            // the final paste at stop remains the source of truth.
            NSLog("Juno preview-streamer give-up uid=\(reportedUid) reason=\(reason) [apple-speech-disabled]")
            self.engineFirstWordTimer?.cancel()
            self.engineFirstWordTimer = nil
        }
        previewStreamer.onBackpressure = { [weak self] reportedUid, reason in
            guard let self else { return }
            guard reportedUid == self.pendingUtteranceId else { return }
            // Caption freshness wins over live correction. A queued/slow
            // preview chunk means the HUD needs scheduling margin for forward
            // motion, so live correction shifts to a slower cadence briefly.
            self.liveAdjudicationBackpressureUntil = max(
                self.liveAdjudicationBackpressureUntil,
                Date.timeIntervalSinceReferenceDate + 0.8
            )
            NSLog("Juno: live correction cadence lowered by preview pressure reason=%@", reason)
        }
        engineFirstWordTimer?.cancel()
        engineFirstWordTimer = nil
    }

    /// First-word fallback is disabled with the LocalAgreement-2 rewrite.
    /// LocalAgreement needs ~1.0 s for its first commit; falling back earlier
    /// made weaker transcripts masquerade as engine output.
    private func armEngineFirstWordTimerIfNeeded() {
        // intentionally no-op
    }

    private func showProvisionalSpeechFallback(reason: String) {
        // intentionally no-op — Apple Speech fallback is disabled.
        _ = reason
    }

    /// Stop preview streaming for the current utterance — called when
    /// dictation finalizes. Sends the trailing chunk so the backend can
    /// flush its decoder.
    private func endEnginePreviewStreaming(completion: ((_ deliveredFinal: Bool) -> Void)? = nil) {
        engineFirstWordTimer?.cancel()
        engineFirstWordTimer = nil
        JunoScreenTermHarvester.shared.deactivate()
        previewStreamer.finish { deliveredFinal in
            completion?(deliveredFinal)
        }
    }

    /// Cancel preview streaming (Esc, error). Drops pending state without
    /// sending a final chunk.
    private func cancelEnginePreviewStreaming(reason: String) {
        engineFirstWordTimer?.cancel()
        engineFirstWordTimer = nil
        JunoScreenTermHarvester.shared.deactivate()
        previewStreamer.cancel(reason: reason)
        liveSource = .none
    }

    private func sessionPCMByteCount() -> Int {
        pcmLock.lock()
        let n = sessionPCMData.count
        pcmLock.unlock()
        return n
    }

    private func scheduleMicWatchdog() {
        micWatchdog?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            guard self.hudState == .checkingMic else { return }
            NSLog("Juno: no microphone frames within \(Self.micNoFrameTimeoutSeconds)s — check permission or input device.")
            self.cancelEnginePreviewStreaming(reason: "mic_no_audio")
            self.teardownRecognition()
            _ = self.recorder.stop()
            self.pcmLock.lock()
            self.sessionPCMData.removeAll(keepingCapacity: false)
            self.framesReceivedThisSession = 0
            self.pcmLock.unlock()
            self.state = "error:mic_no_audio"
            self.dictationStartedAt = nil
            self.stopSilenceTimer()
        }
        micWatchdog = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.micNoFrameTimeoutSeconds, execute: work)
    }

    private func scheduleNoSpeechWatchdog(generation: UInt64) {
        noSpeechWatchdog?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            guard self.matchesCurrentDictationGeneration(generation) else { return }
            guard !self.hasEverDetectedSpeech else { return }
            guard self.hudState == .checkingMic
                || self.hudState == .waitingSpeech
                || self.hudState == .listening
                || self.hudState == .partialCommit
            else { return }
            NSLog(
                "Juno: no speech detected within %.1fs — cancelling stale dictation session",
                Self.noSpeechAutoCancelSeconds
            )
            self.cancelDictation()
        }
        noSpeechWatchdog = work
        DispatchQueue.main.asyncAfter(
            deadline: .now() + Self.noSpeechAutoCancelSeconds,
            execute: work
        )
    }

    private func cancelMicWatchdogIfNeeded() {
        micWatchdog?.cancel()
        micWatchdog = nil
    }

    private func cancelNoSpeechWatchdogIfNeeded() {
        noSpeechWatchdog?.cancel()
        noSpeechWatchdog = nil
    }

    private func recorderMainDidReceiveBuffer(totalFrames: Int64, rms: Float?) {
        cancelMicWatchdogIfNeeded()
        let now = Date.timeIntervalSinceReferenceDate
        if firstAudioFrameAt == 0, totalFrames > 0 {
            firstAudioFrameAt = now
            NSLog(
                "Juno dictation: first audio frames=%lld rms=%@",
                totalFrames,
                rms.map { String(format: "%.5f", $0) } ?? "nil"
            )
        }
        if hudState == .checkingMic {
            state = "waiting_speech"
        }
        if let r = rms {
            let alpha: Float = 0.18
            currentRMS = (1 - alpha) * currentRMS + alpha * r
            if hasEverDetectedSpeech {
                ambientNoiseRMS = (1 - Self.ambientNoiseAlpha) * ambientNoiseRMS
                    + Self.ambientNoiseAlpha * min(r, ambientNoiseRMS * 1.6)
            } else {
                ambientNoiseRMS = (1 - Self.ambientNoiseAlpha) * ambientNoiseRMS + Self.ambientNoiseAlpha * r
            }
            let thr = max(Self.silenceRMSThreshold, ambientNoiseRMS * 2.1)
            if r > thr {
                hasEverDetectedSpeech = true
                cancelNoSpeechWatchdogIfNeeded()
                lastSpeechEnergyAt = now
                updateLastSoundTime(now)
                if !speechDetectedLoggedThisSession {
                    speechDetectedLoggedThisSession = true
                    NSLog(
                        "Juno dictation: speech detected rms=%.5f threshold=%.5f ambient=%.5f",
                        r,
                        thr,
                        ambientNoiseRMS
                    )
                }
                if hudState == .waitingSpeech || hudState == .checkingMic {
                    state = "listening"
                }
                let warmGeneration = dictationSessionGeneration
                let warmDelay: TimeInterval = JunoUserDefaults.hudLiveTranscriptionsEnabled ? 0.75 : 0.0
                DispatchQueue.main.asyncAfter(deadline: .now() + warmDelay) { [weak self] in
                    self?.requestWriterWarmForActiveDictation(
                        generation: warmGeneration,
                        reason: JunoUserDefaults.hudLiveTranscriptionsEnabled
                            ? "speech_detected_live_caption"
                            : "speech_detected_no_live_caption"
                    )
                }
                armEngineFirstWordTimerIfNeeded()
            }
        }
    }

    private func targetApplicationNameForBroker() -> String {
        if let name = targetApp?.name, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return name
        }
        if targetPid > 0,
           let app = NSRunningApplication(processIdentifier: targetPid),
           let name = app.localizedName,
           !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return name
        }
        return targetAppBundleId
    }

    private func frozenContextForBroker(from snap: [String: Any]) -> [String: Any] {
        guard !snap.isEmpty else { return snap }
        var out = snap
        let bid = (snap["frontmost_app_bundle_id"] as? String)
            ?? (snap["app_bundle_id"] as? String)
        let name = (snap["frontmost_app_name"] as? String)
            ?? (snap["app_name"] as? String)
        guard JunoTargetApplicationTracker.isIgnoredSystemSurface(bundleId: bid, name: name) else {
            return out
        }

        if !targetAppBundleId.isEmpty {
            out["frontmost_app_bundle_id"] = targetAppBundleId
            out["app_bundle_id"] = targetAppBundleId
        }
        let targetName = targetApplicationNameForBroker()
        if !targetName.isEmpty {
            out["frontmost_app_name"] = targetName
            out["app_name"] = targetName
        }
        if !targetWindowTitle.isEmpty {
            out["window_title"] = targetWindowTitle
        }
        return out
    }

    private func refreshFrozenContextAtStop() {
        var stopSnap = frozenContextForBroker(from: JunoCapabilitySnapshot.capture())
        stopSnap = JunoSessionContextTape.preservingStartSelectionIfNeeded(
            start: utteranceFrozenContext,
            current: stopSnap
        )
        guard !stopSnap.isEmpty else { return }
        var merged = utteranceFrozenContext ?? [:]
        for (key, value) in stopSnap {
            merged[key] = value
        }
        let stopSecure = (stopSnap["focused_is_secure"] as? Bool) == true
            || (stopSnap["secure_field"] as? Bool) == true
        if lastSecureFlag || stopSecure {
            lastSecureFlag = true
            merged["focused_is_secure"] = true
            merged["secure_field"] = true
        }
        utteranceFrozenContext = merged
        sessionContextTape.capture(reason: "stop_context", base: merged)
    }

    private func hostHintsForBroker() -> [String: Any] {
        let snap = utteranceFrozenContext ?? [:]
        let sel = ((snap["selected_text"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let focused = ((snap["focused_text"] as? String) ?? (snap["focused_value"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let clip = ((snap["clipboard_text"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let snapAppName = snap["frontmost_app_name"] as? String ?? snap["app_name"] as? String ?? ""
        var o: [String: Any] = [
            "surface": "mac_overlay",
            "app_bundle_id": targetAppBundleId,
            "app_name": snapAppName.isEmpty ? targetApplicationNameForBroker() : snapAppName,
            "window_title": targetWindowTitle,
            "pid": Int(targetPid),
            "locale_identifier": Locale.current.identifier,
            "has_selected_text": !sel.isEmpty,
            "has_focused_text": !focused.isEmpty,
            "has_clipboard_text": !clip.isEmpty,
        ]
        if let p = snap["focused_document_path"] as? String, !p.isEmpty { o["focused_document_path"] = p }
        return o
    }

    // MARK: - Silence detection

    private func startSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.checkSilence()
        }
    }

    private func stopSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = nil
    }

    private func checkSilence() {
        guard hudState == .listening else { return }
        guard hasEverDetectedSpeech else { return }

        let now = Date().timeIntervalSinceReferenceDate
        guard now - sessionStartTime > effectiveGraceSeconds else { return }
        rollEnginePreviewSegmentIfNeeded(now: now)
        requestLiveAdjudicationIfNeeded(now: now)
        // Recovery path only: this function has its own stale-HUD, fresh-speech,
        // min-audio, and min-new-audio gates. It rescues cases where both
        // engine preview and fallback text stop moving mid-utterance.
        requestLiveAudioCheckpointIfNeeded(now: now)
    }

    private func rollEnginePreviewSegmentIfNeeded(now: TimeInterval) {
        guard JunoUserDefaults.hudLiveTranscriptionsEnabled else { return }
        guard hasEverDetectedSpeech, lastSpeechEnergyAt > sessionStartTime else { return }
        let segmentAge = previewStreamer.activeSegmentAge ?? 0
        let pauseDue = segmentAge >= Self.previewSegmentMinPauseRollSeconds
            && now - lastSpeechEnergyAt >= Self.previewSegmentPauseSeconds
            && lastSpeechEnergyAt > lastPreviewSegmentRollSpeechAt
        let ageDue = segmentAge >= Self.previewSegmentMaxSeconds
            && liveSource == .engine
        guard pauseDue || ageDue else { return }
        lastPreviewSegmentRollSpeechAt = lastSpeechEnergyAt
        previewStreamer.rollSegment(reason: pauseDue ? "speech_pause" : "segment_age")
    }

    private func requestLiveAdjudicationIfNeeded(now: TimeInterval) {
        guard JunoUserDefaults.hudLiveTranscriptionsEnabled else { return }
        // Live in-speech snapshots are correction overlays for stable HUD
        // text. They must remain bounded and gated here; the broker also
        // coalesces/cancels stale live jobs so final stop delivery can run
        // ahead of old correction work.
        guard JunoUserDefaults.liveAdjudicationEnabled else { return }
        let previewBackpressureActive = now < liveAdjudicationBackpressureUntil
        guard liveAdjudicationSlowCount < 2 else {
            // If the final ASR lane is slow this session, avoid continuous
            // in-speech work and only let the final stop pass use it.
            guard lastSpeechEnergyAt > sessionStartTime,
                  lastSpeechEnergyAt > lastLiveAdjudicationSpeechAt,
                  now - lastSpeechEnergyAt > 0.9
            else { return }
            fireLiveAdjudicationSnapshot(reason: "pause_backoff")
            return
        }

        let visible = rawLivePreviewTextForCorrection()
        guard !visible.isEmpty else { return }
        let words = visible.split { $0.isWhitespace || $0.isNewline }.count
        let chars = visible.count
        guard words > 0 else { return }
        let elapsed = now - sessionStartTime
        let sinceLast = now - lastLiveAdjudicationRequestedAt
        let newWords = words - lastLiveAdjudicationWordCount

        let firstPassDue = lastLiveAdjudicationRequestedAt == 0
            && (words >= 3 || elapsed >= 0.8)
        let pauseDue = lastSpeechEnergyAt > sessionStartTime
            && lastSpeechEnergyAt > lastLiveAdjudicationSpeechAt
            && now - lastSpeechEnergyAt > 0.5
        var cadence: TimeInterval
        let minNewWords: Int
        switch elapsed {
        case ..<5:
            cadence = 0.6
            minNewWords = 2
        case ..<30:
            cadence = 1.0
            minNewWords = 3
        default:
            cadence = 1.5
            minNewWords = 4
        }
        let continuousDue = lastLiveAdjudicationRequestedAt > 0
            && sinceLast >= cadence
            && (newWords >= minNewWords
                || chars - lastLiveAdjudicationCharCount >= 35)

        guard sinceLast >= 0.4 else { return }
        if previewBackpressureActive {
            cadence = max(cadence, 1.8)
            guard pauseDue || (continuousDue && sinceLast >= cadence) else { return }
        }
        guard firstPassDue || pauseDue || continuousDue else { return }
        let reason = pauseDue ? "pause" : (firstPassDue ? "initial" : "continuous")
        fireLiveAdjudicationSnapshot(reason: reason)
    }

    private func requestLiveAudioCheckpointIfNeeded(now: TimeInterval) {
        guard JunoUserDefaults.hudLiveTranscriptionsEnabled else { return }
        // Whisper preview now owns a bounded rolling transcript session. The
        // old checkpoint path sends cumulative full-session audio through the
        // final broker lane while the user is still speaking, which can block
        // final-stop delivery behind an already-running ASR decode. Keep it as
        // a hidden diagnostic opt-in only.
        guard UserDefaults.standard.bool(forKey: "JunoLiveAudioCheckpointEnabled") else { return }
        guard liveSource == .engine else { return }
        guard !liveAudioCheckpointInFlight else { return }
        guard now >= liveAudioCheckpointBackpressureUntil else { return }
        guard !recorderStopped,
              (hudState == .listening || hudState == .waitingSpeech || hudState == .partialCommit),
              !cancelInFlightBrokerInsertion
        else { return }
        guard hasEverDetectedSpeech else { return }
        guard now - sessionStartTime >= Self.liveAudioCheckpointMinAudioSeconds else { return }
        guard lastSpeechEnergyAt > sessionStartTime else { return }

        let visible = rawLivePreviewTextForCorrection()
        guard !visible.isEmpty else { return }
        let lastVisibleChangeAt = lastLiveHUDTextChangeAt > sessionStartTime
            ? lastLiveHUDTextChangeAt
            : sessionStartTime
        guard now - lastVisibleChangeAt >= Self.liveAudioCheckpointStaleAfter else { return }
        let speechFresh = now - lastSpeechEnergyAt <= Self.liveAudioCheckpointSpeechFreshAfter
        let hudStaleEnoughForRecovery =
            now - lastVisibleChangeAt >= Self.liveAudioCheckpointStaleSpeechBypassAfter
        guard speechFresh || hudStaleEnoughForRecovery else { return }

        pcmLock.lock()
        let pcm = sessionPCMData
        pcmLock.unlock()
        let minBytes = Int(JunoSessionWAVBuilder.sampleRate * 2.0 * Self.liveAudioCheckpointMinAudioSeconds)
        let minNewBytes = Int(JunoSessionWAVBuilder.sampleRate * 2.0 * Self.liveAudioCheckpointMinNewAudioSeconds)
        guard pcm.count >= minBytes else { return }
        let newBytesSinceCheckpoint = pcm.count - min(lastLiveAudioCheckpointPCMBytes, pcm.count)
        let newBytesSinceHUDChange = pcm.count - min(lastLiveHUDTextChangePCMBytes, pcm.count)
        guard newBytesSinceCheckpoint >= minNewBytes else { return }
        guard newBytesSinceHUDChange >= minNewBytes else { return }
        let compactedUpload = JunoPCMUploadCompactor.compact(pcm)
        let uploadPCM = compactedUpload.pcm
        let minInterval = liveAudioCheckpointMinInterval(forPCMBytes: uploadPCM.count)
        guard now - lastLiveAudioCheckpointRequestedAt >= minInterval else { return }
        guard SecureFieldPolicy.allowAudioSave(secure: lastSecureFlag) else {
            NSLog("Juno: live audio checkpoint suppressed by SecureFieldPolicy (audio gate)")
            return
        }

        liveAudioCheckpointInFlight = true
        lastLiveAudioCheckpointRequestedAt = now
        lastLiveAudioCheckpointPCMBytes = pcm.count
        sessionContextTape.capture(reason: "live_audio_checkpoint")
        let generation = dictationSessionGeneration
        let wavBytes = JunoSessionWAVBuilder.wavData(fromInt16MonoLittleEndian: uploadPCM)
        let requestStartedAt = Date.timeIntervalSinceReferenceDate
        NSLog(
            "Juno: live audio checkpoint request source=%@ pcm_bytes=%d upload_pcm_bytes=%d dropped_ms=%d retained_regions=%d active_windows=%d visible_chars=%d stale_ms=%d speech_gap_ms=%d min_interval_ms=%d",
            liveSource.rawValue,
            pcm.count,
            uploadPCM.count,
            Int(compactedUpload.droppedDurationSeconds * 1000.0),
            compactedUpload.retainedRegionCount,
            compactedUpload.activeWindowCount,
            visible.count,
            Int((now - lastVisibleChangeAt) * 1000.0),
            Int((now - lastSpeechEnergyAt) * 1000.0),
            Int(minInterval * 1000.0)
        )

        JunoBroker.transcribeWav(
            wavData: wavBytes,
            appBundleId: targetAppBundleId.isEmpty ? nil : targetAppBundleId,
            windowTitle: targetWindowTitle.isEmpty ? nil : targetWindowTitle,
            utteranceId: pendingUtteranceId.isEmpty ? nil : pendingUtteranceId,
            frozenContext: utteranceFrozenContext,
            hostHints: hostHintsForBroker(),
            shellTimeline: utteranceTimelineMs,
            surfaceId: "mac_overlay",
            transcriptStage: "live_adjudication",
            sessionContextTape: sessionContextTape.payload(liveTranscript: visible),
            transcriptHint: nil,
            languageMode: JunoUserDefaults.languageMode
        ) { [weak self] result in
            guard let self else { return }
            DispatchQueue.main.async {
                self.liveAudioCheckpointInFlight = false
                guard self.matchesCurrentDictationGeneration(generation) else { return }
                let completedAt = Date.timeIntervalSinceReferenceDate
                let latency = completedAt - requestStartedAt
                if latency >= Self.liveAudioCheckpointSlowResponseThreshold {
                    let backpressure = min(Self.liveAudioCheckpointMaxBackpressure, latency)
                    self.liveAudioCheckpointBackpressureUntil = max(
                        self.liveAudioCheckpointBackpressureUntil,
                        completedAt + backpressure
                    )
                    NSLog(
                        "Juno: live audio checkpoint backpressure latency_ms=%d backpressure_ms=%d",
                        Int(latency * 1000.0),
                        Int(backpressure * 1000.0)
                    )
                }
                guard !self.recorderStopped,
                      (self.hudState == .listening || self.hudState == .waitingSpeech || self.hudState == .partialCommit),
                      !self.cancelInFlightBrokerInsertion
                else { return }
                switch result {
                case .success(let response):
                    self.applyLiveAudioCheckpointTranscript(response)
                case .failure(let error):
                    let nsError = error as NSError
                    if (nsError.userInfo["JunoCancelled"] as? Bool) == true {
                        NSLog("Juno: live audio checkpoint cancelled reason=%@", nsError.localizedDescription)
                    } else {
                        NSLog("Juno: live audio checkpoint failed: %@", nsError.localizedDescription)
                    }
                }
            }
        }
    }

    private func liveAudioCheckpointMinInterval(forPCMBytes pcmBytes: Int) -> TimeInterval {
        let bytesPerSecond = Double(JunoSessionWAVBuilder.sampleRate) * 2.0
        let audioSeconds = Double(max(0, pcmBytes)) / bytesPerSecond
        if audioSeconds >= 120.0 {
            return 12.0
        }
        if audioSeconds >= 60.0 {
            return 8.0
        }
        return Self.liveAudioCheckpointMinInterval
    }

    private func applyLiveAudioCheckpointTranscript(_ response: JunoBroker.TranscribeResponse) {
        // Audio checkpoint = a Qwen-corrected snapshot of the current visible
        // text returned by the broker mid-utterance. Under the new
        // LocalAgreement-2 contract, mid-utterance Qwen correction is disabled
        // by default; this path is retained only for the legacy audio
        // checkpoint flow if it still fires. The preview-revision guard accepts
        // suffix corrections but refuses unrelated stale checkpoints.
        let trimmed = normalizedHUDText(response.transcript)
        guard !trimmed.isEmpty else { return }
        cancelHUDCommittedReveal()
        let previous = hudTranscriptStore.text
        let accepted = hudTranscriptStore.applyPreviewRevision(committed: trimmed, tail: "")
        guard accepted else {
            NSLog(
                "Juno: live audio checkpoint rejected current_chars=%d incoming_chars=%d",
                previous.count,
                trimmed.count
            )
            return
        }
        livePartialText = hudTranscriptStore.text
        syncHUDFromTranscriptStore()
        if previous != hudTranscriptStore.text {
            bumpCorrectionGeneration()
        }
        NSLog(
            "Juno: live audio checkpoint applied chars=%d",
            hudTranscriptStore.text.count
        )
    }

    private func fireLiveAdjudicationSnapshot(reason: String) {
        let now = Date().timeIntervalSinceReferenceDate
        let visible = rawLivePreviewTextForCorrection()
        guard !visible.isEmpty else { return }
        guard visible != lastLiveAdjudicationVisibleText else { return }
        let snapshot = LiveAdjudicationSnapshot(
            visibleText: visible,
            reason: reason,
            createdAt: now,
            speechAt: lastSpeechEnergyAt,
            wordCount: visible.split { $0.isWhitespace || $0.isNewline }.count,
            charCount: visible.count
        )
        if liveAdjudicationInFlight {
            pendingLiveAdjudicationReason = reason
            pendingLiveAdjudicationSnapshot = snapshot
            return
        }
        sendLiveAdjudicationSnapshot(snapshot)
    }

    private func sendLiveAdjudicationSnapshot(_ snapshot: LiveAdjudicationSnapshot) {
        let visible = snapshot.visibleText
        guard !visible.isEmpty else { return }
        guard visible != lastLiveAdjudicationVisibleText else { return }
        let now = Date().timeIntervalSinceReferenceDate
        liveAdjudicationInFlight = true
        pendingLiveAdjudicationReason = nil
        pendingLiveAdjudicationSnapshot = nil
        lastLiveAdjudicationRequestedAt = now
        lastLiveAdjudicationSpeechAt = snapshot.speechAt
        lastLiveAdjudicationWordCount = snapshot.wordCount
        lastLiveAdjudicationCharCount = snapshot.charCount
        lastLiveAdjudicationVisibleText = visible
        sessionContextTape.capture(reason: "live_\(snapshot.reason)")

        let generation = dictationSessionGeneration
        let requestStartedAt = now
        JunoBroker.correctLiveTranscript(
            visibleText: visible,
            appBundleId: targetAppBundleId.isEmpty ? nil : targetAppBundleId,
            windowTitle: targetWindowTitle.isEmpty ? nil : targetWindowTitle,
            utteranceId: pendingUtteranceId.isEmpty ? nil : pendingUtteranceId,
            frozenContext: utteranceFrozenContext,
            hostHints: hostHintsForBroker(),
            shellTimeline: utteranceTimelineMs,
            surfaceId: "mac_overlay",
            sessionContextTape: sessionContextTape.payload(liveTranscript: visible),
            languageMode: JunoUserDefaults.languageMode
        ) { [weak self] result in
            guard let self else { return }
            DispatchQueue.main.async {
                guard self.matchesCurrentDictationGeneration(generation) else { return }
                self.liveAdjudicationInFlight = false
                self.liveSpeechHint = nil
                guard !self.recorderStopped,
                      (self.hudState == .listening || self.hudState == .waitingSpeech),
                      !self.cancelInFlightBrokerInsertion
                else {
                    self.pendingLiveAdjudicationReason = nil
                    self.pendingLiveAdjudicationSnapshot = nil
                    return
                }
                let latency = Date().timeIntervalSinceReferenceDate - requestStartedAt
                if latency > 2.5 {
                    self.liveAdjudicationSlowCount += 1
                } else if latency < 0.8 {
                    self.liveAdjudicationSlowCount = max(0, self.liveAdjudicationSlowCount - 1)
                }
                switch result {
                case .success(let response):
                    self.applyLiveAdjudicatedTranscript(response)
                case .failure(let error):
                    let nsError = error as NSError
                    if (nsError.userInfo["JunoCancelled"] as? Bool) == true {
                        NSLog("Juno: live adjudication cancelled reason=%@", nsError.localizedDescription)
                    } else {
                        NSLog("Juno: live adjudication failed: %@", nsError.localizedDescription)
                    }
                }
                if let pending = self.pendingLiveAdjudicationSnapshot {
                    self.pendingLiveAdjudicationReason = nil
                    self.pendingLiveAdjudicationSnapshot = nil
                    self.schedulePendingLiveAdjudicationSnapshot(pending)
                }
            }
        }
    }

    private func schedulePendingLiveAdjudicationSnapshot(_ snapshot: LiveAdjudicationSnapshot) {
        let now = Date().timeIntervalSinceReferenceDate
        let minGap: TimeInterval = liveAdjudicationSlowCount > 0 ? 0.45 : 0.18
        let delay = max(0, minGap - (now - lastLiveAdjudicationRequestedAt))
        let generation = dictationSessionGeneration
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            guard self.matchesCurrentDictationGeneration(generation) else { return }
            guard !self.liveAdjudicationInFlight else {
                self.pendingLiveAdjudicationSnapshot = snapshot
                self.pendingLiveAdjudicationReason = snapshot.reason
                return
            }
            guard !self.recorderStopped,
                  (self.hudState == .listening || self.hudState == .waitingSpeech),
                  !self.cancelInFlightBrokerInsertion
            else { return }
            self.sendLiveAdjudicationSnapshot(snapshot)
        }
    }

    // MARK: - Pause commit (broker-driven)

    /// Legacy entry point kept inert for older call sites while the new
    /// flow settles. Pauses now request live adjudication only; broker
    /// delivery and paste are final-stop only.
    private func commitPartialOnPause() {
        requestLiveAdjudicationIfNeeded(now: Date().timeIntervalSinceReferenceDate)
    }

    // MARK: - Broker-on-pause helpers

    /// Send cumulative in-memory PCM as a standalone WAV (no race with ``AVAudioFile`` on disk).
    private func fireBrokerSnapshot() {
        // Privacy gate (4/5): never upload the session WAV when the
        // in-flight utterance started against a secure field. The
        // broker may persist or further process the audio downstream;
        // shell defense-in-depth is mandatory. See ``JunoSecureFieldPolicy``.
        guard SecureFieldPolicy.allowAudioSave(secure: lastSecureFlag) else {
            NSLog("Juno: broker snapshot WAV upload suppressed by SecureFieldPolicy (audio gate)")
            // Finalize cleanly without round-tripping through the
            // broker so the HUD does not get stuck in ``refining``.
            if recorderStopped {
                finalizeDictationSession(brokerSucceeded: false)
            }
            return
        }
        if brokerSnapshotInFlight {
            pendingBrokerSnapshot = true
            return
        }

        let generation = dictationSessionGeneration
        let isFinal = recorderStopped
        if !isFinal {
            NSLog("Juno: ignored non-final broker delivery snapshot; pauses use live adjudication only")
            requestLiveAdjudicationIfNeeded(now: Date().timeIntervalSinceReferenceDate)
            return
        }
        pcmLock.lock()
        let pcm = sessionPCMData
        pcmLock.unlock()
        let snapshotPCMByteCount = pcm.count
        let snapshotSpeechAt = lastPartialSnapshotSpeechAt

        guard pcm.count >= JunoSessionWAVBuilder.minPCMBytesForBroker else {
            NSLog(
                "Juno: broker snapshot skipped short_pcm final=%@ pcm_bytes=%d min_bytes=%d frames=%lld",
                isFinal ? "true" : "false",
                pcm.count,
                JunoSessionWAVBuilder.minPCMBytesForBroker,
                framesReceivedThisSession
            )
            if isFinal {
                finalizeDictationSession(brokerSucceeded: false)
            } else if framesReceivedThisSession > 0 {
                liveSpeechHint = "Still listening…"
            }
            return
        }

        let compactedUpload = JunoPCMUploadCompactor.compact(pcm)
        let uploadPCM = compactedUpload.pcm
        let wavBytes = JunoSessionWAVBuilder.wavData(fromInt16MonoLittleEndian: uploadPCM)
        brokerSnapshotInFlight = true
        sessionContextTape.capture(reason: isFinal ? "final" : "legacy_snapshot")
        let frozenSnap = utteranceFrozenContext
        let uid = pendingUtteranceId.isEmpty ? nil : pendingUtteranceId
        let hints = hostHintsForBroker()
        let brokerRequestSentMs = Self.currentWallClockMs()
        markUtteranceTimeline("final_broker_request_sent_ms", at: brokerRequestSentMs)
        let shellTimeline = utteranceTimelinePayload()
        let finalTranscriptHintCandidates = [
            hudTranscriptStore.rawText,
            liveDisplayTranscript,
            livePartialText,
            engineSessionPartialText,
            engineSessionFinalCandidateText,
        ]
            .map { normalizedHUDText($0) }
        let finalTranscriptHint = finalTranscriptHintCandidates.max(by: { $0.count < $1.count }) ?? ""
        NSLog(
            "Juno: broker snapshot request final=%@ pcm_bytes=%d upload_pcm_bytes=%d dropped_ms=%d retained_regions=%d active_windows=%d utterance=%@",
            isFinal ? "true" : "false",
            pcm.count,
            uploadPCM.count,
            Int(compactedUpload.droppedDurationSeconds * 1000.0),
            compactedUpload.retainedRegionCount,
            compactedUpload.activeWindowCount,
            uid ?? "none"
        )

        JunoBroker.transcribeWav(
            wavData: wavBytes,
            appBundleId: targetAppBundleId.isEmpty ? nil : targetAppBundleId,
            windowTitle: targetWindowTitle.isEmpty ? nil : targetWindowTitle,
            utteranceId: uid,
            frozenContext: frozenSnap,
            hostHints: hints,
            shellTimeline: shellTimeline,
            surfaceId: "mac_overlay",
            transcriptStage: "final_delivery",
            sessionContextTape: sessionContextTape.payload(liveTranscript: finalTranscriptHint),
            transcriptHint: finalTranscriptHint.isEmpty ? nil : finalTranscriptHint,
            languageMode: JunoUserDefaults.languageMode
        ) { [weak self] result in
            guard let self else { return }
            DispatchQueue.main.async {
                guard self.matchesCurrentDictationGeneration(generation) else { return }
                self.brokerSnapshotInFlight = false
                self.markUtteranceTimeline("final_broker_response_received_ms")
                var brokerOk = false
                var consumedByAction = false
                switch result {
                case .success(let r):
                    let trimmed = r.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                    NSLog(
                        "Juno: broker snapshot ok final=%@ transcript_len=%d",
                        isFinal ? "true" : "false",
                        trimmed.count
                    )
                    // Audit Issue #1 — preview→final reconciliation. When
                    // the engine attached a final-stage transcript_patch_v1
                    // envelope, drive the HUD store through the ops path so
                    // the unchanged prefix keeps its preview span identities
                    // ("fix in place") instead of hard-cutting to the new
                    // text. ``applyPatchEnvelope`` falls back to a hard
                    // replace internally when the snapshot can't be
                    // reconciled, and older engines that don't send a patch
                    // skip this branch entirely — ``applyBrokerTranscript``
                    // remains the legacy path for those cases.
                    if !trimmed.isEmpty,
                       let patch = r.transcriptPatch,
                       patch.stage == "final" {
                        self.cancelHUDCommittedReveal()
                        _ = self.hudTranscriptStore.applyPatchEnvelope(patch)
                        self.engineSessionFinalCandidateText = self.hudTranscriptStore.text
                        self.livePartialText = self.hudTranscriptStore.text
                        self.syncHUDFromTranscriptStore()
                    }
                    if self.responseRepresentsNoPasteAction(r) || !trimmed.isEmpty {
                        consumedByAction = self.applyBrokerTranscript(
                            r.transcript,
                            response: r,
                            isFinal: isFinal,
                            snapshotPCMByteCount: snapshotPCMByteCount,
                            snapshotSpeechAt: snapshotSpeechAt
                        )
                    }
                    brokerOk = true
                case .failure(let error):
                    self.brokerSnapshotFailedThisSession = true
                    let nsError = error as NSError
                    let errorCode = (nsError.userInfo["JunoErrorCode"] as? String) ?? ""
                    if errorCode == "low_signal_audio" {
                        self.brokerSnapshotLowSignalThisSession = true
                        self.liveSpeechHint = "No clear speech detected."
                    }
                    NSLog("Juno: broker snapshot failed: \(error.localizedDescription)")
                }

                if !isFinal && self.pendingBrokerSnapshot {
                    self.pendingBrokerSnapshot = false
                    let hasNewSpeechSinceSnapshot = self.lastSpeechEnergyAt > snapshotSpeechAt
                    if consumedByAction && self.recorderStopped && !hasNewSpeechSinceSnapshot {
                        self.goIdleOnMain()
                        return
                    }
                    let hasUsableSnapshotText =
                        !self.normalizedHUDText(self.pendingRevisionFullTranscript).isEmpty
                        || !self.normalizedHUDText(self.lastPastedFromBroker).isEmpty
                        || !self.normalizedHUDText(self.engineSessionFinalCandidateText).isEmpty
                    if self.recorderStopped && brokerOk && hasUsableSnapshotText && !hasNewSpeechSinceSnapshot {
                        self.finalizeDictationSession(brokerSucceeded: true)
                        return
                    }
                    self.fireBrokerSnapshot()
                    return
                }
                if isFinal {
                    if consumedByAction {
                        self.goIdleOnMain()
                    } else {
                        self.finalizeDictationSession(brokerSucceeded: brokerOk)
                    }
                }
            }
        }
    }

    private func responseRepresentsNoPasteAction(_ response: JunoBroker.TranscribeResponse) -> Bool {
        response.isAction && response.pasteKind == "none"
    }

    /// Apply a broker transcript. Final snapshots are staged for the stop/finalize path;
    /// pause snapshots are committed as one complete utterance and the consumed audio is
    /// trimmed so dictation can continue in the same HUD session.
    @discardableResult
    private func applyBrokerTranscript(
        _ transcript: String,
        response: JunoBroker.TranscribeResponse,
        isFinal: Bool,
        snapshotPCMByteCount: Int,
        snapshotSpeechAt: TimeInterval
    ) -> Bool {
        if cancelInFlightBrokerInsertion { return false }
        engineSessionPartialText = ""
        engineSessionFinalCandidateText = transcript
        livePartialText = transcript

        // Voice Actions. Pure action turns suppress paste; mixed turns may
        // carry both actions and writer-rendered text, in which case the
        // broker sets paste_kind to insert/replace and we continue into the
        // normal insertion path after dispatching the action batch.
        //
        // Hard rule from §"Feature is purely additive": when the toggle is
        // OFF, this block is a no-op and dictation behaves exactly as
        // today. The non-action paste path below runs unchanged for plain
        // dictation regardless of the toggle.
        if response.isAction {
            let toggleOn = JunoUserDefaults.actionsEnabled
            if !toggleOn {
                // Voice Actions are purely additive. When the feature is
                // off, action-like utterances should behave like normal
                // dictation instead of disappearing into a blocked state.
                return false
            }

            if let actions = response.actions,
               !actions.isEmpty,
               let uid = response.utteranceId, !uid.isEmpty
            {
                Task { @MainActor in
                    JunoActionExecutor.shared.execute(
                        utteranceId: uid,
                        actions: actions,
                        completion: { [weak self] results in
                            self?.showActionHUDResult(results)
                        }
                    )
                }
                let shouldPasteMixedText =
                    response.pasteKind != "none"
                    && !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                if !shouldPasteMixedText {
                    lastPastedFromBroker = ""
                    accumulatedText = ""
                    pendingRevisionFullTranscript = nil
                    copyableTranscript = nil
                    syncLiveDisplayTranscript()
                    if !isFinal {
                        resetForNextPauseUtterance(
                            snapshotPCMByteCount: snapshotPCMByteCount,
                            snapshotSpeechAt: snapshotSpeechAt
                        )
                    }
                    return true
                }
                // Continue into the normal insertion path below.
            } else {
                // The engine made an explicit action/no-paste decision, but
                // the concrete action payload was missing or could not be
                // decoded. Still suppress the literal command text; pasting
                // "Juno take a note..." is worse than a quiet failed action.
                liveSpeechHint = "Juno action could not run."
                let recovered = (response.recoverableTranscript ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                showTransientActionHUD(
                    title: "Action could not run",
                    subtitle: recovered.isEmpty
                        ? "Try again from the Actions page after checking setup."
                        : "Your words are kept — tap to copy, or see History.",
                    symbolName: "exclamationmark.triangle.fill",
                    isFailure: true
                )
                lastPastedFromBroker = ""
                accumulatedText = ""
                pendingRevisionFullTranscript = nil
                // Never erase the user's words: rejected actions keep the
                // spoken text recoverable from the copy surface + History.
                copyableTranscript = recovered.isEmpty ? nil : recovered
                syncLiveDisplayTranscript()
                if !isFinal {
                    resetForNextPauseUtterance(
                        snapshotPCMByteCount: snapshotPCMByteCount,
                        snapshotSpeechAt: snapshotSpeechAt
                    )
                }
                return true
            }
        }

        if !isFinal, let holdReason = structuredNotesPauseHoldReason(for: transcript) {
            NSLog("Juno: holding structured Notes pause transcript — %@", holdReason)
            pendingRevisionFullTranscript = transcript
            lastPastedFromBroker = ""
            accumulatedText = ""
            liveSpeechHint = holdReason
            syncLiveDisplayTranscript()
            return false
        }

        if isFinal || suppressPartialPasteForSelectionEditing {
            if isFinal {
                pendingFinalBrokerPasteKind = response.pasteKind
                if let writerOutcome = response.metadata?["writer_outcome"] as? [String: Any] {
                    pendingFinalReplaceTarget = writerOutcome["target"] as? String
                    if let n = writerOutcome["target_text_chars"] as? Int {
                        pendingFinalReplaceTargetChars = max(0, n)
                    } else if let n = writerOutcome["target_text_chars"] as? NSNumber {
                        pendingFinalReplaceTargetChars = max(0, n.intValue)
                    } else {
                        pendingFinalReplaceTargetChars = 0
                    }
                } else {
                    pendingFinalReplaceTarget = nil
                    pendingFinalReplaceTargetChars = 0
                }
            }
            lastPastedFromBroker = transcript
            accumulatedText = transcript
            pendingRevisionFullTranscript = nil
            syncLiveDisplayTranscript()
            // Audit fix V-4 / A2: when the engine reports degraded_writer
            // (the LLM writer lane failed to load, so polish modes are
            // running in deterministic-cleanup-only fallback), surface a
            // quiet HUD notice so the user has a signal instead of
            // silently getting unpolished output.
            if response.degradedWriter {
                writerDegradedNotice = true
            }
            return false
        }

        if let holdReason = structuredNotesPauseHoldReason(for: transcript) {
            pendingRevisionFullTranscript = transcript
            liveSpeechHint = "Structuring note…"
            syncLiveDisplayTranscript()
            NSLog(
                "Juno: holding structured Notes pause snapshot reason=%@ transcript_len=%d",
                holdReason,
                transcript.trimmingCharacters(in: .whitespacesAndNewlines).count
            )
            return false
        }

        commitPauseTranscript(
            transcript,
            snapshotPCMByteCount: snapshotPCMByteCount,
            snapshotSpeechAt: snapshotSpeechAt
        )
        return false
    }

    private func structuredNotesPauseHoldReason(for transcript: String) -> String? {
        guard isAppleNotesTarget() else { return nil }
        let compact = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !compact.isEmpty else { return nil }
        let lower = compact.lowercased()
        let hasThreeThingsLead = lower.range(
            of: "\\b(?:3|three)\\s+things?\\b",
            options: .regularExpression
        ) != nil

        if regexContains(#"(?:^|\s)(?:[1-9]|10)[.)](?:\s+(?:[1-9]|10)[.)]){4,}"#, in: compact) {
            return "Waiting for a clean list…"
        }
        guard hasThreeThingsLead else { return nil }

        let hasFirst = containsWord("first", in: compact)
        let hasSecond = containsWord("second", in: compact) || containsNumberedListMarker(2, in: compact)
        let hasThird = containsWord("third", in: compact) || containsNumberedListMarker(3, in: compact)

        if hasThreeThingsLead && hasIncompleteThirdItem(compact) {
            return "Waiting for the third item…"
        }
        if hasThreeThingsLead && !hasCompleteThreeItemList(compact) {
            return "Waiting for all 3 items…"
        }
        guard hasFirst && hasSecond else { return nil }

        if !hasThird {
            return "Waiting for the third item…"
        }
        if lower.hasSuffix("that's") || lower.hasSuffix("thats") || lower.hasSuffix("and") {
            return "Finishing the note…"
        }
        return nil
    }

    private func isAppleNotesTarget() -> Bool {
        let bundleIds = [
            targetAppBundleId,
            utteranceFrozenContext?["frontmost_app_bundle_id"] as? String ?? "",
            utteranceFrozenContext?["app_bundle_id"] as? String ?? "",
        ]
        if bundleIds.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "com.apple.notes" }) {
            return true
        }
        let title = targetWindowTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return title.contains("notes")
    }

    private func hasCompleteThreeItemList(_ text: String) -> Bool {
        regexContains(#"(?s)\b1[.)]\s+\S.+\b2[.)]\s+\S.+\b3[.)]\s+\S"#, in: text)
            || regexContains(
                #"(?s)\bfirst\b.+\bsecond\b.+\bthird(?:ly)?(?:\s+and\s+last|\s+last)?(?:\s+is)?[,;:\s]+\S"#,
                in: text
            )
    }

    private func hasIncompleteThirdItem(_ text: String) -> Bool {
        regexContains(
            #"(?s)\b(?:third(?:ly)?(?:\s+and\s+last|\s+last)?(?:\s+is)?|3[.)])\s*[,;:]?\s*$"#,
            in: text
        )
    }

    private func regexContains(_ pattern: String, in text: String) -> Bool {
        guard let re = try? NSRegularExpression(
            pattern: pattern,
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) else {
            return false
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return re.firstMatch(in: text, options: [], range: range) != nil
    }

    private func containsWord(_ word: String, in text: String) -> Bool {
        text.range(
            of: "\\b\(NSRegularExpression.escapedPattern(for: word))\\b",
            options: [.regularExpression, .caseInsensitive]
        ) != nil
    }

    private func containsNumberedListMarker(_ number: Int, in text: String) -> Bool {
        text.range(
            of: "(^|[\\s.!?])\(number)[.)]\\s+\\S",
            options: [.regularExpression]
        ) != nil
    }

    private func textWithInsertionBoundary(_ transcript: String) -> String {
        let cleaned = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard hasInsertedTextThisDictation, !cleaned.isEmpty else { return transcript }
        guard let first = cleaned.first else { return cleaned }
        let punctuationThatBindsLeft = ".,!?;:%)]}”’"
        if punctuationThatBindsLeft.contains(first) {
            return cleaned
        }
        return " " + cleaned
    }

    private func commitPauseTranscript(
        _ transcript: String,
        snapshotPCMByteCount: Int,
        snapshotSpeechAt: TimeInterval
    ) {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            syncLiveDisplayTranscript()
            return
        }
        let textToPaste = textWithInsertionBoundary(transcript)

        var pasteSucceeded = false
        var pasteKind = "insert"
        var pasteFailureReason: String? = nil
        var pasteAttempted = false
        // Re-resolve the paste target against the user's *current* focus.
        // The user may have clicked into a different field after starting
        // dictation; we honour wherever the caret is now, never where it
        // was when the hotkey fired. See ``refreshPasteTargetFromCurrentFocus``.
        refreshPasteTargetFromCurrentFocus()
        if likelyPasteDestination {
            activateTargetForPasteIfNeeded()
            pasteAttempted = true
            pasteSucceeded = observedUndoSafePaste(textToPaste)
            if pasteSucceeded {
                partialInsertFailed = false
                hasInsertedTextThisDictation = true
                lastPastedFromBroker = textToPaste
                accumulatedText = textToPaste
                copyableTranscript = nil
                triggerDraftFlash()
                let crossed = JunoLifetimeWords.recordWords(from: trimmed)
                JunoMilestoneNotifier.shared.notifyIfMilestone(crossed: crossed, useFullLockup: false)
            } else {
                partialInsertFailed = true
                pendingRevisionFullTranscript = textToPaste
                liveSpeechHint = "Text ready — paste failed, tap again to stop and copy."
                pasteKind = "none"
                pasteFailureReason = "undo_safe_paste_failed"
            }
        } else {
            partialInsertFailed = true
            pendingRevisionFullTranscript = textToPaste
            Clipboard.writeString(textToPaste)
            liveSpeechHint = "Text copied — no active text field."
            pasteKind = "none"
            pasteFailureReason = "no_active_text_field"
        }

        NSLog(
            "Juno: pause commit paste ok=%@ transcript_len=%d paste_kind=%@",
            pasteSucceeded ? "true" : "false",
            trimmed.count,
            pasteKind
        )
        postInsertionCommitted(
            transcript: textToPaste,
            ok: pasteSucceeded,
            pasteKind: pasteKind,
            failureReason: pasteFailureReason,
            pasteAttempted: pasteAttempted
        )
        if pasteSucceeded {
            startCorrectionMonitor(expectedText: textToPaste, pasteKind: pasteKind)
            resetForNextPauseUtterance(
                snapshotPCMByteCount: snapshotPCMByteCount,
                snapshotSpeechAt: snapshotSpeechAt
            )
        } else {
            syncLiveDisplayTranscript()
        }
    }

    private func resetForNextPauseUtterance(
        snapshotPCMByteCount: Int,
        snapshotSpeechAt: TimeInterval
    ) {
        pcmLock.lock()
        let bytesToDrop = min(snapshotPCMByteCount, sessionPCMData.count)
        if bytesToDrop > 0 {
            sessionPCMData.removeSubrange(0..<bytesToDrop)
        }
        framesReceivedThisSession = Int64(sessionPCMData.count / MemoryLayout<Int16>.size)
        pcmLock.unlock()

        let newSpeechAlreadyCaptured = lastSpeechEnergyAt > snapshotSpeechAt
        pendingRevisionFullTranscript = nil
        lastPastedFromBroker = ""
        pendingFinalBrokerPasteKind = nil
        pendingFinalReplaceTarget = nil
        pendingFinalReplaceTargetChars = 0
        accumulatedText = ""
        engineSessionPartialText = ""
        engineSessionFinalCandidateText = ""
        livePartialText = ""
        resetHUDTranscriptStore()
        lastNonEmptyHUDTranscript = ""
        liveSpeechHint = nil
        firstAudioFrameAt = framesReceivedThisSession > 0 ? firstAudioFrameAt : 0
        lastLiveHUDTextChangeAt = 0
        lastLiveHUDTextChangePCMBytes = sessionPCMByteCount()
        lastPreviewSegmentRollSpeechAt = 0
        liveAudioCheckpointInFlight = false
        lastLiveAudioCheckpointRequestedAt = 0
        lastLiveAudioCheckpointPCMBytes = 0
        liveAudioCheckpointBackpressureUntil = 0
        if !newSpeechAlreadyCaptured {
            hasEverDetectedSpeech = false
            speechDetectedLoggedThisSession = false
            lastSpeechEnergyAt = 0
            lastPartialSnapshotSpeechAt = 0
            ambientNoiseRMS = max(ambientNoiseRMS, Self.silenceRMSThreshold)
            if hudState == .listening || hudState == .partialCommit {
                state = "waiting_speech"
            }
        }
        sessionStartTime = Date().timeIntervalSinceReferenceDate
        syncLiveDisplayTranscript(allowFrozen: false)
    }

    private func postInsertionCommitted(
        transcript: String,
        ok: Bool,
        pasteKind: String,
        failureReason: String? = nil,
        pasteAttempted: Bool = true
    ) {
        let uid = pendingUtteranceId.isEmpty ? nil : pendingUtteranceId
        var payload: [String: Any] = [
            "transcript": transcript,
            "transcript_len": transcript.count,
            "target_pid": Int(targetPid),
            "ok": ok,
            "partial_mode": true,
            "broker_on_pause": true,
            "trigger_source": insertionTriggerSource,
            "paste_kind": pasteKind,
            "shell_timeline": utteranceTimelinePayload(),
        ]
        decorateInsertionPayload(&payload, failureReason: ok ? nil : failureReason, pasteAttempted: pasteAttempted)
        if let uid { payload["utterance_id"] = uid }
        // Focus-drift diagnostic: the frontmost PID at the moment juno-paste
        // posted Cmd+V vs the PID we believed we were targeting when the
        // hotkey fired. juno-paste exits 0 even when the keystroke landed
        // in a different app (it's a fire-and-forget CGEvent.post), so
        // this is currently the only signal we have to detect "the user
        // switched apps mid-dictation and the paste went to the wrong
        // place." Backend logs a warning when drift is true; full
        // AX-watch verification remains a separate followup.
        if let pasteFrontmost = Clipboard.lastPasteFrontmostPid {
            payload["paste_frontmost_pid"] = Int(pasteFrontmost)
            payload["paste_frontmost_drifted"] = pasteFrontmost != targetPid
        }
        // Privacy gate (3/5): suppress history append when secure.
        if SecureFieldPolicy.allowHistory(secure: lastSecureFlag) {
            JunoBroker.post(path: "api/broker/insertion/committed", payload: payload)
        } else {
            NSLog("Juno: insertion/committed suppressed by SecureFieldPolicy (history gate)")
        }
    }

    private func decorateInsertionPayload(
        _ payload: inout [String: Any],
        failureReason: String?,
        pasteAttempted: Bool
    ) {
        payload["paste_attempted"] = pasteAttempted
        payload["likely_paste_destination"] = likelyPasteDestination
        if !targetAppBundleId.isEmpty {
            payload["target_bundle_id"] = targetAppBundleId
            payload["target_app_bundle_id"] = targetAppBundleId
            payload["app_bundle_id"] = targetAppBundleId
        }
        let targetName = targetApplicationNameForBroker()
        if !targetName.isEmpty {
            payload["target_app_name"] = targetName
            payload["app_name"] = targetName
        }
        if !targetWindowTitle.isEmpty {
            payload["target_window_title"] = targetWindowTitle
            payload["window_title"] = targetWindowTitle
        }
        if let failureReason, !failureReason.isEmpty {
            payload["paste_failure_reason"] = failureReason
            payload["failure_reason"] = failureReason
        }
    }

    private func finalizeDictationSession(brokerSucceeded: Bool) {
        let pendingRevisionLen = pendingRevisionFullTranscript?.count ?? 0
        NSLog(
            "Juno: finalize dictation broker_ok=%@ transcript_len=%d pending_revision_len=%d partial_failed=%@",
            brokerSucceeded ? "true" : "false",
            lastPastedFromBroker.count,
            pendingRevisionLen,
            partialInsertFailed ? "true" : "false"
        )
        cancelMicWatchdogIfNeeded()
        if cancelInFlightBrokerInsertion {
            // User pressed Esc; broker may still resolve in background but we
            // don't paste, don't toast, don't surface copy-ready.
            cancelInFlightBrokerInsertion = false
            goIdleOnMain()
            return
        }
        // **Resolve paste target NOW**, against whatever the user is
        // focused on at the moment of finalize. This is the single source
        // of truth for ``likelyPasteDestination`` / ``targetPid`` /
        // ``targetAppBundleId`` for the rest of this method — every paste
        // branch below relies on these flags. See
        // ``refreshPasteTargetFromCurrentFocus`` for the policy.
        refreshPasteTargetFromCurrentFocus()

        let pastedText = lastPastedFromBroker
        let pendingFinalText = pendingRevisionFullTranscript?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let finalText = ((pendingFinalText?.isEmpty == false) ? pendingRevisionFullTranscript : nil)
            ?? pastedText
        var committedFinalText = finalText
        var finalPasteSucceeded = false
        var finalPasteKind = pendingFinalBrokerPasteKind
            ?? (suppressPartialPasteForSelectionEditing ? "replace" : "insert")
        var finalPasteAttempted = false
        var finalPasteFailureReason: String? = nil
        let finalPasteAllowed = SecureFieldPolicy.allowPaste(secure: lastSecureFlag)
        var deletedRecentReplaceTarget = false

        func blockedPasteReason() -> String {
            finalPasteAllowed ? "no_active_text_field" : "secure_field"
        }

        func deleteRecentReplaceTargetIfNeeded() {
            // focused_text_before targets count chars from the paragraph
            // start to the caret (incl. trailing whitespace), so the same
            // delete-back-from-caret removal is exact for them too.
            guard finalPasteKind == "replace",
                  !suppressPartialPasteForSelectionEditing,
                  !deletedRecentReplaceTarget,
                  pendingFinalReplaceTarget == "recent_clipboard"
                    || pendingFinalReplaceTarget == "recent_commit"
                    || pendingFinalReplaceTarget == "focused_text_before",
                  pendingFinalReplaceTargetChars > 0 else {
                return
            }
            Clipboard.deleteLastNCharacters(pendingFinalReplaceTargetChars)
            deletedRecentReplaceTarget = true
        }

        if finalText.isEmpty && brokerSnapshotLowSignalThisSession {
            postInsertionCommitted(
                transcript: "",
                ok: false,
                pasteKind: "none",
                failureReason: "low_signal_audio",
                pasteAttempted: false
            )
            insertionTriggerSource = "hotkey"
            liveSpeechHint = "No clear speech detected."
            state = "error:low_signal_audio"
            dictationStartedAt = nil
            refiningStartedAt = nil
            stopWorkbenchStatePolling(clear: true)
            return
        }

        // If the broker never produced any text and we have a live
        // caption draft, surface it via the copy overlay so the user
        // doesn't lose what they said.
        if finalText.isEmpty && brokerSnapshotFailedThisSession {
            let fallback = livePartialText.trimmingCharacters(in: .whitespaces)
            if !fallback.isEmpty {
                copyableTranscript = fallback
            } else {
                pendingRevisionFullTranscript = nil
                postInsertionCommitted(
                    transcript: "",
                    ok: false,
                    pasteKind: "none",
                    failureReason: "transcribe_failed",
                    pasteAttempted: false
                )
                insertionTriggerSource = "hotkey"
                syncLiveDisplayTranscript(allowFrozen: false)
                stopWorkbenchStatePolling(clear: true)
                state = "error:transcribe_failed"
                dictationStartedAt = nil
                refiningStartedAt = nil
                return
            }
        }

        if suppressPartialPasteForSelectionEditing && !finalText.isEmpty {
            if likelyPasteDestination && finalPasteAllowed {
                activateTargetForPasteIfNeeded()
                finalPasteAttempted = true
                finalPasteSucceeded = observedUndoSafePaste(finalText)
                if finalPasteSucceeded {
                    hasInsertedTextThisDictation = true
                    lastPastedFromBroker = finalText
                    accumulatedText = finalText
                    presentFinalTextReveal(for: finalText)
                    triggerDraftFlash()
                    let crossed = JunoLifetimeWords.recordWords(from: finalText)
                    JunoMilestoneNotifier.shared.notifyIfMilestone(crossed: crossed, useFullLockup: false)
                } else {
                    copyableTranscript = finalText
                    finalPasteKind = "none"
                    finalPasteFailureReason = "undo_safe_paste_failed"
                }
            } else {
                Clipboard.writeString(finalText)
                copyableTranscript = finalText
                finalPasteKind = "none"
                finalPasteFailureReason = blockedPasteReason()
            }
        } else if pendingFinalText?.isEmpty == false && !finalText.isEmpty {
            if likelyPasteDestination && finalPasteAllowed {
                activateTargetForPasteIfNeeded()
                let replacingPartial = !pastedText.isEmpty && !partialInsertFailed
                if replacingPartial {
                    Clipboard.deleteLastNCharacters(pastedText.count)
                    finalPasteKind = "replace"
                } else {
                    deleteRecentReplaceTargetIfNeeded()
                }
                finalPasteAttempted = true
                finalPasteSucceeded = observedUndoSafePaste(finalText)
                if finalPasteSucceeded {
                    hasInsertedTextThisDictation = true
                    lastPastedFromBroker = finalText
                    accumulatedText = finalText
                    presentFinalTextReveal(for: finalText)
                    triggerDraftFlash()
                } else {
                    if replacingPartial {
                        _ = Clipboard.undoSafePaste(pastedText)
                    }
                    copyableTranscript = finalText
                    finalPasteKind = "none"
                    finalPasteFailureReason = "undo_safe_paste_failed"
                }
            } else {
                Clipboard.writeString(finalText)
                copyableTranscript = finalText
                finalPasteKind = "none"
                finalPasteFailureReason = blockedPasteReason()
            }
        } else if partialInsertFailed && !finalText.isEmpty {
            if likelyPasteDestination && finalPasteAllowed {
                let textToPaste = textWithInsertionBoundary(finalText)
                committedFinalText = textToPaste
                activateTargetForPasteIfNeeded()
                deleteRecentReplaceTargetIfNeeded()
                finalPasteAttempted = true
                finalPasteSucceeded = observedUndoSafePaste(textToPaste)
                if finalPasteSucceeded {
                    partialInsertFailed = false
                    hasInsertedTextThisDictation = true
                    lastPastedFromBroker = textToPaste
                    accumulatedText = textToPaste
                    presentFinalTextReveal(for: textToPaste)
                    triggerDraftFlash()
                    let crossed = JunoLifetimeWords.recordWords(from: textToPaste)
                    JunoMilestoneNotifier.shared.notifyIfMilestone(crossed: crossed, useFullLockup: false)
                } else {
                    copyableTranscript = textToPaste
                    finalPasteKind = "none"
                    finalPasteFailureReason = "undo_safe_paste_failed"
                }
            } else {
                let textToCopy = textWithInsertionBoundary(finalText)
                committedFinalText = textToCopy
                Clipboard.writeString(textToCopy)
                copyableTranscript = textToCopy
                finalPasteKind = "none"
                finalPasteFailureReason = blockedPasteReason()
            }
        } else if !partialInsertFailed && !finalText.isEmpty {
            if likelyPasteDestination && finalPasteAllowed {
                let textToPaste = textWithInsertionBoundary(finalText)
                committedFinalText = textToPaste
                activateTargetForPasteIfNeeded()
                deleteRecentReplaceTargetIfNeeded()
                finalPasteAttempted = true
                finalPasteSucceeded = observedUndoSafePaste(textToPaste)
                if finalPasteSucceeded {
                    hasInsertedTextThisDictation = true
                    lastPastedFromBroker = textToPaste
                    accumulatedText = textToPaste
                    presentFinalTextReveal(for: textToPaste)
                    triggerDraftFlash()
                    let crossed = JunoLifetimeWords.recordWords(from: textToPaste)
                    JunoMilestoneNotifier.shared.notifyIfMilestone(crossed: crossed, useFullLockup: false)
                } else {
                    copyableTranscript = textToPaste
                    finalPasteKind = "none"
                    finalPasteFailureReason = "undo_safe_paste_failed"
                }
            } else {
                let textToCopy = textWithInsertionBoundary(finalText)
                committedFinalText = textToCopy
                Clipboard.writeString(textToCopy)
                copyableTranscript = textToCopy
                finalPasteKind = "none"
                finalPasteFailureReason = blockedPasteReason()
            }
        }
        pendingRevisionFullTranscript = nil

        let insertionOk = !finalText.isEmpty && finalPasteSucceeded
        // An empty final transcript means no paste was ever attempted (the
        // writer consumed the utterance as a command, or the engine gated
        // it). Without an explicit reason the broker defaults the row to
        // "paste_failed" and History claims Juno "could not insert text" —
        // wrong: there was nothing to insert. Report a distinct code.
        var reportedFailureReason = finalPasteFailureReason
        if !insertionOk, finalText.isEmpty, reportedFailureReason == nil {
            reportedFailureReason = "empty_final_text"
        }
        postInsertionCommitted(
            transcript: committedFinalText,
            ok: insertionOk,
            pasteKind: finalPasteKind,
            failureReason: insertionOk ? nil : reportedFailureReason,
            pasteAttempted: finalPasteAttempted
        )

        // Counter for the post-onboarding Voice Actions nudge — see
        // ``JunoUserDefaults.actionsNudgeShownKey`` and
        // ``JunoActionsHomeCard``. Only successful pastes count so a
        // string of failed sessions never trips the nudge.
        if insertionOk {
            JunoUserDefaults.incrementDictationCompletedCount()
        }

        if finalPasteSucceeded && !committedFinalText.isEmpty {
            startCorrectionMonitor(expectedText: committedFinalText, pasteKind: finalPasteKind)
        }

        insertionTriggerSource = "hotkey"
        syncLiveDisplayTranscript(allowFrozen: false)
        state = "idle"
        dictationStartedAt = nil
        refiningStartedAt = nil
    }

    private func teardownRecognition() {
        // Apple Speech recognition is not part of production dictation.
        // This hook is kept because error/cancel paths already call it.
    }

    // MARK: - Thread-safe last-sound time

    private func updateLastSoundTime(_ t: TimeInterval) {
        soundTimeLock.lock(); defer { soundTimeLock.unlock() }
        _lastSoundTime = t
    }

    private func readLastSoundTime() -> TimeInterval {
        soundTimeLock.lock(); defer { soundTimeLock.unlock() }
        return _lastSoundTime
    }

    // MARK: - Insert + text monitor

    private func stopTextMon() {
        textmonStdout?.readabilityHandler = nil
        textmonStdout = nil
        textMonExpectsReplacePaste = false
        pendingTextMonExpectedPaste = ""
        textMonInitialSnapshot = nil
        if let t = textmonTask {
            t.terminationHandler = nil
            if t.isRunning {
                t.terminate()
            }
            textmonTask = nil
        }
    }

    private func clearTextMonIfCurrent(_ task: Process) {
        guard textmonTask === task else { return }
        textmonStdout?.readabilityHandler = nil
        textmonStdout = nil
        textmonTask?.terminationHandler = nil
        textmonTask = nil
        textMonExpectsReplacePaste = false
        pendingTextMonExpectedPaste = ""
        textMonInitialSnapshot = nil
    }

    private func installTextMonHandlers(task: Process, stdout outHandle: FileHandle, expectedText: String) {
        outHandle.readabilityHandler = { [weak self, weak task] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else {
                if let task {
                    DispatchQueue.main.async { [weak self] in
                        self?.clearTextMonIfCurrent(task)
                    }
                }
                return
            }
            let lines = chunk.split(separator: "\n").map(String.init)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                for line in lines {
                    self.handleTextMonLine(line, expected: expectedText)
                }
            }
        }
        task.terminationHandler = { [weak self, weak task] _ in
            guard let task else { return }
            DispatchQueue.main.async { [weak self] in
                self?.clearTextMonIfCurrent(task)
            }
        }
    }

    private func insertAndMonitor(
        transcript: String,
        utteranceId: String?,
        pasteKind: String = "insert",
        noopReason: String? = nil
    ) {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if pasteKind == "none" && trimmed.isEmpty {
            if let nr = noopReason, !nr.isEmpty {
                liveSpeechHint = "Nothing inserted (\(nr))."
            }
            var payload: [String: Any] = [
                "transcript": "",
                "transcript_len": 0,
                "target_pid": Int(targetPid),
                "ok": false,
                "trigger_source": insertionTriggerSource,
                "paste_kind": pasteKind,
            ]
            decorateInsertionPayload(&payload, failureReason: noopReason, pasteAttempted: false)
            if let utteranceId { payload["utterance_id"] = utteranceId }
            if let noopReason, !noopReason.isEmpty { payload["noop_reason"] = noopReason }
            // Privacy gate (3/5): suppress history append when secure.
            if SecureFieldPolicy.allowHistory(secure: lastSecureFlag) {
                JunoBroker.post(path: "api/broker/insertion/committed", payload: payload)
            } else {
                NSLog("Juno: insertion/committed suppressed by SecureFieldPolicy (history gate)")
            }
            insertionTriggerSource = "hotkey"
            return
        }

        if pasteKind == "none" && !trimmed.isEmpty {
            copyableTranscript = transcript
            liveSpeechHint = "Ambiguous voice command with text selected — copied result instead of pasting."
            var payload: [String: Any] = [
                "transcript": trimmed,
                "transcript_len": trimmed.count,
                "target_pid": Int(targetPid),
                "ok": false,
                "trigger_source": insertionTriggerSource,
                "paste_kind": pasteKind,
            ]
            decorateInsertionPayload(&payload, failureReason: noopReason ?? "paste_kind_none_with_text", pasteAttempted: false)
            if let utteranceId { payload["utterance_id"] = utteranceId }
            if let noopReason, !noopReason.isEmpty { payload["noop_reason"] = noopReason }
            // Privacy gate (3/5): suppress history append when secure.
            if SecureFieldPolicy.allowHistory(secure: lastSecureFlag) {
                JunoBroker.post(path: "api/broker/insertion/committed", payload: payload)
            } else {
                NSLog("Juno: insertion/committed suppressed by SecureFieldPolicy (history gate)")
            }
            insertionTriggerSource = "hotkey"
            return
        }

        guard !transcript.isEmpty else { return }
        let text = transcript
        guard !text.isEmpty else { return }
        let trigger = insertionTriggerSource
        stopTextMon()
        textMonExpectsReplacePaste = (pasteKind == "replace")
        pendingTextMonExpectedPaste = text
        textMonInitialSnapshot = nil
        let ok: Bool
        let pasteAttempted: Bool
        var pasteFailureReason: String? = nil
        // Re-resolve the paste target against current focus before the
        // synthetic Cmd+V fires. See ``refreshPasteTargetFromCurrentFocus``.
        refreshPasteTargetFromCurrentFocus()
        // Privacy gate (5/5): defense-in-depth paste suppression keyed on
        // the sticky ``lastSecureFlag``. See ``JunoSecureFieldPolicy``.
        if !SecureFieldPolicy.allowPaste(secure: lastSecureFlag) {
            pasteAttempted = false
            ok = false
            pasteFailureReason = "secure_field"
            NSLog("Juno: paste suppressed by SecureFieldPolicy (secure-field privacy gate)")
        } else if likelyPasteDestination {
            activateTargetForPasteIfNeeded()
            pasteAttempted = true
            ok = observedUndoSafePaste(text)
            if !ok {
                pasteFailureReason = "undo_safe_paste_failed"
            }
        } else {
            pasteAttempted = false
            Clipboard.writeString(text)
            ok = false
            copyableTranscript = text
            pasteFailureReason = "no_active_text_field"
        }
        if ok {
            presentFinalTextReveal(for: text)
            triggerDraftFlash()
            let crossed = JunoLifetimeWords.recordWords(from: text)
            JunoMilestoneNotifier.shared.notifyIfMilestone(crossed: crossed, useFullLockup: false)
        }

        var payload: [String: Any] = [
            "transcript": text,
            "transcript_len": text.count,
            "target_pid": Int(targetPid),
            "ok": ok,
            "trigger_source": trigger,
            "paste_kind": pasteKind,
        ]
        decorateInsertionPayload(&payload, failureReason: ok ? nil : pasteFailureReason, pasteAttempted: pasteAttempted)
        if let utteranceId { payload["utterance_id"] = utteranceId }
        if let noopReason, !noopReason.isEmpty { payload["noop_reason"] = noopReason }
        // Privacy gate (3/5): broker ``insertion/committed`` triggers
        // history append. Suppress when secure so the broker never sees
        // the transcript. Defense-in-depth even if engine also gates.
        if SecureFieldPolicy.allowHistory(secure: lastSecureFlag) {
            JunoBroker.post(path: "api/broker/insertion/committed", payload: payload)
        } else {
            NSLog("Juno: insertion/committed suppressed by SecureFieldPolicy (history gate)")
        }
        insertionTriggerSource = "hotkey"

        if likelyPasteDestination, !ok {
            copyableTranscript = text
        }

        // Privacy gate (2/5): skip the ``juno-textmon`` learn-from-
        // corrections observer when secure. We must not watch the
        // focused field's post-paste value or feed corrections back
        // into the vocabulary.
        guard SecureFieldPolicy.allowLearning(secure: lastSecureFlag) else {
            NSLog("Juno: textmon (learn) suppressed by SecureFieldPolicy")
            return
        }
        guard ok, targetPid > 0, let bin = HelperBinary.path("juno-textmon") else { return }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: bin)
        task.arguments = [String(targetPid)]

        let stdin = Pipe()
        let stdout = Pipe()
        task.standardInput = stdin
        task.standardOutput = stdout
        task.standardError = Pipe()

        let outHandle = stdout.fileHandleForReading
        installTextMonHandlers(task: task, stdout: outHandle, expectedText: text)

        do {
            try task.run()
            stdin.fileHandleForWriting.write(Data(text.utf8))
            try? stdin.fileHandleForWriting.close()
            textmonStdout = outHandle
            textmonTask = task
        } catch {
            NSLog("Juno: textmon launch failed: \(error.localizedDescription)")
        }
    }

    private func startCorrectionMonitor(expectedText: String, pasteKind: String) {
        // Privacy gate (2/5): never start the AX correction observer
        // for an utterance whose origin field was secure. The pinned
        // ``lastSecureFlag`` defends against TOCTOU focus shifts.
        guard SecureFieldPolicy.allowLearning(secure: lastSecureFlag) else {
            NSLog("Juno: startCorrectionMonitor suppressed by SecureFieldPolicy")
            return
        }
        guard targetPid > 0, let bin = HelperBinary.path("juno-textmon") else { return }
        let expected = expectedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !expected.isEmpty else { return }

        stopTextMon()
        textMonExpectsReplacePaste = (pasteKind == "replace")
        pendingTextMonExpectedPaste = expectedText
        textMonInitialSnapshot = nil

        let task = Process()
        task.executableURL = URL(fileURLWithPath: bin)
        task.arguments = [String(targetPid)]

        let stdin = Pipe()
        let stdout = Pipe()
        task.standardInput = stdin
        task.standardOutput = stdout
        task.standardError = Pipe()

        let outHandle = stdout.fileHandleForReading
        installTextMonHandlers(task: task, stdout: outHandle, expectedText: expectedText)

        do {
            try task.run()
            stdin.fileHandleForWriting.write(Data(expectedText.utf8))
            try? stdin.fileHandleForWriting.close()
            textmonStdout = outHandle
            textmonTask = task
        } catch {
            NSLog("Juno: textmon launch failed: \(error.localizedDescription)")
        }
    }

    private func flashTransientDone(for text: String) {
        let words = text.split { $0.isWhitespace || $0.isNewline }.filter { !$0.isEmpty }.count
        guard words > 0 else { return }
        clearDoneWorkItem?.cancel()
        transientDoneWordCount = words
        let work = DispatchWorkItem { [weak self] in
            self?.transientDoneWordCount = nil
        }
        clearDoneWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.35, execute: work)
    }

    /// Post-paste HUD reveal for the final transcript.
    ///
    /// This is only reached on a **successful** paste — the text now lives in
    /// the user's focused field, so the HUD's job is done. We never keep the
    /// expanded copy-ready island up here (regardless of the live-preview
    /// setting); a brief "Text placed +N" flash acknowledges the insert and the
    /// HUD then dismisses. Leaving the full transcript pinned after a good paste
    /// reads as a "preview that won't go away." The expanded copy-ready island
    /// is reserved for the paste-failed / no-field branches, which set
    /// ``copyableTranscript`` directly so the user can still copy.
    private func presentFinalTextReveal(for text: String) {
        copyableTranscript = nil
        flashTransientDone(for: text)
    }

    private func showActionHUDResult(_ results: [JunoActionResult]) {
        guard !results.isEmpty else { return }
        let summary = JunoActionBatchFormatter.summarize(results)

        let symbol: String = {
            switch summary.tone {
            case .allSaved: return "checkmark.circle.fill"
            case .partial: return "checkmark.circle.badge.questionmark"
            case .blocked: return "lock.fill"
            case .failed: return "exclamationmark.triangle.fill"
            case .allPending: return "arrow.triangle.2.circlepath"
            }
        }()
        let isFailure = (summary.tone == .blocked || summary.tone == .failed)

        // For single-action all-saved, prefer the kind-specific destination
        // subtitle (e.g. "Saved to Notes \u{2192} Juno folder.") so
        // the chip teaches users where each action lands. For multi-
        // action all-saved, the brand-island has no room for the verbose
        // formatter detail — collapse to the destination tail ("Apple
        // Notes · Reminders") and let the toast carry the per-row
        // breakdown. For partial / blocked, the formatter's detail line
        // is strictly more informative than a per-kind subtitle.
        let subtitle: String = {
            if summary.tone == .allSaved {
                if results.count == 1 {
                    return actionHUDSubtitle(for: results.first)
                }
                return multiActionDestinationSubtitle(for: results)
            }
            if let detail = summary.detail { return detail }
            switch summary.tone {
            case .blocked: return "Open Actions to finish setup."
            case .failed: return "Open History for details."
            case .partial: return "Open Actions to finish the rest."
            case .allPending: return "Syncing\u{2026}"
            case .allSaved: return actionHUDSubtitle(for: results.first)
            }
        }()

        let displaySeconds = actionHUDDisplaySeconds(summary: summary, resultCount: results.count)
        let nativeKind = (summary.tone == .allSaved && results.count == 1) ? results.first?.kind : nil
        showTransientActionHUD(
            title: summary.oneLine,
            subtitle: subtitle,
            symbolName: symbol,
            kind: nativeKind,
            isFailure: isFailure,
            displaySeconds: displaySeconds
        )
    }

    private func actionHUDDisplaySeconds(
        summary: JunoActionBatchSummary,
        resultCount: Int
    ) -> TimeInterval {
        let count = max(1, resultCount)
        if summary.tone == .allSaved, count == 1 {
            return 3.0
        }
        if summary.tone == .allSaved {
            return min(5.0, 3.6 + Double(count) * 0.35)
        }
        return min(5.0, 4.0 + Double(count) * 0.35)
    }

    /// Destination-list subtitle for the brand-island multi-action result
    /// pill. Dedupes kinds in the order they first appear so the user can
    /// recognize the ordering of their own request.
    private func multiActionDestinationSubtitle(for results: [JunoActionResult]) -> String {
        var seen: [JunoActionKind] = []
        for r in results where !seen.contains(r.kind) { seen.append(r.kind) }
        let names = seen.map { kind -> String in
            switch kind {
            case .note:     return "Notes"
            case .reminder: return "Reminders"
            case .alarm:    return "Alarm"
            }
        }
        return names.joined(separator: " · ")
    }

    private func actionHUDSubtitle(for result: JunoActionResult?) -> String {
        guard let result else { return "Saved by Juno." }
        switch result.kind {
        case .reminder:
            return "Added to Reminders."
        case .note:
            // Tell the user where to look. Without this they dictate a
            // note, see "Saved", and don't realize it lives in a folder
            // called "Juno" (often under iCloud) — leading to the
            // recurring "the note never gets saved" complaint.
            return "Saved to Notes \u{2192} \(JunoNotesFolderName) folder."
        case .alarm:
            // Alarms are deliberately Calendar events, not a custom
            // local notification — that way the alert fires even when
            // Juno is closed or crashed. The copy makes that explicit so
            // users don't go looking in a Clock app that doesn't exist
            // on macOS.
            return "Alarm saved as a Calendar alert."
        }
    }

    private func showTransientActionHUD(
        title: String,
        subtitle: String,
        symbolName: String,
        kind: JunoActionKind? = nil,
        isFailure: Bool,
        displaySeconds: TimeInterval = 2.4
    ) {
        clearActionHUDWorkItem?.cancel()
        transientActionHUDResult = JunoActionHUDResult(
            title: title,
            subtitle: subtitle,
            symbolName: symbolName,
            kind: kind,
            isFailure: isFailure
        )
        let work = DispatchWorkItem { [weak self] in
            self?.transientActionHUDResult = nil
        }
        clearActionHUDWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + displaySeconds, execute: work)
    }

    private func triggerDraftFlash() {
        draftFlashActive = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.32) { [weak self] in
            self?.draftFlashActive = false
        }
    }

    private func handleTextMonLine(_ line: String, expected: String) {
        if line.hasPrefix("INITIAL:") {
            let initial = String(line.dropFirst("INITIAL:".count))
            recordTextMonInitialSnapshot(initial, pasted: expected)
            return
        }
        if line.hasPrefix("INITIAL_B64:") {
            let b64 = String(line.dropFirst("INITIAL_B64:".count))
            if let data = Data(base64Encoded: b64), let s = String(data: data, encoding: .utf8) {
                recordTextMonInitialSnapshot(s, pasted: expected)
            }
            return
        }
        if line == "NO_ELEMENT" || line == "NO_VALUE" {
            textMonInitialSnapshot = nil
            return
        }
        guard line.hasPrefix("CHANGED:") || line.hasPrefix("CHANGED_B64:") else { return }
        var observed: String = ""
        if line.hasPrefix("CHANGED:") {
            observed = String(line.dropFirst("CHANGED:".count))
        } else if line.hasPrefix("CHANGED_B64:") {
            let b64 = String(line.dropFirst("CHANGED_B64:".count))
            if let data = Data(base64Encoded: b64), let s = String(data: data, encoding: .utf8) {
                observed = s
            }
        }
        guard !observed.isEmpty, observed != expected else { return }
        guard shouldLearnTextMonCorrection(observed: observed, expected: expected) else { return }
        JunoBroker.post(
            path: "api/broker/learning/observe_correction",
            payload: [
                "source": "mac_post_paste_monitor",
                "utterance_id": pendingUtteranceId,
                "observed": expected,
                "corrected": observed,
                "app_bundle_id": targetAppBundleId,
            ]
        )
    }

    private func recordTextMonInitialSnapshot(_ fieldSnapshot: String, pasted: String) {
        textMonInitialSnapshot = fieldSnapshot
        verifyPasteLandedIfNeeded(fieldSnapshot: fieldSnapshot, pasted: pasted)
    }

    private func verifyPasteLandedIfNeeded(fieldSnapshot: String, pasted: String) {
        let p = pasted.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !p.isEmpty else { return }
        if fieldSnapshot.contains(p) { return }
        // Terminals and rich editors re-wrap pasted text inside their AX
        // value (hard newlines at column width, NBSP), so verbatim
        // containment false-flags any paste longer than one visual line.
        // That reopened the HUD as copy-ready after a successful paste —
        // which also suppressed the dictation hotkey until the user pressed
        // Esc (production 2026-06-11). Collapse all whitespace on both sides
        // before declaring the paste missing.
        if Self.whitespaceCollapsed(fieldSnapshot).contains(Self.whitespaceCollapsed(p)) { return }
        copyableTranscript = pasted
        transientDoneWordCount = nil
        if textMonExpectsReplacePaste {
            liveSpeechHint = "Replace may not have landed — check the selection or use Copy (⌘Z to undo)."
        } else {
            liveSpeechHint = "Text may not have landed — use Copy if needed."
        }
    }

    static func whitespaceCollapsed(_ value: String) -> String {
        value
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func shouldLearnTextMonCorrection(observed: String, expected: String) -> Bool {
        if textMonExpectsReplacePaste { return true }
        let expectedTrimmed = expected.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !expectedTrimmed.isEmpty else { return false }
        let initial = textMonInitialSnapshot?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        // For plain insert into an empty field, the whole watched value is
        // the dictated text, so a later whole-field value is a real user
        // correction. If the field had surrounding text, juno-textmon only
        // gives us the entire field value; learning that as a correction
        // would poison memory with unrelated context.
        return initial == expectedTrimmed
    }

}

// MARK: - Hotkey bridge (subprocess)

final class HotkeyBridge {
    private var task: Process?
    private let onDown: () -> Void
    private let onUp: () -> Void
    private let onEscape: () -> Void
    private let onCopy: () -> Void
    private let shouldHandleCopy: () -> Bool
    private let shouldSuppressDictationShortcut: () -> Bool

    init(
        onDown: @escaping () -> Void,
        onUp: @escaping () -> Void,
        onEscape: @escaping () -> Void = {},
        onCopy: @escaping () -> Void = {},
        shouldHandleCopy: @escaping () -> Bool = { false },
        shouldSuppressDictationShortcut: @escaping () -> Bool = { false }
    ) {
        self.onDown = onDown
        self.onUp = onUp
        self.onEscape = onEscape
        self.onCopy = onCopy
        self.shouldHandleCopy = shouldHandleCopy
        self.shouldSuppressDictationShortcut = shouldSuppressDictationShortcut
    }

    func start() {
        // Idempotent: `JunoShellApp.init` calls `start()` directly when
        // onboarding is already completed, and `applicationDidFinishLaunching`
        // *also* invokes `startHotkeyBridge()`. Without this guard, both
        // call sites fire on relaunch and we end up with two `juno-hotkey`
        // subprocesses both writing the same hotkey events into our stdout
        // handler. That fires `onDown` twice in rapid succession, which
        // toggles dictation start → stop within milliseconds — the HUD
        // never visibly appears even though the bridge is "working".
        if let existing = task, existing.isRunning { return }
        guard let bin = HelperBinary.path("juno-hotkey") else {
            NSLog("Juno: juno-hotkey not found; push-to-talk disabled")
            return
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: bin)

        let stdout = Pipe()
        task.standardOutput = stdout
        task.standardError = Pipe()

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            let shortcut = JunoShortcutPreference.stored
            for raw in text.split(separator: "\n") {
                let line = String(raw)
                // Escape is emitted by ``juno-hotkey`` as a bare "ESC"
                // line whenever the global key listener observes Esc.
                // Used to dismiss the HUD / cancel dictation while another
                // app has focus (where a local NSEvent monitor wouldn't
                // see the keypress). Requires Accessibility / Input
                // Monitoring trust on the helper binary.
                if line.hasPrefix("HOTKEY_DEGRADED:") {
                    // The helper's global key/Esc/flags monitor failed to
                    // install — almost always because Accessibility / Input
                    // Monitoring isn't granted. Previously this only went to
                    // the helper's stderr (which nobody read), so the dictation
                    // key silently received nothing and users reported "I press
                    // it and nothing happens". Log it and nudge the permission
                    // prompt so the cause is visible and actionable.
                    let which = String(line.dropFirst("HOTKEY_DEGRADED:".count))
                    NSLog("Juno: hotkey helper DEGRADED (%@) — Accessibility/Input Monitoring likely not granted", which)
                    DispatchQueue.main.async {
                        JunoPermissionMonitor.shared.noteHotkeyMonitorDegraded(which)
                    }
                    continue
                }
                if line == JunoHotkeyEventLine.escape {
                    NSLog("Juno: hotkey ESC")
                    DispatchQueue.main.async { self.onEscape() }
                    continue
                }
                if JunoHotkeyEventLine.isCopyLine(line) {
                    NSLog("Juno: hotkey COPY")
                    DispatchQueue.main.async {
                        guard self.shouldHandleCopy() else { return }
                        self.onCopy()
                    }
                    continue
                }
                if self.isDownEvent(line, shortcut: shortcut) {
                    NSLog("Juno: hotkey down matched %@", shortcut.rawValue)
                    DispatchQueue.main.async {
                        if self.shouldSuppressDictationShortcut() {
                            // The copy-ready panel is showing (HUD idle with a
                            // leftover transcript). This press used to be a
                            // silent no-op, and since it didn't clear the
                            // copy-ready state, repeated presses were ALSO
                            // dropped — the user had to wait for the panel to
                            // time out, i.e. "press the key twice/thrice
                            // before the overlay appears". A press of the
                            // dictation key is an unambiguous "start a new
                            // dictation": begin it. beginPushToTalk() clears
                            // copyableTranscript and dismisses the panel. ⌘C
                            // (a separate COPY event) still copies the text
                            // first, and the transcript also remains in
                            // History, so starting over loses nothing.
                            NSLog("Juno: hotkey down while copy-ready -> dismiss + begin new dictation")
                        }
                        self.onDown()
                    }
                } else if self.isUpEvent(line, shortcut: shortcut) {
                    NSLog("Juno: hotkey up matched %@", shortcut.rawValue)
                    DispatchQueue.main.async { self.onUp() }
                }
            }
        }

        task.terminationHandler = { [weak self] _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self?.start() }
        }

        do {
            try task.run()
            self.task = task
            NSLog("Juno: hotkey bridge started shortcut=%@", JunoShortcutPreference.stored.rawValue)
            // Register a stop closure with the runtime singleton so
            // applicationWillTerminate can reach back into this bridge
            // (a Swift `App`-owned `private let` is otherwise opaque to
            // the AppDelegate). Captures self weakly so the bridge can
            // still deallocate normally if it's restarted.
            JunoShellRuntime.shared.terminateHotkeyBridge = { [weak self] in
                self?.stop()
            }
        } catch {
            NSLog("Juno: hotkey bridge launch failed: \(error.localizedDescription)")
        }
    }

    func stop() {
        task?.terminationHandler = nil
        task?.terminate()
        task = nil
    }

    private func isDownEvent(_ line: String, shortcut: JunoShortcutPreference) -> Bool {
        switch shortcut {
        case .fn: return line == "FN_DOWN"
        case .rightCommand: return line == "RIGHT_MOD_DOWN:RightCommand"
        case .rightOption: return line == "RIGHT_MOD_DOWN:RightOption"
        case .optionSpace: return line == "OPT_SPACE_DOWN"
        case .controlSpace: return line == "CTRL_SPACE_DOWN"
        }
    }

    private func isUpEvent(_ line: String, shortcut: JunoShortcutPreference) -> Bool {
        switch shortcut {
        case .fn: return line == "FN_UP"
        case .rightCommand: return line == "RIGHT_MOD_UP:RightCommand"
        case .rightOption: return line == "RIGHT_MOD_UP:RightOption"
        case .optionSpace: return line == "OPT_SPACE_UP"
        case .controlSpace: return line == "CTRL_SPACE_UP"
        }
    }
}

// MARK: - App commands (work when Juno is the active app)

extension Notification.Name {
    /// Posted to open the main Juno window (Home).
    static let junoOpenMainWindow = Notification.Name("junoOpenMainWindow")
    /// Posted to open Settings in the main window.
    static let junoOpenSettingsWindow = Notification.Name("junoOpenSettingsWindow")
    /// Posted to open History in the main window.
    static let junoOpenHistoryWindow = Notification.Name("junoOpenHistoryWindow")
    /// Posted when the bundled engine cannot be spawned (missing script, bad repo, exec failure).
    static let junoBrokerBootstrapFailed = Notification.Name("junoBrokerBootstrapFailed")
    /// Posted when the engine supervisor's online/offline state changes.
    /// Object is a ``JunoEngineSupervisor.State`` value.
    static let junoEngineSupervisorStateChanged = Notification.Name("junoEngineSupervisorStateChanged")
    /// Posted by ``JunoSetupModel`` after every successful broker setup
    /// snapshot. Object is the ``BrokerSetupStatusResponse``. The
    /// lifecycle observes this so it can re-evaluate ``phase`` once
    /// install state advances past its initial probe (e.g.
    /// ``.needsModels`` → ``.modelsLoaded`` once ``overallReady=true``).
    /// Without this, the lifecycle freezes at the first terminal phase
    /// and the Home gate stays stuck even after install completes.
    static let junoSetupSnapshotUpdated = Notification.Name("junoSetupSnapshotUpdated")
    /// Posted by ``JunoPermissionMonitor`` when ``canDictate`` flips.
    /// Object is a ``Bool``. Lifecycle observes so it can promote out of
    /// ``.needsPermissions`` once Mic/Accessibility are granted in
    /// System Settings.
    static let junoPermissionsCanDictateChanged = Notification.Name("junoPermissionsCanDictateChanged")
}

// MARK: - App

private enum JunoShellAppBootstrap {
    static var didStart = false
}

@main
struct JunoShellApp: App {
    @NSApplicationDelegateAdaptor(JunoShellAppDelegate.self) private var appDelegate
    @StateObject private var controller: DictationController
    @StateObject private var surface: SurfaceEditingModel
    @ObservedObject private var updater = JunoUpdater.shared
    @State private var brokerHealthy = false
    @State private var menuRecent: [UtteranceHistoryEntry] = []
    @State private var menuRecentLoaded = false
    @State private var menuRecentFailed = false
    @State private var overlay: JunoOverlayCoordinator
    private let hotkey: HotkeyBridge

    init() {
        // Hard OS floor must run before app-owned helpers, polling, overlays,
        // or update checks start. `applicationDidFinishLaunching` keeps a
        // no-op safety check, but this is the real preflight gate.
        JunoSystemRequirements.enforceMinimumOSOrTerminate()
        // **Run BEFORE legacy-defaults migration.** If the install was
        // re-run (TCC wiped) but the prefs plist still says
        // ``JunoOnboardingCompleted=true``, we reset the onboarding flag
        // so the welcome flow runs again — otherwise the user lands on
        // Home with "Permissions needed" half-states. Updates with
        // intact TCC are a no-op. See ``JunoFreshInstallGuard``.
        JunoFreshInstallGuard.runOnce()
        JunoLegacyDefaultsMigration.runOnce()
        JunoUserDefaults.migrateWhisperPreviewDefaults()
        JunoLocalAppLogTee.installIfEnabled()
        JunoSingleInstance.exitIfAlreadyRunning()
        JunoTargetApplicationTracker.shared.start()
        let ctrl = DictationController()
        let surf = SurfaceEditingModel()
        let ovr = JunoOverlayCoordinator()
        _controller = StateObject(wrappedValue: ctrl)
        _surface = StateObject(wrappedValue: surf)
        _overlay = State(initialValue: ovr)
        JunoShellRuntime.shared.controller = ctrl
        JunoShellRuntime.shared.surface = surf
        // Tap-to-toggle: first tap starts, second tap stops.
        // Key-up is intentionally ignored.
        self.hotkey = HotkeyBridge(
            onDown: {
                ctrl.insertionTriggerSource = "hotkey"
                ctrl.toggleDictation()
            },
            onUp: { },
            // Esc dismisses the HUD / cancels dictation. ``juno-hotkey`` emits
            // a global ESC so this works while ANOTHER app is focused (where
            // a local NSEvent monitor wouldn't fire). The local
            // ``OverlayWindow`` Esc monitor was deliberately removed in
            // favour of this single source.
            onEscape: {
                ctrl.cancelDictation()
            },
            onCopy: {
                ctrl.copyCopyableTranscriptToClipboard()
            },
            shouldHandleCopy: {
                JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
                    hotkeyLine: JunoHotkeyEventLine.copy,
                    copyableTranscript: ctrl.copyableTranscript,
                    hudStateWire: ctrl.state
                )
            },
            shouldSuppressDictationShortcut: {
                JunoCopyReadyShortcutPolicy.shouldSuppressDictationShortcut(
                    copyableTranscript: ctrl.copyableTranscript,
                    hudStateWire: ctrl.state
                )
            }
        )
        let hotkeyBridge = self.hotkey
        JunoShellRuntime.shared.startHotkeyBridge = {
            hotkeyBridge.start()
        }
        if JunoUserDefaults.onboardingCompleted {
            hotkeyBridge.start()
        }

        if !JunoShellAppBootstrap.didStart {
            JunoShellAppBootstrap.didStart = true
            // Log where we launched from and, if running translocated (opened
            // from the DMG / a quarantined folder), warn the user to move Juno
            // to Applications — otherwise macOS keeps resetting TCC grants and
            // the hotkey/mic silently stop working.
            JunoAppLocation.logLaunchLocation()
            Task { @MainActor in JunoAppLocation.offerInstallToApplicationsIfNeeded() }
            surf.startPolling()
            ovr.install(controller: ctrl)
            JunoPermissionMonitor.shared.startMonitoring()
            JunoUpdater.shared.startIfConfigured()
            // Pull the HUD cue MP3 off disk during launch so the first
            // open/close transition doesn't pay file-load latency on the
            // main thread (previously caused the close cue to lag the
            // visual fade by 30–100 ms).
            JunoHUDSound.prewarm()
        }
        Task { @MainActor in
            JunoDockVisibility.applyCurrent()
        }
    }

    var body: some Scene {
        MenuBarExtra {
            JunoMenuCommandCenter(
                controller: controller,
                surface: surface,
                updater: updater,
                brokerHealthy: brokerHealthy,
                statusLabel: brokerStatusLabel,
                statusState: menuStatusBadgeState,
                onboardingCompleted: JunoUserDefaults.onboardingCompleted,
                recent: menuRecent,
                recentLoaded: menuRecentLoaded,
                recentFailed: menuRecentFailed
            )
            .onAppear {
                JunoBrandMarkRasterizer.installDockIconIfNeeded()
                refreshMenuCommandCenter()
            }
            .onReceive(Timer.publish(every: 8, on: .main, in: .common).autoconnect()) { _ in
                refreshMenuCommandCenter()
            }
        } label: {
            menuBarLabel
        }
        .menuBarExtraStyle(.window)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Juno") {
                    NSApp.orderFrontStandardAboutPanel(options: [
                        .applicationVersion: JunoProductIdentity.versionSummary,
                        NSApplication.AboutPanelOptionKey(rawValue: "Version"): "",
                    ])
                }
            }
            CommandGroup(after: .appInfo) {
                Button("Open Juno") {
                    NotificationCenter.default.post(name: .junoOpenMainWindow, object: nil)
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
                Button("History") {
                    NotificationCenter.default.post(name: .junoOpenHistoryWindow, object: nil)
                }
                .keyboardShortcut("h", modifiers: .command)
                Button("Settings…") {
                    NotificationCenter.default.post(name: .junoOpenSettingsWindow, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
                Button("Check for Updates…") {
                    JunoUpdater.shared.checkForUpdates()
                }
                .disabled(updater.isConfigured && !updater.canCheckForUpdates)
            }
        }
    }

    private func refreshMenuCommandCenter() {
        JunoBroker.pingHealth { ok in
            brokerHealthy = ok
        }
        menuRecentLoaded = false
        JunoBroker.fetchHistory(limit: 3) { result in
            menuRecentLoaded = true
            switch result {
            case .success(let resp):
                menuRecent = resp.entries ?? []
                menuRecentFailed = false
            case .failure:
                menuRecent = []
                menuRecentFailed = true
            }
        }
    }

    private var brokerStatusLabel: String {
        if !brokerHealthy { return "Voice engine offline" }
        switch controller.hudState {
        case .idle: return "Ready"
        case .checkingCapability: return "Checking…"
        case .checkingMic: return "Checking microphone…"
        case .waitingSpeech: return "Waiting for speech…"
        case .listening: return "Listening"
        case .partialCommit: return "Polishing…"
        case .refining: return "Transcribing…"
        case .blocked: return "Blocked"
        case .error: return "Error"
        case .unknown: return "Ready"
        }
    }

    private var menuStatusBadgeState: JunoStatusBadge.State {
        if !brokerHealthy { return .warning }
        switch controller.hudState {
        case .idle: return .ok
        case .listening, .partialCommit, .waitingSpeech: return .ok
        case .checkingCapability, .checkingMic, .refining: return .neutral
        case .blocked, .error: return .error
        case .unknown: return .ok
        }
    }

    private var isDictating: Bool {
        switch controller.hudState {
        case .listening, .partialCommit, .checkingMic, .waitingSpeech: return true
        default: return false
        }
    }

    private var isProcessing: Bool {
        controller.hudState == .refining || controller.hudState == .checkingCapability
    }

    @ViewBuilder
    private var menuBarLabel: some View {
        ZStack(alignment: .bottomTrailing) {
            Image(nsImage: JunoBrandMarkRasterizer.menuBarTemplateImage())
                .interpolation(.high)
                .resizable()
                .frame(width: 20, height: 22)
                .opacity(isProcessing ? 0.5 : 1.0)
                .animation(.easeInOut(duration: 0.25), value: isProcessing)

            // Live recording indicator dot
            if isDictating {
                Circle()
                    .fill(Color.red)
                    .frame(width: 5, height: 5)
                    .offset(x: 3, y: 3)
                    .transition(.scale(scale: 0.3).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isDictating)
    }
}

// MARK: - Menu-bar command center

private struct JunoMenuCommandCenter: View {
    @ObservedObject var controller: DictationController
    @ObservedObject var surface: SurfaceEditingModel
    @ObservedObject var updater: JunoUpdater
    let brokerHealthy: Bool
    let statusLabel: String
    let statusState: JunoStatusBadge.State
    let onboardingCompleted: Bool
    let recent: [UtteranceHistoryEntry]
    let recentLoaded: Bool
    let recentFailed: Bool

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            header
            recentSection
            commandGrid
            textTools
            footer
        }
        .padding(10)
        .frame(width: 344)
        .background(menuBackground)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Color.white.opacity(scheme == .dark ? 0.08 : 0.44))
                .frame(height: 0.6)
        }
        .tint(JunoDesignTokens.accent)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 8) {
                ZStack {
                    Circle()
                        .fill(JunoDesignTokens.iconBg)
                        .shadow(color: Color.black.opacity(scheme == .dark ? 0.22 : 0.10), radius: 7, y: 3)
                    JunoCommaMark(color: .white, scale: 0.66)
                        .frame(width: 11, height: 16)
                }
                .frame(width: 26, height: 26)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Juno")
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    JunoStatusBadge(state: statusState, label: statusLabel)
                        .scaleEffect(0.78, anchor: .leading)
                        .frame(height: 17, alignment: .leading)
                }

                Spacer(minLength: 6)

                Button {
                    runPrimaryAction()
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: primaryActionSymbol)
                            .font(.system(size: 10.4, weight: .semibold))
                            .symbolRenderingMode(.hierarchical)
                        Text(primaryActionTitle)
                            .font(.system(size: 10.8, weight: .semibold, design: .rounded))
                            .lineLimit(1)
                        if isProcessing && brokerHealthy {
                            ProgressView()
                                .controlSize(.small)
                                .scaleEffect(0.64)
                        }
                    }
                    .foregroundStyle(primaryActionForeground)
                    .padding(.horizontal, 10)
                    .frame(height: 26)
                    .background(primaryActionBackground)
                }
                .buttonStyle(.plain)
                .junoNoFocusRing()
                .disabled(isProcessing && brokerHealthy)
            }

            if !onboardingCompleted || !brokerHealthy {
                Text(headerMessage)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 2)
    }

    private var recentSection: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.s) {
            HStack {
                JunoEyebrow(text: "Recent")
                Spacer(minLength: 0)
                Button("See all") {
                    JunoMainWindow.show(surface: surface, controller: controller, section: .history)
                }
                .buttonStyle(.plain)
                .junoType(.label)
                .foregroundStyle(JunoDesignTokens.accent)
                .keyboardShortcut("h", modifiers: .command)
                .junoNoFocusRing()
            }

            if recent.isEmpty {
                emptyRecentState
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(recent.prefix(3).enumerated()), id: \.element.id) { index, entry in
                        JunoMenuRecentDictationRow(entry: entry) {
                            JunoMainWindowNavigator.shared.openHistory(utteranceId: entry.utteranceId)
                            JunoMainWindow.show(surface: surface, controller: controller, section: .history)
                        }
                        if index < min(recent.count, 3) - 1 {
                            Divider()
                                .padding(.leading, 42)
                        }
                    }
                }
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(rowSurface)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(hairline.opacity(scheme == .dark ? 0.72 : 0.52), lineWidth: 0.55)
                )
            }
        }
    }

    private var emptyRecentState: some View {
        HStack(alignment: .center, spacing: 9) {
                Image(systemName: recentFailed ? "wifi.exclamationmark" : "clock")
                    .font(.system(size: 12.5, weight: .medium))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(recentFailed ? Color.orange : JunoTheme.secondaryText(scheme))
                    .frame(width: 18, height: 18)
                VStack(alignment: .leading, spacing: 3) {
                    Text(emptyRecentTitle)
                        .font(.system(size: 11.5, weight: .semibold, design: .rounded))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                        .lineLimit(1)
                    Text(emptyRecentMessage)
                        .font(.system(size: 10.8, weight: .medium, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 4)
            if onboardingCompleted && brokerHealthy && !isDictating && !isProcessing {
                Button {
                    controller.toggleDictation()
                } label: {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 11.5, weight: .bold))
                        .frame(width: 26, height: 26)
                        .background(Circle().fill(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.12)))
                }
                .buttonStyle(.plain)
                .foregroundStyle(JunoDesignTokens.accent)
                .junoNoFocusRing()
                .help("Start dictation")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(rowSurface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(hairline.opacity(scheme == .dark ? 0.72 : 0.52), lineWidth: 0.55)
        )
    }

    private var commandGrid: some View {
        HStack(spacing: 4) {
            JunoMenuCommandTile(title: "Open", symbol: "macwindow", shortcut: "⇧⌘O") {
                JunoMainWindow.show(surface: surface, controller: controller)
            }
            .keyboardShortcut("o", modifiers: [.command, .shift])

            JunoMenuCommandTile(title: "History", symbol: "clock", shortcut: "⌘H") {
                JunoMainWindow.show(surface: surface, controller: controller, section: .history)
            }
            .keyboardShortcut("h", modifiers: .command)

            JunoMenuCommandTile(title: "Settings", symbol: "gearshape", shortcut: "⌘,") {
                JunoMainWindow.show(surface: surface, controller: controller, section: .settings)
            }
            .keyboardShortcut(",", modifiers: .command)

            JunoMenuCommandTile(title: "Help", symbol: "waveform.path.ecg") {
                JunoBrokerHelpWindow.show()
            }
        }
        .padding(3)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(rowSurface.opacity(scheme == .dark ? 0.74 : 0.82))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(hairline.opacity(scheme == .dark ? 0.54 : 0.34), lineWidth: 0.5)
        )
    }

    private var textTools: some View {
        HStack(spacing: 6) {
            JunoMenuInlineButton(title: "Transform clipboard", symbol: "wand.and.stars") {
                JunoShellSessionActions.runTransformPolishFromPasteboard()
            }
            JunoMenuInlineButton(title: "Rewrite selection", symbol: "text.cursor") {
                JunoShellSessionActions.runTransformPolishFromSelection()
            }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        }
    }

    private var footer: some View {
        // Hairline rule above the footer so it reads as a distinct strip,
        // not just text drifting at the bottom of the panel.
        VStack(spacing: JunoUI.Spacing.s) {
            JunoHairlineRule(.faint)
                .padding(.horizontal, -JunoUI.Spacing.s)
            HStack(spacing: JunoUI.Spacing.m) {
                JunoMenuFooterLink(
                    label: "Updates",
                    badge: updater.updateAvailable,
                    disabled: updater.isConfigured && !updater.canCheckForUpdates
                ) {
                    JunoUpdater.shared.checkForUpdates()
                }

                JunoMenuFooterLink(label: "Diagnostics", badge: false, disabled: false) {
                    JunoDiagnosticsWindow.show()
                }

                Spacer(minLength: 0)

                JunoMenuFooterLink(label: "Quit", badge: false, disabled: false) {
                    NSApp.terminate(nil)
                }
                .keyboardShortcut("q", modifiers: .command)
            }
            .padding(.horizontal, 2)
        }
    }

    private var menuBackground: some ShapeStyle {
        LinearGradient(
            colors: [
                JunoTheme.windowBackground(scheme),
                JunoTheme.cardBackground(scheme).opacity(scheme == .dark ? 0.84 : 0.76),
            ],
            startPoint: .top,
            endPoint: .bottom
        )
    }

    private var rowSurface: Color {
        scheme == .dark
            ? Color.white.opacity(0.055)
            : Color.white.opacity(0.58)
    }

    private var hairline: Color {
        scheme == .dark ? Color.white.opacity(0.12) : Color.black.opacity(0.08)
    }

    private var emptyRecentTitle: String {
        if !recentLoaded { return "Checking recent dictations" }
        if recentFailed { return "Recent dictations unavailable" }
        return "Your last few dictations will appear here"
    }

    private var emptyRecentMessage: String {
        if !recentLoaded { return "Checking local History on this Mac." }
        if recentFailed { return "Try again once the voice engine is ready." }
        return "Start speaking and this becomes a quick way back."
    }

    private var headerMessage: String {
        if !onboardingCompleted {
            return "Finish setup to enable dictation, permissions, and the local voice engine."
        }
        return "The voice engine is offline. Help can repair or restart Juno."
    }

    private var primaryActionTitle: String {
        if !onboardingCompleted { return "Finish setup" }
        if !brokerHealthy { return "Help" }
        if isDictating { return "Stop" }
        if isProcessing { return "Working…" }
        return "Dictate"
    }

    private var primaryActionSymbol: String {
        if !onboardingCompleted { return "checklist" }
        if !brokerHealthy { return "wrench.and.screwdriver" }
        if isDictating { return "stop.fill" }
        if isProcessing { return "waveform" }
        return "mic.fill"
    }

    private var primaryActionFill: Color {
        if !onboardingCompleted { return JunoDesignTokens.accent }
        if !brokerHealthy { return Color.orange.opacity(scheme == .dark ? 0.18 : 0.11) }
        if isDictating { return JunoDesignTokens.danger.opacity(scheme == .dark ? 0.18 : 0.10) }
        if isProcessing { return rowSurface }
        return scheme == .dark ? Color.white.opacity(0.10) : JunoDesignTokens.iconBg
    }

    private var primaryActionStroke: Color {
        if !onboardingCompleted { return JunoDesignTokens.accent.opacity(0.45) }
        if !brokerHealthy { return Color.orange.opacity(0.32) }
        if isDictating { return JunoDesignTokens.danger.opacity(0.35) }
        if isProcessing { return hairline.opacity(0.72) }
        return scheme == .dark ? Color.white.opacity(0.16) : Color.black.opacity(0.10)
    }

    private var primaryActionForeground: Color {
        if !onboardingCompleted { return .white }
        if !brokerHealthy { return Color.orange }
        if isDictating { return JunoDesignTokens.danger }
        if isProcessing { return JunoTheme.secondaryText(scheme) }
        return scheme == .dark ? JunoDesignTokens.paper : .white
    }

    private var primaryActionBackground: some View {
        Capsule(style: .continuous)
            .fill(primaryActionFill)
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(primaryActionStroke, lineWidth: 0.55)
            )
            .shadow(
                color: Color.black.opacity((!brokerHealthy || isDictating || isProcessing) ? 0 : (scheme == .dark ? 0.18 : 0.12)),
                radius: 8,
                y: 3
            )
    }

    private var isDictating: Bool {
        switch controller.hudState {
        case .listening, .partialCommit, .checkingMic, .waitingSpeech: return true
        default: return false
        }
    }

    private var isProcessing: Bool {
        controller.hudState == .refining || controller.hudState == .checkingCapability
    }

    private func runPrimaryAction() {
        if !onboardingCompleted {
            JunoMainWindow.show(surface: surface, controller: controller)
        } else if !brokerHealthy {
            JunoBrokerHelpWindow.show()
        } else if isDictating {
            controller.toggleDictation()
        } else if !isProcessing {
            controller.toggleDictation()
        }
    }
}

private struct JunoMenuRecentDictationRow: View {
    let entry: UtteranceHistoryEntry
    let action: () -> Void
    @Environment(\.colorScheme) private var scheme
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 7) {
                appIcon
                    .padding(.top, 2)
                Circle()
                    .fill(entry.outcomeColor)
                    .frame(width: 6, height: 6)
                    .padding(.top, 7)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 5) {
                        Text(entry.displayAppName)
                            .font(.system(size: 10.2, weight: .semibold, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .lineLimit(1)
                        Text(entry.historyTimestampLabel)
                            .font(.system(size: 9.6, weight: .medium, design: .monospaced))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.8))
                            .lineLimit(1)
                        Spacer(minLength: 0)
                    }
                Text(entry.historyPreviewText)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                        .lineLimit(2)
                        .lineSpacing(1.2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(hovered ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.10 : 0.055) : Color.clear)
            )
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                hovered = hovering
            }
        }
    }

    private var appIcon: some View {
        let bundleId = entry.context?.appBundleId
        let url = bundleId.flatMap { NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) }
        let image = url.map { NSWorkspace.shared.icon(forFile: $0.path) }
            ?? NSImage(systemSymbolName: "app", accessibilityDescription: nil)
            ?? NSImage()
        return Image(nsImage: image)
            .resizable()
            .scaledToFit()
            .frame(width: 16, height: 16)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}

private struct JunoMenuCommandTile: View {
    let title: String
    let symbol: String
    var shortcut: String? = nil
    let action: () -> Void
    @Environment(\.colorScheme) private var scheme
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: symbol)
                    .font(.system(size: 11.5, weight: .medium))
                    .symbolRenderingMode(.hierarchical)
                Text(title)
                    .junoType(.caption)
                    .lineLimit(1)
                    .minimumScaleFactor(0.78)
            }
            .foregroundStyle(JunoTheme.primaryText(scheme))
            .frame(maxWidth: .infinity)
            .frame(height: 40)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(hovered ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.12 : 0.06) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(hovered ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.20 : 0.12) : Color.clear, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
        .help(shortcut.map { "\(title) \($0)" } ?? title)
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                hovered = hovering
            }
        }
    }
}

private struct JunoMenuInlineButton: View {
    let title: String
    let symbol: String
    let action: () -> Void
    @Environment(\.colorScheme) private var scheme
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: symbol)
                    .font(.system(size: 10.6, weight: .medium))
                    .symbolRenderingMode(.hierarchical)
                Text(title)
                    .junoType(.caption)
                    .lineLimit(1)
                    .minimumScaleFactor(0.78)
            }
            .foregroundStyle(JunoTheme.primaryText(scheme))
            .padding(.horizontal, 8)
            .frame(height: 26)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(hovered ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.11 : 0.055) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(hovered ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.20 : 0.12) : Color.clear, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                hovered = hovering
            }
        }
    }
}

/// Footer link in the menu-bar panel. Renders as muted text by default,
/// brightens on hover, and shows a small accent dot when `badge == true`.
/// Replaces bare `.buttonStyle(.plain)` text links
/// as "understyled" — Updates in particular needs to surface availability.
private struct JunoMenuFooterLink: View {
    let label: String
    let badge: Bool
    let disabled: Bool
    let action: () -> Void

    @Environment(\.colorScheme) private var scheme
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: JunoUI.Spacing.xs) {
                Text(label)
                    .junoType(.label)
                if badge && !disabled {
                    Circle()
                        .fill(JunoDesignTokens.accent)
                        .frame(width: 5, height: 5)
                        .accessibilityLabel("Update available")
                }
            }
            .foregroundStyle(foreground)
            .padding(.horizontal, JunoUI.Spacing.s)
            .padding(.vertical, JunoUI.Spacing.xs)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(hovered && !disabled ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.10 : 0.05) : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
        .disabled(disabled)
        .onHover { h in
            withAnimation(JunoUI.Motion.dim) { hovered = h }
        }
    }

    private var foreground: Color {
        if disabled {
            return JunoTheme.secondaryText(scheme).opacity(0.45)
        }
        if hovered {
            return JunoTheme.primaryText(scheme)
        }
        return JunoTheme.secondaryText(scheme).opacity(0.86)
    }
}

// MARK: - App delegate (Dock clicks / launch behavior)

final class JunoShellRuntime {
    static let shared = JunoShellRuntime()
    var controller: DictationController?
    var surface: SurfaceEditingModel?
    var brokerProcess: Process?
    /// Set by ``HotkeyBridge.start`` so ``JunoShellAppDelegate.applicationWillTerminate``
    /// can clean up the long-running ``juno-hotkey`` child process on
    /// Cmd-Q. Without this teardown, the helper outlives the UI and
    /// becomes a zombie until the next Juno launch (observed during
    /// PR #32 lock-fix verification).
    var terminateHotkeyBridge: () -> Void = {}
    /// Starts the in-process hotkey monitors after onboarding completes.
    /// Keeping them out of the first-run text-entry flow prevents app-level
    /// key monitors from sitting in front of the onboarding name field.
    var startHotkeyBridge: () -> Void = {}
}

/// Opens the main shell window using the shared surface (menu-bar bootstrap runs in `App.init`, so this is safe before `MenuBarExtra` content mounts).
enum JunoShellWindowOpener {
    @MainActor
    static func showMainWindow(section: MainSidebar = .home) {
        guard JunoUserDefaults.onboardingCompleted else {
            JunoOnboardingWindow.showIfNeeded()
            JunoWindowActivation.activateApp()
            return
        }
        guard let surface = JunoShellRuntime.shared.surface else {
            junoOnboardingLog.error("showMainWindow skipped: Juno surface (runtime) is nil")
            return
        }
        guard let controller = JunoShellRuntime.shared.controller else {
            junoOnboardingLog.error("showMainWindow skipped: Juno controller (runtime) is nil")
            return
        }
        JunoMainWindow.show(surface: surface, controller: controller, section: section)
    }
}

enum JunoLaunchHealthAudit {
    private static let maxSetupStatusAttempts = 4
    private static let setupStatusRetryDelay: TimeInterval = 0.32

    @MainActor
    static func run() {
        guard JunoUserDefaults.onboardingCompleted else { return }
        JunoPermissionMonitor.shared.refresh()
        let needsPermissions = !JunoPermissionMonitor.shared.canDictate
        attemptFetchSetupStatus(attempt: 0, needsPermissions: needsPermissions)
    }

    @MainActor
    private static func attemptFetchSetupStatus(attempt: Int, needsPermissions: Bool) {
        JunoBroker.fetchSetupStatus { result in
            Task { @MainActor in
                switch result {
                case .success(let status):
                    let needsSetup = !(status.overallReady ?? false)
                    finishIfNeeded(needsPermissions: needsPermissions, needsSetup: needsSetup)
                case .failure:
                    if attempt + 1 < maxSetupStatusAttempts {
                        DispatchQueue.main.asyncAfter(deadline: .now() + setupStatusRetryDelay) {
                            attemptFetchSetupStatus(attempt: attempt + 1, needsPermissions: needsPermissions)
                        }
                        return
                    }
                    // Transient HTTP races are common right after `ensureRunningIfPossible`.
                    // If /healthz says the broker is warming, do not treat this as "needs setup"
                    // for navigation.
                    JunoBroker.pingHealthDetailed { snapshot in
                        Task { @MainActor in
                            if snapshot?.reachable == true, snapshot?.warmState == "warming" {
                                return
                            }
                            finishIfNeeded(needsPermissions: needsPermissions, needsSetup: true)
                        }
                    }
                }
            }
        }
    }

    @MainActor
    private static func finishIfNeeded(needsPermissions: Bool, needsSetup: Bool) {
        if needsPermissions || needsSetup {
            NotificationCenter.default.post(name: .junoOpenMainWindow, object: nil)
            JunoShellWindowOpener.showMainWindow(section: .home)
        }
    }
}

final class JunoShellAppDelegate: NSObject, NSApplicationDelegate {
    private var openWindowObservers: [NSObjectProtocol] = []

    func applicationWillFinishLaunching(_ notification: Notification) {
        // MenuBarExtra’s `onAppear` can run late; set the Dock tile early so SwiftPM
        // builds don’t sit on a blank/generic executable icon.
        JunoBrandMarkRasterizer.installDockIconIfNeeded()
        JunoUserDefaults.appearancePreference.applyToSharedApplication()
        // Launch splash window deleted: it was a floating-level NSWindow
        // with canBecomeKey=true that, after orderOut, lingered in the
        // process window list and silently captured key/main status from
        // the visible main window — the actual cause of the "main window
        // visible but unclickable after relaunch" bug. Loading state is
        // already surfaced on the Home page via the "Voice engine offline"
        // / phase badge in JunoHomeHeroCard, so the splash was redundant.
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Defensive no-op on supported systems. The real preflight gate runs
        // in `JunoShellApp.init` before helpers / polling / onboarding start.
        JunoSystemRequirements.enforceMinimumOSOrTerminate()
        registerOpenWindowNotificationsIfNeeded()
        JunoDockVisibility.applyCurrent()
        if JunoClickDeliveryProbe.isRequested {
            Task { @MainActor in
                // Exercise the PR #66 hotkey bridge path during the hardware
                // click probe. Calling twice intentionally verifies that
                // HotkeyBridge.start() remains idempotent and does not spawn
                // duplicate juno-hotkey helpers on relaunch/test startup.
                JunoShellRuntime.shared.startHotkeyBridge()
                JunoShellRuntime.shared.startHotkeyBridge()
                JunoClickDeliveryProbe.showIfRequested()
            }
            return
        }
        if JunoUserDefaults.onboardingCompleted {
            // Drive the engine through its launch phases via the unified
            // lifecycle. ``boot`` is idempotent and itself calls
            // ``JunoLocalBrokerBootstrap.ensureRunningIfPossible`` after a
            // preflight check, so we no longer need the legacy deferred spawn
            // here. The supervisor still observes the live process for ongoing
            // health/respawn — lifecycle just owns the launch-time UX.
            Task { @MainActor in
                JunoShellRuntime.shared.startHotkeyBridge()
                JunoEngineLifecycle.shared.boot()
            }
            // Menu-bar app: show main chrome when launched from Finder/Dock.
            showMainWindowEventually()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
                Task { @MainActor in
                    JunoLaunchHealthAudit.run()
                }
            }
        } else {
            JunoOnboardingWindow.showIfNeeded()
        }
    }

    /// Clean up long-running child processes that would otherwise
    /// outlive the UI. Best-effort: fires on the standard AppKit-clean
    /// termination paths (Cmd-Q from the menu bar, ``NSApp.terminate(_:)``,
    /// "Juno > Quit" menu item). It does NOT fire on SIGKILL — the
    /// lockfile O_CLOEXEC fix from PR #32 already prevents the orphan
    /// helpers from blocking the next launch in that case. AppleScript
    /// "tell application Juno to quit" against an ad-hoc-signed bundle
    /// also bypasses the handler in our testing; the handler is wired
    /// to the AppDelegate, so anything that doesn't reach NSApp's
    /// shutdown sequence won't trigger it.
    ///
    /// Why we only kill these two specifically: ``juno-paste``,
    /// ``juno-capability``, ``juno-host``, and ``juno-textmon`` are
    /// invoked synchronously with ``waitUntilExit()`` and can't outlive
    /// the UI. Only the workbench python process (long-running HTTP
    /// server) and the hotkey bridge (long-running event reader) are
    /// long-running children that can become orphans.
    func applicationWillTerminate(_ notification: Notification) {
        NSLog("Juno: applicationWillTerminate — cleaning up child processes")
        // Stop the supervisor first so it does not fire a final ping
        // against an engine we are about to SIGTERM. Called directly
        // (not via Task) so it runs before the app finishes terminating;
        // ``applicationWillTerminate`` is itself on the main thread and
        // ``Timer.invalidate()`` is safe to call here.
        JunoEngineSupervisor.shared.stop()
        if let proc = JunoShellRuntime.shared.brokerProcess, proc.isRunning {
            proc.terminate()
        }
        JunoShellRuntime.shared.terminateHotkeyBridge()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if JunoClickDeliveryProbe.isRequested {
            Task { @MainActor in
                JunoShellRuntime.shared.startHotkeyBridge()
                JunoShellRuntime.shared.startHotkeyBridge()
                JunoClickDeliveryProbe.showIfRequested()
            }
            return true
        }
        if JunoUserDefaults.onboardingCompleted {
            // Cheap health re-probe so a dead engine after a previous Dock
            // close doesn't greet the user with a buggy main window. No-op
            // when we're already healthy; degrades + re-spawns otherwise.
            Task { @MainActor in
                JunoEngineLifecycle.shared.reprobe(reason: .appReopen)
            }
            // Dock click should not reset navigation. If we already have a window controller,
            // just bring it forward without touching `JunoMainWindowNavigator.shared.section`.
            Task { @MainActor in
                if !flag, JunoMainWindow.activateIfPresent() {
                    // Same deferred kick as ``showMainWindowEventually``: Dock reopen with an
                    // existing controller only ran ``bringToFront`` once — before SwiftUI had
                    // installed first responder — leaving the main window visible but unclickable.
                    scheduleDeferredMainWindowActivationKickForReopen()
                    return
                }
                if flag, JunoMainWindow.activateIfPresent() {
                    scheduleDeferredMainWindowActivationKickForReopen()
                    return
                }
                showMainWindowEventually()
            }
        } else {
            JunoOnboardingWindow.showIfNeeded()
        }
        return true
    }

    private func registerOpenWindowNotificationsIfNeeded() {
        guard openWindowObservers.isEmpty else { return }
        let nc = NotificationCenter.default
        let q = OperationQueue.main
        openWindowObservers.append(nc.addObserver(forName: .junoOpenMainWindow, object: nil, queue: q) { _ in
            Task { @MainActor in JunoShellWindowOpener.showMainWindow(section: .home) }
        })
        openWindowObservers.append(nc.addObserver(forName: .junoOpenSettingsWindow, object: nil, queue: q) { _ in
            Task { @MainActor in JunoShellWindowOpener.showMainWindow(section: .settings) }
        })
        openWindowObservers.append(nc.addObserver(forName: .junoOpenHistoryWindow, object: nil, queue: q) { _ in
            Task { @MainActor in JunoShellWindowOpener.showMainWindow(section: .history) }
        })
    }

    /// Second ``activateIfPresent`` after SwiftUI layout settles (see ``showMainWindowEventually``).
    private func scheduleDeferredMainWindowActivationKickForReopen() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            Task { @MainActor in
                _ = JunoMainWindow.activateIfPresent()
            }
        }
    }

    private func showMainWindowEventually(remainingAttempts: Int = 20) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            if JunoShellRuntime.shared.surface != nil {
                Task { @MainActor in
                    // Preserve whatever section the user was last on (default is `.home`).
                    let current = JunoMainWindowNavigator.shared.section
                    JunoShellWindowOpener.showMainWindow(section: current)
                }
                // RELAUNCH-PATH ACTIVATION FIX: on first launch, the main
                // window is shown twice in quick succession (once via the
                // .junoOpenMainWindow notification handler, once directly
                // after the onboarding-finish handler completes) — and that
                // double-bringToFront is what makes activation stick.
                //
                // On relaunch the onboarding path is skipped, so this is the
                // only showMainWindow call. The single bringToFront fires
                // before the SwiftUI hosting view has fully laid out and
                // claimed first responder, and AppKit ends up routing events
                // to a dead first-responder slot — symptom: "main window
                // visible after relaunch but unclickable, AX still works,
                // window has focused=false even though AXMain=true".
                //
                // Fire a second activate-and-orderFront 350ms later, after
                // the SwiftUI tree has settled. activateIfPresent is a
                // no-op if the window controller doesn't exist; otherwise
                // it routes through JunoWindowActivation.bringToFront,
                // which sets .regular policy + activate + orderFrontRegardless
                // + makeKeyAndOrderFront with a follow-up tick.
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    Task { @MainActor in
                        _ = JunoMainWindow.activateIfPresent()
                    }
                }
            } else if remainingAttempts > 0 {
                self.showMainWindowEventually(remainingAttempts: remainingAttempts - 1)
            } else {
                junoOnboardingLog.error("showMainWindowEventually gave up: surface still nil after retries")
            }
        }
    }
}

// MARK: - Memory management window

enum MemoryManagementWindow {
    private static var windowController: NSWindowController?

    @MainActor
    static func show() {
        if let existing = windowController, let w = existing.window {
            JunoWindowActivation.bringToFront(w)
            return
        }
        let content = MemoryManagementView()
        let hosting = NSHostingController(rootView: content)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Juno · Memory"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 720, height: 480))
        window.center()
        window.isReleasedWhenClosed = false
        let controller = NSWindowController(window: window)
        windowController = controller
        controller.showWindow(nil)
        JunoWindowActivation.bringToFront(window)
    }
}
