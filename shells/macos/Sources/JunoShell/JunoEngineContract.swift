import Foundation

// MARK: - Shell ↔ loopback engine contract (see juno_v2/runtime/shell_engine_contract.py)

/// Version handshake for ``GET /api/broker/engine/compatibility``.
/// Must match ``SHELL_ENGINE_PROTOCOL_VERSION`` in Python; bump both together.
///
/// v2 (production-grade revamp, 2026-04-30): the compatibility response
/// now carries ``runtime_role``, ``instance_id``, ``bundle_id``, ``pid``.
/// The shell refuses to attach unless ``runtime_role`` matches
/// ``requiredRuntimeRole`` below. This eliminates the "another developer
/// process answered /healthz on the same port" silent-misattach class.
///
/// v3 (2026-05-03): the compatibility response also carries the active
/// deployment profile. The shell rejects engines whose live runtime contract
/// no longer matches the bundled production profile.
///
/// v4 (2026-05-05): the bundled macOS production profile reports the
/// writer residency policy so stale packaged engines can be rejected.
///
/// v5 (2026-05-13): live preview is a separate local HTTP service so captions
/// do not share the runtime process MLX lock with Qwen.
///
/// v6 (2026-05-23): packaged production writer residency changed to
/// on-demand with TTL reaping so background idle does not keep Qwen pinned.
enum JunoEngineContract {
    static let shellEngineProtocolVersion: Int = 6

    /// Canonical ``runtime_role`` reported by ``runtime.service`` (the
    /// production engine). Standalone ``python -m juno_v2.workbench.server``
    /// reports a different role and is rejected.
    static let requiredRuntimeRole: String = "juno_runtime_service"

    /// Default bundle identifier; overridden by ``Bundle.main.bundleIdentifier``
    /// at runtime. Kept here so the engine helper script can default to the
    /// same value when ``JUNO_BUNDLE_ID`` is not set in the spawn env.
    static let defaultBundleId: String = "com.juno.shell"

    static let expectedPreviewBackend: String = "streaming_local_http_json"
    static let expectedFinalBackend: String = "mlx_whisper"
    static let expectedWriterBackend: String = "mlx_lm"
    static let expectedWriterResidencyPolicy: String = "on_demand"

    /// ``Juno.app/Contents/Resources/engine`` when ``run_engine.sh`` is packaged.
    static func bundledEngineRoot() -> URL? {
        let d = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/engine", isDirectory: true)
        let sh = d.appendingPathComponent("run_engine.sh", isDirectory: false)
        let fm = FileManager.default
        guard fm.isExecutableFile(atPath: sh.path) else { return nil }
        let bundledPython = d.appendingPathComponent(".venv/bin/python", isDirectory: false)
        let packageDir = d.appendingPathComponent("site/juno_v2", isDirectory: true)
        let flatPackageDir = d.appendingPathComponent("juno_v2", isDirectory: true)
        guard fm.isExecutableFile(atPath: bundledPython.path)
                || fm.fileExists(atPath: packageDir.path)
                || fm.fileExists(atPath: flatPackageDir.path)
        else {
            return nil
        }
        return d
    }

    /// Packaged copy of ``scripts/install_juno_launchd.sh`` for non-checkout installs.
    static func bundledLaunchdInstallerURL() -> URL? {
        let u = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources/scripts/install_juno_launchd.sh", isDirectory: false)
        guard FileManager.default.fileExists(atPath: u.path) else { return nil }
        return u
    }
}
