import AppKit
import Darwin
import Foundation

/// Ensures only one Juno UI process per user. Uses an flock lock in
/// Application Support so duplicate launches (Terminal, Finder, scripts)
/// exit immediately and attempt to foreground the existing instance.
enum JunoSingleInstance {
    private static var lockFD: Int32 = -1

    /// Call once at startup before creating windows. Exits the process if
    /// another Juno already holds the lock.
    ///
    /// Two correctness fixes vs the original implementation:
    ///
    /// * `O_CLOEXEC` on the lock fd. Without it, every spawned helper
    ///   (`juno-hotkey`, `juno-paste`, the bundled python engine, etc.)
    ///   inherits the lockfile descriptor — so when the Juno UI process
    ///   crashes but a helper outlives it (common when the workbench engine
    ///   is still running an HTTP server), the helper keeps the `flock`
    ///   held and the next Juno launch silently exits with "Juno is already
    ///   running". Setting `O_CLOEXEC` makes the kernel drop the fd before
    ///   `exec` so the lock truly tracks the UI process lifetime.
    ///
    /// * Stale-PID recovery. If `flock` blocks, we read the PID written by
    ///   the previous instance and check it with `kill(pid, 0)`. If the
    ///   process is gone (`ESRCH`) the lock file is genuinely stale (a
    ///   kernel edge case or filesystem hiccup) and we unlink + retry. If
    ///   the PID is alive we keep the existing "another Juno is running"
    ///   behavior and try to activate the peer.
    static func exitIfAlreadyRunning() {
        // Production-grade path: ``<support-root>/runtime/instance.lock``
        // where ``<support-root>`` is keyed off the bundle id. Falls back
        // to the legacy ``Application Support/Juno/JunoShell.lock`` only
        // long enough to free a stale lock from a prior install.
        guard let runtimeDir = JunoSupportPaths.runtimeDir() else { return }
        let path = runtimeDir.appendingPathComponent("instance.lock").path

        // One-shot migration: if a legacy lockfile exists from a prior
        // install and is no longer held, unlink it so we don't leave
        // orphaned state cluttering Application Support.
        if let legacy = JunoSupportPaths.legacySupportRoot()?
            .appendingPathComponent("JunoShell.lock").path,
            FileManager.default.fileExists(atPath: legacy) {
            let legacyFD = open(legacy, O_RDWR | O_CLOEXEC)
            if legacyFD >= 0 {
                if flock(legacyFD, LOCK_EX | LOCK_NB) == 0 {
                    flock(legacyFD, LOCK_UN)
                    _ = Darwin.unlink(legacy)
                }
                _ = close(legacyFD)
            }
        }

        if let fd = takeLock(path: path) {
            persistOurPID(to: fd)
            return
        }

        // First flock attempt blocked. Distinguish between a live peer and
        // a genuinely stale lock before deciding whether to exit.
        let recordedPID = readPID(fromLockFileAt: path)
        let peerAlive = recordedPID.map { isAlive(pid: $0) } ?? false

        if !peerAlive {
            // Stale lockfile: unlink it and retry once. If a second process
            // raced us, the retry will block again and we'll fall into the
            // peer-alive branch below.
            _ = Darwin.unlink(path)
            if let fd = takeLock(path: path) {
                persistOurPID(to: fd)
                return
            }
        }

        // Real peer is running — activate it, surface a message on stderr,
        // and bow out. Without the message, double-launches look like a
        // crash or a broken Dock tile.
        if let pid = recordedPID, peerAlive {
            tryActivatePeer(pid: pid)
        }
        let msg = "Juno is already running — brought the existing app forward. This process exits.\n"
        if let data = msg.data(using: .utf8) {
            data.withUnsafeBytes { raw in
                if let base = raw.bindMemory(to: UInt8.self).baseAddress {
                    _ = write(STDERR_FILENO, base, data.count)
                }
            }
        }
        exit(0)
    }

    /// Try to acquire the exclusive lock. Returns the fd on success, nil if
    /// the file couldn't be opened or `flock` would block.
    private static func takeLock(path: String) -> Int32? {
        let fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0o644)
        guard fd >= 0 else { return nil }
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            _ = close(fd)
            return nil
        }
        lockFD = fd
        return fd
    }

    private static func persistOurPID(to fd: Int32) {
        ftruncate(fd, 0)
        _ = lseek(fd, 0, SEEK_SET)
        let pidLine = "\(ProcessInfo.processInfo.processIdentifier)\n"
        if let data = pidLine.data(using: .utf8) {
            data.withUnsafeBytes { raw in
                guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return }
                _ = write(fd, base, data.count)
            }
        }
    }

    private static func readPID(fromLockFileAt path: String) -> pid_t? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let line = String(data: data, encoding: .utf8)?
                .split(separator: "\n").first.map(String.init),
              let pid = Int32(line.trimmingCharacters(in: .whitespacesAndNewlines)),
              pid > 0,
              pid != ProcessInfo.processInfo.processIdentifier
        else { return nil }
        return pid
    }

    /// `kill(pid, 0)` returns 0 if the process exists, -1 with errno=ESRCH
    /// if it doesn't (and EPERM if it exists but we can't signal it — still
    /// alive). Both EPERM and 0 mean "alive"; only ESRCH means "gone".
    private static func isAlive(pid: pid_t) -> Bool {
        if kill(pid, 0) == 0 { return true }
        return errno != ESRCH
    }

    private static func tryActivatePeer(pid: pid_t) {
        let app = NSRunningApplication(processIdentifier: pid)
        app?.activate(options: [.activateAllWindows, .activateIgnoringOtherApps])
    }
}
