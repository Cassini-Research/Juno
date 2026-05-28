import AppKit
import Foundation

enum JunoHUDSound {
    private static let fileName = "Juno_HUD_Sound"
    private static let fileExtension = "mp3"
    private static let packagedResourcePath = "Contents/Resources/JunoShell_JunoShell.bundle/Juno_HUD_Sound.mp3"

    /// Min spacing between two consecutive HUD sounds. Multiple call sites
    /// (open + close + the old copy-button cue) used to fire on the same UI
    /// transition, producing a stutter. The gate dedupes any cluster.
    private static let minIntervalSeconds: TimeInterval = 0.18
    private static var lastPlayAt: TimeInterval = 0
    private static let gate = NSLock()

    /// Cached MP3 bytes. Loaded once on first use (or via ``prewarm()``)
    /// so subsequent plays don't pay disk I/O on the main thread. Each
    /// play wraps these bytes in a fresh ``NSSound(data:)`` because
    /// ``NSSound`` instances cannot be reliably re-played mid-flight —
    /// constructing from in-memory data avoids the file-load cost that
    /// previously made the close cue lag the visual fade by 30–100 ms.
    private static let cachedData: Data? = {
        guard let url = packagedResourceURL() else {
            return nil
        }
        return try? Data(contentsOf: url)
    }()

    private static func packagedResourceURL() -> URL? {
        let appResourceURL = Bundle.main.bundleURL.appendingPathComponent(packagedResourcePath, isDirectory: false)
        if FileManager.default.isReadableFile(atPath: appResourceURL.path) {
            return appResourceURL
        }
        return Bundle.module.url(forResource: fileName, withExtension: fileExtension)
    }

    /// Optional warm-up. Touching ``cachedData`` here lets the app pull the
    /// MP3 off disk during launch instead of on the first user transition.
    static func prewarm() {
        DispatchQueue.global(qos: .utility).async {
            _ = cachedData
        }
    }

    private static func playGated(volume: Float) {
        gate.lock()
        let now = Date().timeIntervalSinceReferenceDate
        let allowed = now - lastPlayAt >= minIntervalSeconds
        if allowed { lastPlayAt = now }
        gate.unlock()
        guard allowed, let data = cachedData, let sound = NSSound(data: data) else { return }
        sound.volume = volume
        sound.play()
    }

    /// Played once when the HUD transitions idle → visible.
    static func playOpen() {
        playGated(volume: 0.72)
    }

    /// Played once when the HUD transitions visible → idle (fade-out start).
    /// Slightly quieter so it lands as a soft "out" rather than a second beat.
    static func playClose() {
        playGated(volume: 0.42)
    }

    /// Legacy entry point — kept for any older call sites; routes to open.
    @available(*, deprecated, message: "Use playOpen() / playClose() instead.")
    static func playMain(volume: Float = 0.72) {
        playGated(volume: volume)
    }
}
