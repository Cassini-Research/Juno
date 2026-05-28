import ApplicationServices
import AVFoundation
import Combine
import Speech
import AppKit

// MARK: - Permission monitor

/// Shared observable that tracks microphone, Accessibility, and speech-recognition
/// auth status. Polls occasionally and refreshes on app-activation events so views
/// auto-update without requiring explicit user action after visiting System Settings.
@MainActor
final class JunoPermissionMonitor: ObservableObject {
    static let shared = JunoPermissionMonitor()

    @Published private(set) var micStatus: AVAuthorizationStatus = .notDetermined
    @Published private(set) var axGranted: Bool = false
    @Published private(set) var speechStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined

    /// True when the minimum set of permissions for dictation is in place.
    var canDictate: Bool { micStatus == .authorized && axGranted }

    private var timer: AnyCancellable?
    private var activationObserver: Any?
    private var appActiveObserver: NSObjectProtocol?
    private var accessibilityPromptShownThisSession = false

    private init() {
        refresh()
    }

    func startMonitoring() {
        guard timer == nil else { return }
        // Permission state changes are normally discovered when the app is
        // activated after System Settings. Keep a slow background poll only as
        // a safety net; frequent AX/Speech checks create TCC log churn and
        // keep an otherwise idle menu-bar app awake.
        timer = Timer.publish(every: 60, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.refresh() }

        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            // Refresh when focus returns to Juno after the user may have
            // changed settings in another app (e.g. System Settings).
            Task { @MainActor [weak self] in self?.refresh() }
        }

        // Menu-bar apps often miss workspace activation; this fires when Juno is foregrounded again.
        appActiveObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in self?.refresh() }
        }
    }

    func stopMonitoring() {
        timer = nil
        if let obs = activationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(obs)
            activationObserver = nil
        }
        if let o = appActiveObserver {
            NotificationCenter.default.removeObserver(o)
            appActiveObserver = nil
        }
    }

    func refresh() {
        let prevCanDictate = canDictate
        micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        // Use the same trust signal as the capability gate and paste path.
        // A listen-only CGEvent tap can succeed in states where AX queries and
        // Cmd+V injection still fail, which made the UI report "granted" while
        // paste was actually blocked.
        axGranted = AXIsProcessTrusted()
        speechStatus = SFSpeechRecognizer.authorizationStatus()
        // Wake the lifecycle when canDictate flips. Without this, a user
        // stuck on the "Finish setup" gate at .needsPermissions stays
        // there indefinitely after granting perms in System Settings,
        // because the lifecycle's waitForSetup() exits on .needsPermissions
        // and never re-evaluates.
        if canDictate != prevCanDictate {
            NotificationCenter.default.post(
                name: .junoPermissionsCanDictateChanged, object: canDictate
            )
        }
    }

    func requestMic(completion: @escaping (Bool) -> Void = { _ in }) {
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] _ in
            DispatchQueue.main.async {
                self?.refresh()
                completion(self?.micStatus == .authorized)
            }
        }
    }

    func requestSpeechRecognition(completion: @escaping (Bool) -> Void = { _ in }) {
        SFSpeechRecognizer.requestAuthorization { [weak self] _ in
            DispatchQueue.main.async {
                self?.refresh()
                completion(self?.speechStatus == .authorized)
            }
        }
    }

    /// Open the Accessibility pane in System Settings. Used by explicit
    /// "Open Accessibility" buttons in onboarding and Settings — the user
    /// asked to navigate there, that's exactly what we do, no system
    /// sheet, no second window.
    func openAccessibilitySettings() {
        refresh()
        guard !axGranted else { return }
        openAXSettings()
    }

    /// Just-in-time AX nudge for the dictation path. Fires the macOS-native
    /// trust sheet at most once per app session — that sheet has its own
    /// "Open System Settings" button so we don't open Settings ourselves.
    /// Without the once-per-session guard, every blocked hotkey press
    /// would re-trigger the sheet and steal focus mid-paste.
    func nudgeAccessibilityPrompt() {
        refresh()
        guard !axGranted else { return }
        guard !accessibilityPromptShownThisSession else { return }
        accessibilityPromptShownThisSession = true
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let opts = [key: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
    }

    func openMicSettings() {
        JunoSystemSettingsLinks.openMicrophonePrivacy()
    }

    func openAXSettings() {
        JunoSystemSettingsLinks.openAccessibilityPrivacy()
    }

    func openSpeechRecognitionSettings() {
        JunoSystemSettingsLinks.openSpeechRecognitionPrivacy()
    }

    // MARK: - Display helpers

    var micStatusLabel: String {
        switch micStatus {
        case .authorized: return "Granted"
        case .denied: return "Denied — open System Settings to grant."
        case .notDetermined: return "Not yet requested."
        case .restricted: return "Restricted by system policy."
        @unknown default: return "Unknown"
        }
    }

    var speechStatusLabel: String {
        switch speechStatus {
        case .authorized: return "Granted"
        case .denied: return "Denied — open System Settings to grant."
        case .notDetermined: return "Not yet requested."
        case .restricted: return "Restricted by system policy."
        @unknown default: return "Unknown"
        }
    }
}
