import AppKit
import Combine
import Foundation

/// Refreshes ``juno-capability`` + broker ``editing_profile`` for HUD / Home labels.
final class SurfaceEditingModel: ObservableObject {
    @Published var editingStyle: String = ""
    @Published var appName: String = ""
    @Published var appCategory: String = ""
    /// Frontmost app bundle id (from capability JSON or broker profile when present).
    @Published var appBundleId: String = ""

    private var timer: AnyCancellable?

    func startPolling() {
        timer?.cancel()
        timer = Timer.publish(every: 20.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.refresh()
            }

        _ = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.refresh()
        }

        refresh()
    }

    func refresh() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let cap = Self.readCapabilityPayload()
            let capBundle = (cap["app_bundle_id"] as? String)
                ?? (cap["frontmost_app_bundle_id"] as? String)
                ?? ""
            JunoBroker.postEditingProfile(payload: cap) { result in
                DispatchQueue.main.async {
                    guard let self else { return }
                    switch result {
                    case .success(let p):
                        self.editingStyle = p.editingStyle ?? ""
                        self.appName = p.appName ?? (cap["app_name"] as? String) ?? ""
                        self.appCategory = p.appCategory ?? ""
                        let profileBundle = (p.appBundleId ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                        self.appBundleId = profileBundle.isEmpty ? capBundle : profileBundle
                    case .failure:
                        self.appBundleId = capBundle
                    }
                }
            }
        }
    }

    private static func readCapabilityPayload() -> [String: Any] {
        JunoCapabilitySnapshot.capture()
    }
}
