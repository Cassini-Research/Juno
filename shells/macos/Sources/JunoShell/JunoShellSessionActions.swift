import AppKit
import Foundation

/// Menu-driven broker session actions (Transform) for the macOS shell.
enum JunoShellSessionActions {
    /// Matches `SurfaceId.MAC_OVERLAY` in `juno_core_v3/policy/surface_gate.py`.
    private static let macSurfaceId = "mac_overlay"

    private static func frontmostPidForSession() -> pid_t {
        NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
    }

    /// Uses clipboard string as `selected_text` for `POST /api/broker/session/transform`.
    @MainActor
    static func runTransformPolishFromPasteboard() {
        let frontPid = frontmostPidForSession()
        let pb = NSPasteboard.general
        let selected = (pb.string(forType: .string) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !selected.isEmpty else {
            JunoSessionResultWindow.presentTransformEmptyClipboardGuidance()
            return
        }
        let payload: [String: Any] = [
            "selected_text": selected,
            "surface_id": macSurfaceId,
            "hint": "polish",
            "transform_id": "polish",
        ]
        JunoBroker.post(path: "api/broker/session/transform", payload: payload) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let data):
                    let parsed = JunoBrokerSessionResponse.parse(data)
                    JunoSessionResultWindow.presentBrokerResponse(parsed, flow: .transformFromClipboard, frontmostPid: frontPid)
                case .failure(let err):
                    JunoSessionResultWindow.presentTransportFailure(
                        flow: .transformFromClipboard,
                        localizedDescription: err.localizedDescription,
                        frontmostPid: frontPid
                    )
                }
            }
        }
    }

    /// Uses the current Accessibility selection as `selected_text` for
    /// `POST /api/broker/session/transform`, then shows a preview-first
    /// result window so the user can paste the rewritten text back.
    @MainActor
    static func runTransformPolishFromSelection() {
        let frontPid = frontmostPidForSession()
        let snapshot = JunoCapabilitySnapshot.capture()
        let selected = ((snapshot["selected_text"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !selected.isEmpty else {
            JunoSessionResultWindow.presentTransformEmptySelectionGuidance()
            return
        }

        let payload: [String: Any] = [
            "selected_text": selected,
            "surface_id": macSurfaceId,
            "hint": "polish",
            "transform_id": "polish",
            "transform_source": "selection",
        ]
        JunoBroker.post(path: "api/broker/session/transform", payload: payload) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let data):
                    let parsed = JunoBrokerSessionResponse.parse(data)
                    JunoSessionResultWindow.presentBrokerResponse(parsed, flow: .transformSelection, frontmostPid: frontPid)
                case .failure(let err):
                    JunoSessionResultWindow.presentTransportFailure(
                        flow: .transformSelection,
                        localizedDescription: err.localizedDescription,
                        frontmostPid: frontPid
                    )
                }
            }
        }
    }
}
