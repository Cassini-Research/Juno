// JunoActionChip.swift
//
// Ephemeral HUD row for a voice action while the executor runs. Paired with
// ``JunoActionRequest`` / ``JunoActionResult`` from ``JunoActionDTOs``.

import Foundation

struct JunoActionChip: Identifiable, Hashable {
    let id: String
    let kind: JunoActionKind
    let bodyPreview: String
    var status: JunoActionStatus?
    var errorMessage: String?

    init(from request: JunoActionRequest) {
        self.id = request.id
        self.kind = request.kind
        self.bodyPreview = request.body
        self.status = nil
        self.errorMessage = nil
    }

    private init(
        id: String,
        kind: JunoActionKind,
        bodyPreview: String,
        status: JunoActionStatus?,
        errorMessage: String?
    ) {
        self.id = id
        self.kind = kind
        self.bodyPreview = bodyPreview
        self.status = status
        self.errorMessage = errorMessage
    }

    static func merge(initial: [JunoActionChip], results: [JunoActionResult]) -> [JunoActionChip] {
        var out: [JunoActionChip] = []
        let n = max(initial.count, results.count)
        for i in 0..<n {
            if i < initial.count, i < results.count {
                let chip = initial[i]
                let r = results[i]
                out.append(
                    JunoActionChip(
                        id: chip.id,
                        kind: chip.kind,
                        bodyPreview: chip.bodyPreview,
                        status: r.status,
                        errorMessage: r.error
                    )
                )
            } else if i < initial.count {
                out.append(initial[i])
            }
        }
        return out
    }
}
