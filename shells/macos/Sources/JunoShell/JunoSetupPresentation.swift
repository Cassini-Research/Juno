import Foundation

enum JunoSetupLaneRole: String, CaseIterable, Identifiable {
    case preview
    case liveCorrector
    case final
    case writer

    var id: String { rawValue }

    var title: String {
        switch self {
        case .preview: return "Live captions"
        case .liveCorrector: return "Live correction"
        case .final: return "High-quality transcription"
        case .writer: return "Smart formatting"
        }
    }

    var caption: String {
        switch self {
        case .preview: return "Shows what you're saying as you speak."
        case .liveCorrector: return "Fixes stable caption text while you keep speaking."
        case .final: return "The accurate pass that lands after each pause."
        case .writer: return "Cleans up punctuation, casing, and stray filler."
        }
    }

    var symbol: String {
        switch self {
        case .preview: return "waveform.circle.fill"
        case .liveCorrector: return "text.viewfinder"
        case .final: return "text.badge.checkmark"
        case .writer: return "wand.and.stars"
        }
    }
}

struct JunoSetupLaneViewModel: Identifiable {
    let role: JunoSetupLaneRole
    let ready: Bool
    let required: Bool
    let repoId: String
    let modelName: String

    var id: String { role.id }
    var title: String { role.title }
    var caption: String { role.caption }
    var symbol: String { role.symbol }
}

enum JunoSetupPresentation {
    @MainActor
    static func laneItems(from setup: JunoSetupModel) -> [JunoSetupLaneViewModel] {
        let checkByName = Dictionary(uniqueKeysWithValues: setup.checks.map { ($0.name, $0) })
        let previewReady = setup.checks.first { $0.name == "preview_model" }?.ok ?? setup.previewModelReady
        let liveCorrectorReady = setup.checks.first { $0.name == "live_corrector_model" }?.ok ?? setup.liveCorrectorModelReady
        let finalReady = setup.checks.first { $0.name == "final_model" }?.ok ?? setup.finalModelReady
        let writerReady = setup.checks.first { $0.name == "writer_model" }?.ok ?? setup.writerModelReady
        let writerBackend = setup.writerBackend.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let writerRequired = setup.writerRequired || !(writerBackend.isEmpty || writerBackend == "none")

        return [
            JunoSetupLaneViewModel(
                role: .preview,
                ready: previewReady,
                required: true,
                repoId: setup.previewRepoId,
                modelName: displayModelName(
                    repoId: setup.previewRepoId,
                    checkDetail: checkByName["preview_model"]?.detail
                )
            ),
            JunoSetupLaneViewModel(
                role: .liveCorrector,
                ready: liveCorrectorReady,
                required: setup.liveCorrectorRequired,
                repoId: setup.liveCorrectorRepoId,
                modelName: displayModelName(
                    repoId: setup.liveCorrectorRepoId,
                    checkDetail: checkByName["live_corrector_model"]?.detail
                )
            ),
            JunoSetupLaneViewModel(
                role: .final,
                ready: finalReady,
                required: true,
                repoId: setup.finalRepoId,
                modelName: displayModelName(
                    repoId: setup.finalRepoId,
                    checkDetail: checkByName["final_model"]?.detail
                )
            ),
            JunoSetupLaneViewModel(
                role: .writer,
                ready: writerReady,
                required: writerRequired,
                repoId: setup.writerRepoId,
                modelName: displayModelName(
                    repoId: setup.writerRepoId,
                    checkDetail: checkByName["writer_model"]?.detail
                )
            ),
        ]
    }

    static func displayModelName(repoId: String, checkDetail: String?) -> String {
        let repo = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = modelIdFromCheckDetail(checkDetail) ?? ""
        let candidate = !repo.isEmpty ? repo : fallback
        guard !candidate.isEmpty else { return "Model pending" }
        return prettifyModelSlug(candidate)
    }

    private static func modelIdFromCheckDetail(_ detail: String?) -> String? {
        guard let detail else { return nil }
        let parts = detail.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        for part in parts {
            if part.hasPrefix("path=") {
                let value = String(part.dropFirst("path=".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty {
                    return value
                }
            }
            if part.hasPrefix("backend=") {
                // If path is empty in runtime-health responses, backend often carries
                // the active model/backend identity, e.g. "streaming_local_http_json".
                let backend = String(part.dropFirst("backend=".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                if !backend.isEmpty, backend != "none" {
                    return backend
                }
            }
        }
        return nil
    }

    private static func prettifyModelSlug(_ modelId: String) -> String {
        let slug = modelId.split(separator: "/").last.map(String.init) ?? modelId
        let replaced = slug.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
        let words = replaced.split(separator: " ").map { part -> String in
            if part.uppercased() == part {
                return String(part)
            }
            if part.count <= 3 {
                return part.uppercased()
            }
            return part.prefix(1).uppercased() + part.dropFirst()
        }
        return words.joined(separator: " ")
    }
}
