import AppKit
import SwiftUI

// MARK: - Broker wrappers for memory management

/// Thin async wrappers for the `/api/broker/memory/*` HTTP endpoints.
///
/// Contracts mirror `juno_v2/workbench/server.py`:
///   - GET  /api/broker/memory/vocab            → {ok, entries}
///   - POST /api/broker/memory/vocab            → upsert {term, canonical_form?, boost?, scope?}
///   - POST /api/broker/memory/vocab/remove     → {term}
///   - GET  /api/broker/memory/replacement      → {ok, entries}
///   - POST /api/broker/memory/replacement      → upsert {trigger, replacement, scope?}
///   - POST /api/broker/memory/replacement/remove
///   - GET/POST snippet + style + correction    — same shape.
///
/// We keep the surface tiny: a single `get`/`post` helper plus one
/// typed wrapper per category. The UI owns all state and issues
/// follow-up GET calls after mutations so the list always reflects
/// what the broker actually stored (catching silent server-side
/// rejections like invalid scopes).
extension JunoBroker {
    /// `GET path` returning a decoded `[String: Any]` JSON object.
    /// Any network / decode failure surfaces as an empty dict with
    /// `ok=false`, which the UI renders as an inline error state.
    static func getJSON(
        path: String,
        completion: @escaping ([String: Any]) -> Void
    ) {
        if path.hasPrefix("api/broker/") && path != "api/broker/engine/compatibility" {
            ensureCompatible { result in
                switch result {
                case .success:
                    getJSONAfterCompatibility(path: path, completion: completion)
                case .failure(let err):
                    completion([
                        "ok": false,
                        "error": err.localizedDescription,
                        "error_code": "engine_incompatible",
                    ])
                }
            }
            return
        }
        getJSONAfterCompatibility(path: path, completion: completion)
    }

    private static func getJSONAfterCompatibility(
        path: String,
        completion: @escaping ([String: Any]) -> Void
    ) {
        if shouldUseUDS {
            callBrokerRPC(httpMethod: "GET", path: path) { result in
                switch result {
                case .success(let out):
                    completion(out.object)
                case .failure:
                    completion(["ok": false, "error": "broker_unreachable"])
                }
            }
            return
        }
        let url = resolveURL(path: path)
        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                DispatchQueue.main.async {
                    completion(["ok": false, "error": "broker_unreachable"])
                }
                return
            }
            DispatchQueue.main.async { completion(obj) }
        }.resume()
    }

    /// `POST path` JSON-encoded payload; decoded JSON object result.
    static func postJSON(
        path: String,
        payload: [String: Any],
        completion: @escaping ([String: Any]) -> Void
    ) {
        post(path: path, payload: payload) { result in
            let obj: [String: Any]
            switch result {
            case .success(let data):
                obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])
                    ?? ["ok": false, "error": "bad_response"]
            case .failure(let err):
                obj = ["ok": false, "error": err.localizedDescription]
            }
            DispatchQueue.main.async { completion(obj) }
        }
    }

    static func deleteJSON(
        path: String,
        completion: @escaping ([String: Any]) -> Void
    ) {
        ensureCompatible { result in
            switch result {
            case .success:
                deleteJSONAfterCompatibility(path: path, completion: completion)
            case .failure(let err):
                completion([
                    "ok": false,
                    "error": err.localizedDescription,
                    "error_code": "engine_incompatible",
                ])
            }
        }
    }

    private static func deleteJSONAfterCompatibility(
        path: String,
        completion: @escaping ([String: Any]) -> Void
    ) {
        if shouldUseUDS {
            callBrokerRPC(httpMethod: "DELETE", path: path) { result in
                switch result {
                case .success(let out):
                    completion(out.object)
                case .failure:
                    completion(["ok": false, "error": "broker_unreachable"])
                }
            }
            return
        }
        let url = resolveURL(path: path)
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        JunoLocalBrokerAuth.attach(to: &req)
        URLSession.shared.dataTask(with: req) { data, _, _ in
            guard let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                DispatchQueue.main.async {
                    completion(["ok": false, "error": "broker_unreachable"])
                }
                return
            }
            DispatchQueue.main.async { completion(obj) }
        }.resume()
    }
}

// MARK: - View-model for the memory manager

/// Observable memory manager state. One instance lives per window.
/// Holds the five category lists + per-category form fields so the
/// UI doesn't need scattered `@State` that would lose its value on
/// tab-switch.
@MainActor
final class MemoryStoreViewModel: ObservableObject {
    // Lists (raw `[String: Any]` entries exactly as the broker
    // returns them). We render a subset of the keys the UI cares
    // about; the rest is kept around for display but not editable
    // from the shell — the workbench already shows everything.
    @Published var vocab: [[String: Any]] = []
    @Published var replacements: [[String: Any]] = []
    @Published var snippets: [[String: Any]] = []
    @Published var corrections: [[String: Any]] = []

    // Category-specific form state. Flat String fields so SwiftUI
    // can bind directly.
    @Published var vocabTerm: String = ""
    @Published var vocabCanonical: String = ""

    @Published var replacementTrigger: String = ""
    @Published var replacementText: String = ""
    @Published var replacementScope: String = "global"

    @Published var snippetTrigger: String = ""
    @Published var snippetBody: String = ""
    @Published var snippetScope: String = "global"

    @Published var statusMessage: String = ""
    @Published var statusIsSuccess: Bool = false
    @Published var isBusy: Bool = false

    private var statusClearTask: Task<Void, Never>? = nil

    /// Show a transient success banner at the page footer. Auto-clears
    /// after ~2.5 seconds so it doesn't linger past the moment the user
    /// would have looked away.
    func flashSuccess(_ message: String) {
        statusMessage = message
        statusIsSuccess = true
        statusClearTask?.cancel()
        statusClearTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            guard !Task.isCancelled else { return }
            guard let self else { return }
            // Only clear if the message is still the one we set (the
            // user may have triggered a fresh status in the meantime).
            if self.statusMessage == message {
                self.statusMessage = ""
                self.statusIsSuccess = false
            }
        }
    }

    // MARK: Refresh

    func refreshAll() {
        refreshCategory("vocab")
        refreshCategory("replacement")
        refreshCategory("snippet")
        refreshCategory("correction")
    }

    func refreshCategory(_ category: String) {
        JunoBroker.getJSON(path: "api/broker/memory/\(category)") { [weak self] obj in
            guard let self else { return }
            let entries = (obj["entries"] as? [[String: Any]]) ?? []
            switch category {
            case "vocab":
                self.vocab = entries
                self.seedJunoVocabIfNeeded()
            case "replacement":  self.replacements = entries
            case "snippet":      self.snippets = entries
            case "correction":   self.corrections = entries
            default: break
            }
        }
    }

    // MARK: Vocab

    /// Stored as fold-keys so "Juno", "JUNO", "j u n o" all collide on
    /// the same protected entry. Matches the Python broker's
    /// PROTECTED_VOCABULARY fold-key check in memory/store.py.
    static let protectedVocabTerms: Set<String> = ["juno"]

    static func isProtectedVocabTerm(_ raw: String) -> Bool {
        guard let key = JunoMemoryFold.foldKeyOrNil(raw) else { return false }
        return protectedVocabTerms.contains(key)
    }

    static func learnedVocabTermAllowed(_ raw: String) -> Bool {
        raw.unicodeScalars.reduce(0) { count, scalar in
            CharacterSet.alphanumerics.contains(scalar) ? count + 1 : count
        } >= 3
    }

    private func seedJunoVocabIfNeeded() {
        let alreadyPresent = vocab.contains {
            JunoMemoryFold.foldKey(($0["term"] as? String) ?? "") == "juno"
        }
        guard !alreadyPresent else { return }
        JunoBroker.postJSON(
            path: "api/broker/memory/vocab",
            payload: ["term": "Juno", "canonical_form": "Juno"]
        ) { [weak self] _ in self?.refreshCategory("vocab") }
    }

    func addVocab() {
        let term = vocabTerm.trimmingCharacters(in: .whitespaces)
        guard !term.isEmpty else {
            statusMessage = "Term cannot be empty"
            return
        }
        let canonical = vocabCanonical.trimmingCharacters(in: .whitespacesAndNewlines)
        guard Self.learnedVocabTermAllowed(term),
              Self.learnedVocabTermAllowed(canonical.isEmpty ? term : canonical) else {
            statusMessage = "Term must be at least 3 characters"
            return
        }
        if Self.isProtectedVocabTerm(term) {
            statusMessage = "That term is reserved"
            return
        }
        let payload: [String: Any] = [
            "term": term,
            "canonical_form": canonical.isEmpty ? term : canonical,
        ]
        let savedTerm = term  // capture before mutate clears fields
        mutate(path: "api/broker/memory/vocab", payload: payload, onSuccess: { [weak self] _ in
            self?.vocabTerm = ""
            self?.vocabCanonical = ""
            self?.refreshCategory("vocab")
            self?.flashSuccess("Saved \u{201C}\(savedTerm)\u{201D}")
        })
    }

    func removeVocab(term: String) {
        guard !Self.isProtectedVocabTerm(term) else { return }
        mutate(path: "api/broker/memory/vocab/remove",
               payload: ["term": term],
               onSuccess: { [weak self] _ in self?.refreshCategory("vocab") })
    }

    /// Remove old vocab entry and, once the remove completes, add the current
    /// vocabTerm/vocabCanonical values.  This serialises the two operations so
    /// the server never sees an upsert before the old row is gone — which would
    /// otherwise produce a spurious ``vocab_conflict`` error when only the
    /// canonical form changed.
    func removeVocabThenAdd(oldTerm: String) {
        guard !Self.isProtectedVocabTerm(oldTerm) else { return }
        isBusy = true
        statusMessage = ""
        JunoBroker.postJSON(
            path: "api/broker/memory/vocab/remove",
            payload: ["term": oldTerm]
        ) { [weak self] _ in
            // Proceed with the add regardless of whether the remove found a row
            // (the old term may already have been absent on a retry).
            self?.addVocab()
        }
    }

    // MARK: Replacement

    func addReplacement() {
        let trig = replacementTrigger.trimmingCharacters(in: .whitespaces)
        let repl = replacementText
        guard !trig.isEmpty, !repl.isEmpty else {
            statusMessage = "Trigger and replacement are required"
            return
        }
        let payload: [String: Any] = [
            "trigger": trig,
            "replacement": repl,
            "scope": replacementScope.isEmpty ? "global" : replacementScope,
        ]
        let savedTrig = trig
        mutate(path: "api/broker/memory/replacement", payload: payload, onSuccess: { [weak self] _ in
            self?.replacementTrigger = ""
            self?.replacementText = ""
            self?.refreshCategory("replacement")
            self?.flashSuccess("Saved replacement \u{201C}\(savedTrig)\u{201D}")
        })
    }

    func removeReplacement(trigger: String, scope: String) {
        mutate(path: "api/broker/memory/replacement/remove",
               payload: ["trigger": trigger, "scope": scope],
               onSuccess: { [weak self] _ in self?.refreshCategory("replacement") })
    }

    // MARK: Snippet

    func addSnippet() {
        let trig = snippetTrigger.trimmingCharacters(in: .whitespaces)
        let body = snippetBody
        guard !trig.isEmpty, !body.isEmpty else {
            statusMessage = "Shortcut phrase and expansion are required"
            return
        }
        let payload: [String: Any] = [
            "trigger": trig,
            "body": body,
            "scope": snippetScope.isEmpty ? "global" : snippetScope,
        ]
        let savedTrig = trig
        mutate(path: "api/broker/memory/snippet", payload: payload, onSuccess: { [weak self] _ in
            self?.snippetTrigger = ""
            self?.snippetBody = ""
            self?.snippetScope = "global"
            self?.refreshCategory("snippet")
            self?.flashSuccess("Saved snippet \u{201C}\(savedTrig)\u{201D}")
        })
    }

    func removeSnippet(trigger: String, scope: String) {
        mutate(path: "api/broker/memory/snippet/remove",
               payload: ["trigger": trigger, "scope": scope],
               onSuccess: { [weak self] _ in self?.refreshCategory("snippet") })
    }

    // MARK: Correction

    func removeCorrection(observed: String, corrected: String?) {
        var payload: [String: Any] = ["observed": observed]
        if let corrected, !corrected.isEmpty {
            payload["corrected"] = corrected
        }
        mutate(path: "api/broker/memory/correction/remove",
               payload: payload,
               onSuccess: { [weak self] _ in self?.refreshCategory("correction") })
    }

    // MARK: Internal

    /// Runs an upsert/remove POST and surfaces errors as
    /// `statusMessage`. `onSuccess` is called only when the server
    /// reports `ok=true`, so the UI doesn't clear the form fields
    /// until we know the mutation actually stuck.
    private func mutate(
        path: String,
        payload: [String: Any],
        onSuccess: @escaping ([String: Any]) -> Void
    ) {
        isBusy = true
        statusMessage = ""
        statusIsSuccess = false
        JunoBroker.postJSON(path: path, payload: payload) { [weak self] obj in
            guard let self else { return }
            self.isBusy = false
            let ok = (obj["ok"] as? Bool) ?? false
            if ok {
                onSuccess(obj)
            } else {
                let err = (obj["error"] as? String) ?? "request failed"
                self.statusMessage = err
                self.statusIsSuccess = false
            }
        }
    }
}

// MARK: - SwiftUI views

struct MemoryManagementView: View {
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @StateObject private var store = MemoryStoreViewModel()
    @State private var selected: Category = .vocab
    @State private var hoveredCategory: Category? = nil
    @State private var selectedEntryKey: String? = nil
    @State private var showAddSheet: Bool = false
    @State private var searchText: String = ""
    @Environment(\.colorScheme) private var scheme

    /// Order: most common user tasks first (matches product IA plan).
    enum Category: String, CaseIterable, Identifiable {
        case vocab, correction, snippet, replacement
        var id: String { rawValue }
        var label: String {
            switch self {
            case .vocab:       return "Vocabulary"
            case .correction:  return "Corrections"
            case .snippet:     return "Snippets"
            case .replacement: return "Replacements"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            JunoPageHeader(
                eyebrow: "Snippets & Memory",
                title: "Teach Juno your words",
                subtitle: "Names, snippets, replacements, styles, and learned fixes stay on this Mac."
            )
                .padding(.horizontal, JunoTheme.PageInsets.detail)
                .padding(.top, JunoTheme.PageInsets.rail)

            Divider().padding(.top, JunoTheme.PageInsets.sectionGap)

            HStack(spacing: 0) {
                // Category rail — custom rows match main sidebar (rounded selection, not `List` rectangle).
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: 2) {
                        ForEach(Category.allCases) { c in
                            memoryCategoryRow(c)
                        }
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 8)
                }
                .junoSplitPanePadding()
                .junoSubpaneSurface()
                .frame(
                    minWidth: JunoTheme.SplitColumns.memoryCategoryRailMin,
                    idealWidth: JunoTheme.SplitColumns.memoryCategoryRailIdeal,
                    maxWidth: JunoTheme.SplitColumns.memoryCategoryRailMax
                )

                Divider()

                // Content: list + detail editor.
                MemoryCategoryDetail(
                    store: store,
                    category: selected,
                    selectedEntryKey: $selectedEntryKey,
                    showAddSheet: $showAddSheet,
                    searchText: $searchText
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            Divider()
            HStack(spacing: 8) {
                if store.isBusy { ProgressView().controlSize(.small) }
                if !store.statusMessage.isEmpty {
                    if store.statusIsSuccess {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(JunoDesignTokens.meadow)
                    }
                    Text(store.statusMessage)
                        .font(.system(size: 11, design: .rounded))
                        .foregroundStyle(
                            store.statusIsSuccess
                                ? JunoDesignTokens.meadow
                                : JunoTheme.secondaryText(scheme)
                        )
                        .lineLimit(1)
                }
                Spacer()
            }
            .padding(10)
            .animation(.easeOut(duration: 0.18), value: store.statusMessage)
        }
        .frame(minHeight: 520)
        .onAppear {
            store.refreshAll()
            consumePendingMemoryNavigation()
        }
        .onChange(of: windowNav.pendingMemoryVocabPrefill) { _ in
            consumePendingMemoryNavigation()
        }
        .onChange(of: windowNav.pendingMemoryCategoryRaw) { _ in
            consumePendingMemoryNavigation()
        }
    }

    private func consumePendingMemoryNavigation() {
        let rawOpt = windowNav.pendingMemoryCategoryRaw
        let preOpt = windowNav.pendingMemoryVocabPrefill
        windowNav.pendingMemoryCategoryRaw = nil
        windowNav.pendingMemoryVocabPrefill = nil
        if let raw = rawOpt?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty,
           let cat = Category(rawValue: raw) {
            selected = cat
        }
        if let pre = preOpt?.trimmingCharacters(in: .whitespacesAndNewlines), !pre.isEmpty {
            // Vocabulary terms are atomic strings — names, acronyms,
            // 1-3 word jargon. The Save Phrase popover used to forward
            // the entire transcript here, which produced an Add
            // Vocabulary sheet with a sentence in the Term field.
            // Reject anything that doesn't look like a term so the
            // dialog stays clean.
            let words = pre.split(separator: " ").count
            if pre.count <= 40 && words <= 3 {
                store.vocabTerm = pre
            }
            showAddSheet = true
        }
    }

    private func categoryIcon(_ category: Category) -> String {
        switch category {
        case .vocab:       return "textformat.abc"
        case .correction:  return "arrow.triangle.2.circlepath"
        case .snippet:     return "doc.plaintext"
        case .replacement: return "arrow.right"
        }
    }

    private func memoryCategoryRow(_ item: Category) -> some View {
        let isSelected = selected == item
        let isHovered = hoveredCategory == item
        return Button {
            selected = item
        } label: {
            HStack(spacing: 10) {
                Image(systemName: categoryIcon(item))
                    .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 18, alignment: .center)
                Text(item.label)
                    .font(.system(size: 12, weight: isSelected ? .semibold : .regular, design: .rounded))
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                Spacer(minLength: 0)
            }
            .foregroundStyle(
                isSelected
                    ? JunoDesignTokens.accent
                    : (isHovered ? JunoTheme.primaryText(scheme) : JunoTheme.secondaryText(scheme))
            )
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(
                        isSelected
                            ? JunoDesignTokens.accent.opacity(0.12)
                            : (isHovered ? Color.white.opacity(0.05) : Color.clear)
                    )
            )
        }
        .buttonStyle(.plain)
        .focusable(false)
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityLabel(item.label)
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                hoveredCategory = hovering ? item : nil
            }
        }
    }
}

private struct MemoryCategoryDetail: View {
    @ObservedObject var store: MemoryStoreViewModel
    let category: MemoryManagementView.Category
    @Binding var selectedEntryKey: String?
    @Binding var showAddSheet: Bool
    @Binding var searchText: String
    @Environment(\.colorScheme) private var scheme

    private var title: String { category.label }
    private var help: String {
        switch category {
        case .vocab:
            return "Add names and terms you want Juno to recognize reliably."
        case .correction:
            return "Corrections are learned automatically from what you fix after a paste. Remove any that are wrong."
        case .snippet:
            return "Snippets expand a short trigger into longer text."
        case .replacement:
            return "Replacements swap a phrase for fixed text."
        }
    }

    private var canAdd: Bool { category != .correction }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .center, spacing: 10) {
                    Text(title)
                        .font(.system(size: 17, weight: .semibold, design: .rounded))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Spacer(minLength: 0)
                    HStack(spacing: 8) {
                        Button {
                            store.refreshCategory(category.rawValue)
                        } label: {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        .junoSecondaryActionButton()
                        if canAdd {
                            Button {
                                prepareAddSheetDefaults()
                                showAddSheet = true
                            } label: {
                                Label("Add", systemImage: "plus")
                            }
                            .junoPrimaryActionButton()
                        }
                    }
                }
                Text(help)
                    .font(.system(size: 12, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                JunoInlineSearchField(
                    prompt: "Search \(category.label.lowercased())…",
                    text: $searchText
                )
                .frame(maxWidth: .infinity)
            }
            // Flat insets, no card-on-card chrome. The previous
            // .junoSubpaneSurface() drew a second rounded slab inside
            // the page's already-carded layout, so the header read as
            // floating above the list/detail split. Plain padding lets
            // the page chrome carry the visual weight.
            .padding(.horizontal, JunoTheme.PageInsets.detail)
            .padding(.vertical, JunoTheme.PageInsets.rail)

            Divider()

            switch category {
            case .vocab:
                MemoryListDetail(
                    entries: filteredVocab,
                    selectedKey: $selectedEntryKey,
                    emptyTitle: "No vocabulary yet",
                    emptyMessage: "Add a name or term you use often. Juno will prefer it when recognizing your speech.",
                    keyForEntry: { vocabKey($0) },
                    row: { entry in
                        let rawTerm = str(entry["term"])
                        let isJuno = MemoryStoreViewModel.isProtectedVocabTerm(rawTerm)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(isJuno ? "Juno" : rawTerm)
                                .font(.system(.subheadline, design: .rounded).weight(.semibold))
                            if let canonical = entry["canonical_form"] as? String,
                               !canonical.isEmpty,
                               canonical != str(entry["term"]),
                               !isJuno {
                                Text(canonical)
                                    .font(.caption2)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                            }
                            if let source = entry["source"] as? String, !source.isEmpty, !isJuno {
                                Text(source.replacingOccurrences(of: "_", with: " ").uppercased())
                                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.82))
                            }
                        }
                    },
                    detail: { entry in
                        VocabEditorCard(store: store, entry: entry)
                    },
                    onDelete: { entry in
                        store.removeVocab(term: str(entry["term"]))
                    },
                    canDelete: { entry in
                        !MemoryStoreViewModel.isProtectedVocabTerm(str(entry["term"]))
                    }
                )
            case .snippet:
                MemoryListDetail(
                    entries: filteredSnippets,
                    selectedKey: $selectedEntryKey,
                    emptyTitle: "No snippets yet",
                    emptyMessage: "Create expansions like \"signoff\" → your multi‑line signature.",
                    keyForEntry: { snippetKey($0) },
                    row: { entry in
                        // Show trigger on top, then a one-line preview of
                        // the expansion body so the user can tell rows
                        // apart at a glance without clicking. The scope
                        // moves to a small chip on the trigger row.
                        VStack(alignment: .leading, spacing: 3) {
                            HStack(alignment: .center, spacing: 6) {
                                Text(str(entry["trigger"]))
                                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                                    .lineLimit(1)
                                if let label = scopeChipLabel(for: str(entry["scope"], fallback: "global")) {
                                    Text(label)
                                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                        .tracking(0.4)
                                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 1)
                                        .background(
                                            RoundedRectangle(cornerRadius: 4, style: .continuous)
                                                .fill(JunoTheme.secondaryText(scheme).opacity(0.10))
                                        )
                                }
                                Spacer(minLength: 0)
                            }
                            Text(snippetBodyPreview(str(entry["body"])))
                                .font(.caption2)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .lineLimit(1)
                                .truncationMode(.tail)
                        }
                    },
                    detail: { entry in
                        SnippetEditorCard(store: store, entry: entry)
                    },
                    onDelete: { entry in
                        store.removeSnippet(trigger: str(entry["trigger"]), scope: str(entry["scope"], fallback: "global"))
                    }
                )
            case .replacement:
                MemoryListDetail(
                    entries: filteredReplacements,
                    selectedKey: $selectedEntryKey,
                    emptyTitle: "No replacements yet",
                    emptyMessage: "Replace a phrase with fixed text, like \"my email\" → your address.",
                    keyForEntry: { replacementKey($0) },
                    row: { entry in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(str(entry["trigger"]))
                                .font(.system(.subheadline, design: .rounded).weight(.semibold))
                            Text("→ \(str(entry["replacement"]))")
                                .font(.caption2)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .lineLimit(1)
                        }
                    },
                    detail: { entry in
                        ReplacementEditorCard(store: store, entry: entry)
                    },
                    onDelete: { entry in
                        store.removeReplacement(trigger: str(entry["trigger"]), scope: str(entry["scope"], fallback: "global"))
                    }
                )
            case .correction:
                MemoryListDetail(
                    entries: filteredCorrections,
                    selectedKey: $selectedEntryKey,
                    emptyTitle: "No corrections yet",
                    emptyMessage: "After you fix a pasted line, Juno learns from it. Your learned fixes will show up here.",
                    keyForEntry: { correctionKey($0) },
                    row: { entry in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(str(entry["observed"]))
                                .font(.system(.subheadline, design: .rounded).weight(.semibold))
                            Text("→ \(str(entry["corrected"]))")
                                .font(.caption2)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .lineLimit(1)
                        }
                    },
                    detail: { entry in
                        MemoryDetailCard(
                            title: str(entry["observed"]),
                            subtitle: "→ \(str(entry["corrected"]))",
                            primaryActionTitle: "Remove",
                            primaryRole: .destructive,
                            primaryAction: {
                                store.removeCorrection(
                                    observed: str(entry["observed"]),
                                    corrected: entry["corrected"] as? String
                                )
                            }
                        )
                    },
                    onDelete: { entry in
                        store.removeCorrection(observed: str(entry["observed"]), corrected: entry["corrected"] as? String)
                    }
                )
            }
        }
        .sheet(isPresented: $showAddSheet) {
            AddMemoryItemSheet(store: store, category: category, onDone: { newKey in
                showAddSheet = false
                // Auto-select the just-added row so the user lands in
                // its detail editor instead of the "Select an item"
                // empty state. The refresh callback below repopulates
                // the list; selectedEntryKey resolves to the right row
                // once that completes. Without this, the user is
                // greeted by their just-created item *not* being
                // selected — which read as "nothing happened".
                if let newKey {
                    selectedEntryKey = newKey
                }
                store.refreshCategory(category.rawValue)
            })
        }
        .onChange(of: category) { _ in
            // Reset per-category selection and search, and refresh the list.
            selectedEntryKey = nil
            searchText = ""
            store.statusMessage = ""
            store.refreshCategory(category.rawValue)
        }
    }

    private func prepareAddSheetDefaults() {
        switch category {
        case .vocab:
            if store.vocabCanonical.isEmpty { store.vocabCanonical = "" }
        case .replacement:
            if store.replacementScope.isEmpty { store.replacementScope = "global" }
        case .snippet:
            store.snippetTrigger = ""
            store.snippetBody = ""
            store.snippetScope = "global"
        case .correction:
            break
        }
    }

    private func contains(_ entry: [String: Any], keys: [String]) -> Bool {
        guard !searchText.isEmpty else { return true }
        let q = searchText.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return true }
        return keys.contains { k in
            (entry[k] as? String)?.lowercased().contains(q) ?? false
        }
    }

    private var filteredVocab: [[String: Any]] {
        store.vocab.filter { contains($0, keys: ["term", "canonical_form"]) }
    }
    private var filteredSnippets: [[String: Any]] {
        store.snippets.filter { contains($0, keys: ["trigger", "body", "scope"]) }
    }
    private var filteredReplacements: [[String: Any]] {
        store.replacements.filter { contains($0, keys: ["trigger", "replacement", "scope"]) }
    }
    private var filteredCorrections: [[String: Any]] {
        store.corrections.filter { contains($0, keys: ["observed", "corrected"]) }
    }

    private func vocabKey(_ entry: [String: Any]) -> String {
        "vocab:\(str(entry["term"]))"
    }
    private func snippetKey(_ entry: [String: Any]) -> String {
        "snippet:\(str(entry["trigger"])):\(str(entry["scope"], fallback: "global"))"
    }
    private func replacementKey(_ entry: [String: Any]) -> String {
        "replacement:\(str(entry["trigger"])):\(str(entry["scope"], fallback: "global"))"
    }
    private func correctionKey(_ entry: [String: Any]) -> String {
        "correction:\(str(entry["observed"])):\(str(entry["corrected"]))"
    }

    private func canonicalSubtitle(term: String, canonical: String?) -> String {
        let c = (canonical ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !c.isEmpty, c != term else { return "" }
        return "Canonical: \(c)"
    }

    /// One-line preview of a snippet's expansion body for the list row.
    /// Collapses line breaks to spaces, trims, and caps at ~70 chars so
    /// the row stays single-line even for paragraph-sized expansions.
    private func snippetBodyPreview(_ body: String) -> String {
        let collapsed = body
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if collapsed.isEmpty {
            return "(empty body)"
        }
        if collapsed.count <= 70 {
            return collapsed
        }
        return String(collapsed.prefix(67)).trimmingCharacters(in: .whitespaces) + "…"
    }

    /// Human-readable chip label for a wire scope value. Returns nil for
    /// the default "global" so we don't clutter rows with the obvious
    /// case.
    private func scopeChipLabel(for wire: String) -> String? {
        let key = wire.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch key {
        case "", "global":      return nil
        case "messaging":       return "MESSAGING"
        case "email":           return "EMAIL"
        case "docs":            return "DOCS"
        case "forms":           return "FORMS"
        case "code":            return "CODE"
        case "terminal":        return "TERMINAL"
        default:                return key.uppercased()
        }
    }
}

struct MemoryListRowModel: Identifiable {
    let id: String
    let key: String
    let entry: [String: Any]
}

enum MemoryListRows {
    static func make(
        entries: [[String: Any]],
        keyForEntry: ([String: Any]) -> String
    ) -> [MemoryListRowModel] {
        var seen: [String: Int] = [:]
        return entries.map { entry in
            let key = keyForEntry(entry)
            let count = (seen[key] ?? 0) + 1
            seen[key] = count
            return MemoryListRowModel(
                id: count == 1 ? key : "\(key)#\(count)",
                key: key,
                entry: entry
            )
        }
    }
}

private struct MemoryListDetail<RowContent: View, DetailContent: View>: View {
    let entries: [[String: Any]]
    @Binding var selectedKey: String?
    let emptyTitle: String
    let emptyMessage: String
    let keyForEntry: ([String: Any]) -> String
    @ViewBuilder let row: ([String: Any]) -> RowContent
    @ViewBuilder let detail: ([String: Any]) -> DetailContent
    var onDelete: (([String: Any]) -> Void)? = nil
    var canDelete: (([String: Any]) -> Bool)? = nil
    @Environment(\.colorScheme) private var scheme

    private var rows: [MemoryListRowModel] {
        MemoryListRows.make(entries: entries, keyForEntry: keyForEntry)
    }

    private var selectedEntry: [String: Any]? {
        guard let k = selectedKey else { return nil }
        return rows.first(where: { $0.key == k })?.entry
    }

    var body: some View {
        let rowModels = rows
        HStack(spacing: 0) {
            // Left: stable width (reuse secondary list column tokens — avoids nested `NavigationSplitView` jump).
            Group {
                if rowModels.isEmpty {
                    JunoChromeEmptyState(
                        title: emptyTitle,
                        message: emptyMessage,
                        symbol: "sparkles",
                        compact: true
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    // No `List(selection:)` — macOS paints a system
                    // accent layer on top of `.listRowBackground` when
                    // a List has a selection binding, so the highlight
                    // ended up as Juno's accent + macOS system blue
                    // stacked, with white text forced over both. Here
                    // each row is its own Button driving `selectedKey`
                    // and the row owns its own background + text
                    // colors. Plain List still supports swipeActions,
                    // so swipe-to-delete keeps working.
                    List {
                        ForEach(rowModels) { item in
                            let entry = item.entry
                            let key = item.key
                            let isSelected = selectedKey == key
                            let deletable = onDelete != nil && (canDelete?(entry) ?? true)
                            Button {
                                if selectedKey != key {
                                    selectedKey = key
                                }
                            } label: {
                                row(entry)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                                            .fill(isSelected
                                                  ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.10)
                                                  : Color.clear)
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                                            .strokeBorder(
                                                isSelected
                                                    ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.45 : 0.30)
                                                    : Color.clear,
                                                lineWidth: 0.7
                                            )
                                    )
                                    .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            }
                            .buttonStyle(.plain)
                            .junoNoFocusRing()
                            .listRowInsets(EdgeInsets(top: 1, leading: 6, bottom: 1, trailing: 6))
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                if deletable {
                                    Button(role: .destructive) {
                                        onDelete?(entry)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                            }
                        }
                    }
                    .junoSplitListChrome()
                    .environment(\.defaultMinListRowHeight, 34)
                }
            }
            // Tighter than the global `secondaryList*` tokens because
            // the memory list rows carry mostly short triggers
            // ("signoff", "QBR", "my email") with a one-line preview
            // beneath. The right detail pane is where the editing
            // happens; give it the room. We still allow ~260pt so a
            // long trigger like "back end api docs" can fit before
            // truncating.
            .frame(
                minWidth: 200,
                idealWidth: 220,
                maxWidth: 260,
                maxHeight: .infinity,
                alignment: .leading
            )

            Divider()

            Group {
                if let entry = selectedEntry {
                    ScrollView(.vertical, showsIndicators: false) {
                        detail(entry)
                            .junoDetailPagePadding()
                            // Cap the editor's content width so it
                            // doesn't sprawl across a wide window —
                            // long-form replacement/snippet text is
                            // unreadable at 1000pt+. The cap also
                            // stabilises the surrounding column so
                            // selecting a row no longer reflows the
                            // page (filled-state and empty-state
                            // settle to the same outer geometry).
                            // 640pt picked to comfortably fit the
                            // Expansion editor + scope picker + two
                            // action buttons in one column without
                            // crowding, while keeping line length
                            // readable.
                            .frame(maxWidth: 640, alignment: .topLeading)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                    .id(keyForEntry(entry))
                } else if entries.isEmpty {
                    Color.clear
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    JunoChromeEmptyState(
                        title: "Select an item",
                        message: "Choose a row on the left to view or edit details.",
                        symbol: "sidebar.left",
                        compact: true
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .junoDetailPagePadding()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .tint(JunoDesignTokens.accent)
    }
}

private struct MemoryDetailCard: View {
    let title: String
    let subtitle: String
    let primaryActionTitle: String
    let primaryRole: ButtonRole?
    let primaryAction: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.system(size: 17, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)
            if !subtitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(subtitle)
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider().opacity(0.3)
            HStack {
                Spacer()
                Button(primaryActionTitle, role: primaryRole) { primaryAction() }
                    .junoPrimaryActionButton()
            }
        }
        .junoPageCard()
    }
}

private enum JunoSnippetScopePreset: String, CaseIterable, Identifiable {
    case everywhere
    case messages
    case email
    case documents
    case forms
    case code
    case terminal
    case custom

    var id: String { rawValue }

    var label: String {
        switch self {
        case .everywhere: return "Everywhere"
        case .messages: return "Messages"
        case .email: return "Email"
        case .documents: return "Documents"
        case .forms: return "Forms"
        case .code: return "Code"
        case .terminal: return "Terminal"
        case .custom: return "Custom"
        }
    }

    var fixedWire: String? {
        switch self {
        case .everywhere: return "global"
        case .messages: return "messaging"
        case .email: return "email"
        case .documents: return "docs"
        case .forms: return "forms"
        case .code: return "code"
        case .terminal: return "terminal"
        case .custom: return nil
        }
    }

    static func preset(forScopeWire s: String) -> JunoSnippetScopePreset {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch t {
        case "", "global": return .everywhere
        case "messaging": return .messages
        case "email": return .email
        case "docs": return .documents
        case "forms": return .forms
        case "code": return .code
        case "terminal": return .terminal
        default: return .custom
        }
    }
}

private struct SnippetEditorCard: View {
    @ObservedObject var store: MemoryStoreViewModel
    let entry: [String: Any]?
    /// When adding from a sheet, skip the nested card chrome so the sheet reads as one surface.
    var flatChrome: Bool = false

    @State private var trigger: String = ""
    @State private var snippetScopePreset: JunoSnippetScopePreset = .everywhere
    @State private var customSnippetScope: String = ""
    @State private var snippetBodyText: String = ""
    @State private var dirty: Bool = false
    @Environment(\.colorScheme) private var scheme

    private var isAdding: Bool { entry == nil }

    private var resolvedWireScope: String {
        if snippetScopePreset == .custom {
            let t = customSnippetScope.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? "global" : t
        }
        return snippetScopePreset.fixedWire ?? "global"
    }

    var body: some View {
        let inner = VStack(alignment: .leading, spacing: 14) {
            Text("Snippet")
                .font(.system(.headline, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            if isAdding {
                Text("Say the shortcut on its own and Juno expands it. Works with single words like \"signoff\" or short phrases like \"sign off\".")
                    .font(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }

            // Vertical stack — trigger, then scope, then (optional) custom
            // scope, then expansion. The previous HStack(trigger | scope)
            // collapsed the trigger column to invisible at narrow detail-
            // pane widths and made the page jump every time the user
            // selected a different row. A single column wins at every
            // width without that jank.
            VStack(alignment: .leading, spacing: 6) {
                Text("Shortcut phrase").font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
                TextField("e.g. signoff", text: $trigger)
                    .textFieldStyle(.roundedBorder)
                    .focusEffectDisabled()
                    .onChange(of: trigger) { _ in
                        dirty = true
                        syncStoreIfAdding()
                    }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Where it applies").font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
                GenericGlyphPopoverPicker(
                    selection: Binding(
                        get: { snippetScopePreset.rawValue },
                        set: { newRaw in
                            let newValue = JunoSnippetScopePreset(rawValue: newRaw) ?? .everywhere
                            snippetScopePreset = newValue
                            dirty = true
                            if newValue != .custom {
                                customSnippetScope = ""
                            } else if customSnippetScope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                customSnippetScope = ""
                            }
                            syncStoreIfAdding()
                        }
                    ),
                    options: JunoSnippetScopePreset.allCases.map {
                        GenericGlyphPopoverPicker.Option(
                            value: $0.rawValue,
                            title: $0.label,
                            subtitle: "",
                            systemName: "scope"
                        )
                    }
                )
            }

            if snippetScopePreset == .custom {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Custom scope").font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
                    TextField("Scope key", text: $customSnippetScope)
                        .textFieldStyle(.roundedBorder)
                        .focusEffectDisabled()
                        .onChange(of: customSnippetScope) { _ in
                            dirty = true
                            syncStoreIfAdding()
                        }
                }
            }

            Text("Scoped snippets match that kind of app first; “Everywhere” still works as a fallback.")
                .font(.caption2)
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.9))
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 6) {
                Text("Expansion").font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $snippetBodyText)
                        .font(.system(.body, design: .rounded))
                        .focusEffectDisabled()
                        // 110pt minimum so a short snippet body doesn't
                        // open a 220pt void below the picker. The editor
                        // still grows with content — typing a long body
                        // expands it naturally — but a one-line snippet
                        // shows the editor at a reasonable single-line
                        // size instead of a wall of empty space.
                        .frame(minHeight: 110)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(JunoTheme.border(scheme).opacity(0.5), lineWidth: 0.5)
                        )
                        .onChange(of: snippetBodyText) { _ in
                            dirty = true
                            syncStoreIfAdding()
                        }
                    if snippetBodyText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Text("Text Juno inserts when you say the shortcut…")
                            .font(.system(.body, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.55))
                            .padding(.top, 10)
                            .padding(.leading, 6)
                            .allowsHitTesting(false)
                    }
                }
            }

            if !isAdding {
                HStack(spacing: 10) {
                    Button("Remove", role: .destructive) {
                        guard let entry else { return }
                        store.removeSnippet(trigger: str(entry["trigger"]), scope: str(entry["scope"], fallback: "global"))
                    }
                    .junoSecondaryActionButton()
                    Spacer()
                    Button("Save") {
                        let oldTrigger = entry.map { str($0["trigger"]) } ?? ""
                        let oldScope = entry.map { str($0["scope"], fallback: "global") } ?? "global"
                        let newTrigger = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
                        let newScope = resolvedWireScope
                        store.snippetTrigger = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
                        store.snippetScope = newScope
                        store.snippetBody = snippetBodyText
                        if !oldTrigger.isEmpty, (oldTrigger != newTrigger || oldScope != newScope) {
                            store.removeSnippet(trigger: oldTrigger, scope: oldScope)
                        }
                        store.addSnippet()
                        dirty = false
                    }
                    .junoPrimaryActionButton()
                    .disabled(
                        trigger.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || snippetBodyText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || !dirty
                    )
                }
            }
        }

        Group {
            if isAdding && flatChrome {
                inner
            } else {
                inner.padding(16).premiumCard()
            }
        }
        .onAppear {
            if let entry {
                trigger = str(entry["trigger"])
                let wire = str(entry["scope"], fallback: "global")
                snippetScopePreset = JunoSnippetScopePreset.preset(forScopeWire: wire)
                customSnippetScope = snippetScopePreset == .custom ? wire : ""
                snippetBodyText = str(entry["body"])
                dirty = false
            } else {
                trigger = store.snippetTrigger
                snippetBodyText = store.snippetBody
                let wire = store.snippetScope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? "global" : store.snippetScope
                snippetScopePreset = JunoSnippetScopePreset.preset(forScopeWire: wire)
                customSnippetScope = snippetScopePreset == .custom ? wire : ""
                dirty = true
                syncStoreIfAdding()
            }
        }
    }

    private func syncStoreIfAdding() {
        guard isAdding else { return }
        store.snippetTrigger = trigger
        store.snippetBody = snippetBodyText
        store.snippetScope = resolvedWireScope
    }
}

private struct AddMemoryItemSheet: View {
    @ObservedObject var store: MemoryStoreViewModel
    let category: MemoryManagementView.Category
    /// Returns the key the parent should auto-select on the refreshed
    /// list, computed from form state *before* ``submit()`` clears the
    /// fields. ``nil`` for the correction category (read-only) or when
    /// the form somehow has no trigger.
    let onDone: (String?) -> Void
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Add \(category.label)")
                        .font(.system(.title2, design: .rounded).weight(.semibold))
                    Text(sheetHelp)
                        .font(.callout)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)

                    Divider().opacity(0.35)

                    Group {
                        switch category {
                        case .vocab:
                            VocabEditorCard(store: store, entry: nil, flatChrome: true)
                        case .snippet:
                            SnippetEditorCard(store: store, entry: nil, flatChrome: true)
                        case .replacement:
                            ReplacementEditorCard(store: store, entry: nil)
                        case .correction:
                            EmptyView()
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 560)

            Divider().opacity(0.35)
                .padding(.top, 12)

            HStack {
                Button("Cancel") { dismiss() }
                    .junoSecondaryActionButton()
                Spacer()
                Button("Add") {
                    // Capture the would-be selection key *before* submit
                    // runs — the store viewmodel's mutate callback clears
                    // the form fields on success, so by the time onDone
                    // fires we'd have nothing left to derive the key from.
                    let pendingKey = pendingSelectionKey
                    submit()
                    onDone(pendingKey)
                    dismiss()
                }
                .junoPrimaryActionButton()
                .disabled(!canSubmit)
            }
            .padding(.top, 12)
        }
        .padding(18)
        .frame(minWidth: 640)
    }

    private var sheetHelp: String {
        switch category {
        case .vocab:
            return "Teach Juno how you say a name or term, and optionally how it should be spelled. Great for people, products, acronyms, and jargon."
        case .snippet:
            return "Snippets expand a shortcut into longer text. The shortcut can be a single word like \"brb\" or a short phrase like \"sign off\". Pick where each shortcut applies, or use Everywhere."
        case .replacement: return "Replacements are one-to-one swaps, like \"my address\" → \"123 Main St\"."
        case .correction:  return ""
        }
    }

    private var canSubmit: Bool {
        switch category {
        case .vocab:
            let term = store.vocabTerm.trimmingCharacters(in: .whitespacesAndNewlines)
            let canonical = store.vocabCanonical.trimmingCharacters(in: .whitespacesAndNewlines)
            return MemoryStoreViewModel.learnedVocabTermAllowed(term)
                && MemoryStoreViewModel.learnedVocabTermAllowed(canonical.isEmpty ? term : canonical)
        case .snippet:
            return !store.snippetTrigger.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !store.snippetBody.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .replacement:
            return !store.replacementTrigger.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !store.replacementText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .correction:
            return false
        }
    }

    private func submit() {
        switch category {
        case .vocab:
            store.addVocab()
        case .snippet:
            if store.snippetScope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                store.snippetScope = "global"
            }
            store.addSnippet()
        case .replacement:
            if store.replacementScope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                store.replacementScope = "global"
            }
            store.addReplacement()
        case .correction:
            break
        }
    }

    /// Compute the list key the just-added row will appear under, so the
    /// parent can auto-select it once the refresh callback brings it
    /// back from the server. Key format mirrors the per-category
    /// ``MemoryCategoryDetail.{vocab,snippet,...}Key(_:)`` functions —
    /// any divergence here would break auto-select silently.
    private var pendingSelectionKey: String? {
        let scopeOrGlobal: (String) -> String = { s in
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? "global" : t
        }
        switch category {
        case .vocab:
            let t = store.vocabTerm.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : "vocab:\(t)"
        case .snippet:
            let t = store.snippetTrigger.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : "snippet:\(t):\(scopeOrGlobal(store.snippetScope))"
        case .replacement:
            let t = store.replacementTrigger.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : "replacement:\(t):\(scopeOrGlobal(store.replacementScope))"
        case .correction:
            return nil
        }
    }
}

// MARK: - Premium editors

private struct VocabEditorCard: View {
    @ObservedObject var store: MemoryStoreViewModel
    let entry: [String: Any]?
    /// When adding from a sheet, omit the nested card chrome so the sheet reads as one surface.
    var flatChrome: Bool = false

    @State private var term: String = ""
    @State private var canonical: String = ""
    @State private var dirty: Bool = false
    @Environment(\.colorScheme) private var scheme

    private var isProtectedEntry: Bool {
        guard let entry else { return false }
        return MemoryStoreViewModel.isProtectedVocabTerm(str(entry["term"]))
    }

    var body: some View {
        Group {
            if entry == nil && flatChrome {
                editorForm
            } else {
                editorForm.padding(16).premiumCard()
            }
        }
        .onAppear {
            if let entry {
                term = str(entry["term"])
                canonical = str(entry["canonical_form"])
                dirty = false
            } else {
                term = store.vocabTerm
                canonical = store.vocabCanonical
                dirty = true
            }
        }
        .onChange(of: term) { newVal in
            guard entry == nil else { return }
            store.vocabTerm = newVal
        }
        .onChange(of: canonical) { newVal in
            guard entry == nil else { return }
            store.vocabCanonical = newVal
        }
    }

    @ViewBuilder
    private var editorForm: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Vocabulary")
                .font(.system(.headline, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            if !isProtectedEntry {
                Text("Add names and terms you want Juno to recognize reliably.")
                    .font(.callout)
                    .foregroundStyle(JunoTheme.sheetBodySecondary(scheme))
            }

            if isProtectedEntry {
                VStack(alignment: .leading, spacing: 8) {
                    label("Term")
                    Text("Juno")
                        .font(.system(.body, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Label("Built-in", systemImage: "lock.fill")
                        .font(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    label("Term")
                    TextField("As you say it", text: $term)
                        .textFieldStyle(.roundedBorder)
                        .focusEffectDisabled()
                        .onChange(of: term) { _ in dirty = true }
                }

                VStack(alignment: .leading, spacing: 6) {
                    label("Canonical form (optional)")
                    TextField("Preferred spelling", text: $canonical)
                        .textFieldStyle(.roundedBorder)
                        .focusEffectDisabled()
                        .onChange(of: canonical) { _ in dirty = true }
                }
            }

            if let entry {
                HStack(spacing: 10) {
                    let entryTerm = str(entry["term"])
                    let isProtected = MemoryStoreViewModel.isProtectedVocabTerm(entryTerm)
                    if !isProtected {
                        Button("Remove", role: .destructive) {
                            store.removeVocab(term: entryTerm)
                        }
                        .junoSecondaryActionButton()
                    }
                    Spacer()
                    if !isProtectedEntry {
                        Button("Save") {
                            let oldTerm = str(entry["term"])
                            let newTerm = term.trimmingCharacters(in: .whitespacesAndNewlines)
                            store.vocabTerm = newTerm
                            store.vocabCanonical = canonical.trimmingCharacters(in: .whitespacesAndNewlines)
                            // Always remove then re-add so canonical-form edits
                            // don't produce a server-side vocab_conflict error.
                            if !oldTerm.isEmpty {
                                store.removeVocabThenAdd(oldTerm: oldTerm)
                            } else {
                                store.addVocab()
                            }
                            dirty = false
                        }
                        .junoPrimaryActionButton()
                        .disabled(
                            !MemoryStoreViewModel.learnedVocabTermAllowed(term.trimmingCharacters(in: .whitespacesAndNewlines))
                                || !MemoryStoreViewModel.learnedVocabTermAllowed(
                                    canonical.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                        ? term.trimmingCharacters(in: .whitespacesAndNewlines)
                                        : canonical.trimmingCharacters(in: .whitespacesAndNewlines)
                                )
                                || !dirty
                        )
                    }
                }
            }
        }
    }

    private func label(_ s: String) -> some View {
        Text(s).font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
    }
}

private enum JunoScopePreset: String, CaseIterable, Identifiable {
    case global = "global"
    case custom = "custom"
    var id: String { rawValue }
    var label: String {
        switch self {
        case .global: return "All apps"
        case .custom: return "Specific app"
        }
    }
}

private struct ReplacementEditorCard: View {
    @ObservedObject var store: MemoryStoreViewModel
    let entry: [String: Any]?
    @State private var trigger: String = ""
    @State private var scope: String = "global"
    @State private var scopePreset: JunoScopePreset = .global
    @State private var replacement: String = ""
    @State private var dirty: Bool = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Replacement")
                .font(.system(.headline, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Text("Replace a phrase with fixed text. Great for emails, signatures, and common replies.")
                .font(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))

            // Single column. The previous HStack collapsed the trigger
            // field at narrow detail-pane widths; vertical stack works
            // at every width without reflow.
            VStack(alignment: .leading, spacing: 6) {
                label("Trigger")
                TextField("e.g. my email", text: $trigger)
                    .textFieldStyle(.roundedBorder)
                    .focusEffectDisabled()
                    .onChange(of: trigger) { _ in
                        dirty = true
                        syncReplacementAddToStore()
                    }
            }

            VStack(alignment: .leading, spacing: 6) {
                label("Scope")
                GenericGlyphPopoverPicker(
                    selection: Binding(
                        get: { scopePreset.rawValue },
                        set: { newRaw in
                            let newValue = JunoScopePreset(rawValue: newRaw) ?? .global
                            scopePreset = newValue
                            dirty = true
                            switch newValue {
                            case .global: scope = "global"
                            case .custom: break
                            }
                            syncReplacementAddToStore()
                        }
                    ),
                    options: JunoScopePreset.allCases.map {
                        GenericGlyphPopoverPicker.Option(
                            value: $0.rawValue,
                            title: $0.label,
                            subtitle: "",
                            systemName: "app.badge"
                        )
                    }
                )
            }

            if scopePreset == .custom {
                VStack(alignment: .leading, spacing: 6) {
                    label("App name")
                    TextField("e.g. mail, messages, slack", text: $scope)
                        .textFieldStyle(.roundedBorder)
                        .focusEffectDisabled()
                        .onChange(of: scope) { _ in
                            dirty = true
                            syncReplacementAddToStore()
                        }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                label("Replacement text")
                TextEditor(text: $replacement)
                    .font(.system(.body, design: .rounded))
                    .focusEffectDisabled()
                    // 110pt minimum — same calibration as the snippet
                    // body editor, so a one-line replacement doesn't
                    // open a 220pt void below the picker.
                    .frame(minHeight: 110)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(JunoTheme.border(scheme).opacity(0.5), lineWidth: 0.5)
                    )
                    .onChange(of: replacement) { _ in
                        dirty = true
                        syncReplacementAddToStore()
                    }
            }

            if entry != nil {
                HStack(spacing: 10) {
                    if let entry {
                        Button("Remove", role: .destructive) {
                            store.removeReplacement(
                                trigger: str(entry["trigger"]),
                                scope: str(entry["scope"], fallback: "global")
                            )
                        }
                        .junoSecondaryActionButton()
                    }
                    Spacer()
                    Button("Save") {
                        let oldTrigger = entry.map { str($0["trigger"]) } ?? ""
                        let oldScope = entry.map { str($0["scope"], fallback: "global") } ?? "global"
                        let newTrigger = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
                        let newScope = scope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "global" : scope
                        store.replacementTrigger = newTrigger
                        store.replacementScope = newScope
                        store.replacementText = replacement
                        if !oldTrigger.isEmpty, (oldTrigger != newTrigger || oldScope != newScope) {
                            store.removeReplacement(trigger: oldTrigger, scope: oldScope)
                        }
                        store.addReplacement()
                        dirty = false
                    }
                    .junoPrimaryActionButton()
                    .disabled(trigger.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || replacement.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || !dirty)
                }
            }
        }
        .padding(16)
        .premiumCard()
        .onAppear {
            if let entry {
                trigger = str(entry["trigger"])
                replacement = str(entry["replacement"])
                scope = str(entry["scope"], fallback: "global")
                scopePreset = (scope == "global") ? .global : .custom
                dirty = false
            } else {
                trigger = store.replacementTrigger
                replacement = store.replacementText
                scope = store.replacementScope.isEmpty ? "global" : store.replacementScope
                scopePreset = (scope == "global") ? .global : .custom
                dirty = true
                syncReplacementAddToStore()
            }
        }
    }

    private func resolvedReplacementWireScope() -> String {
        scope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "global" : scope
    }

    private func syncReplacementAddToStore() {
        guard entry == nil else { return }
        store.replacementTrigger = trigger
        store.replacementText = replacement
        store.replacementScope = resolvedReplacementWireScope()
    }

    private func label(_ s: String) -> some View {
        Text(s).font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
    }
}

// Tiny helpers to pull optional strings out of broker JSON safely.
private func str(_ any: Any?, fallback: String = "") -> String {
    (any as? String) ?? fallback
}
