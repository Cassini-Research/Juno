import AppKit
import SwiftUI

struct JunoModesView: View {
    @Environment(\.colorScheme) private var scheme
    @StateObject private var model = ModesViewModel()

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 10) {
                JunoSplitColumnTitleRow(title: "Styles", trailing: {
                    Button {
                        model.showCreateSheet = true
                    } label: {
                        Label("New style", systemImage: "plus")
                    }
                    .junoPrimaryActionButton()
                })

                ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Built-in".uppercased())
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .tracking(1.0)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .padding(.horizontal, 2)
                        connectedModesSection {
                            ForEach(Array(model.builtinModeIds.enumerated()), id: \.element) { index, id in
                                modeRowButton(
                                    selection: .builtin(id),
                                    contextMenu: {
                                        Button("Activate") { model.activateBuiltin(id: id) }
                                    }
                                ) {
                                    ModeRow(
                                        scheme: scheme,
                                        title: JunoUserFacingCopy.builtinModeTitle(id: id),
                                        subtitle: model.builtinSubtitle(id: id),
                                        icon: "sparkles",
                                        active: model.isBuiltinActive(id: id),
                                        isSelected: model.selection == .builtin(id)
                                    )
                                }
                                if index < model.builtinModeIds.count - 1 {
                                    Divider().padding(.leading, 36)
                                }
                            }
                        }

                        Text("Custom".uppercased())
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .tracking(1.0)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .padding(.horizontal, 2)
                        if model.customModes.isEmpty {
                            connectedModesSection {
                                Text("No custom styles yet")
                                    .font(.system(size: 12, design: .rounded))
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 12)
                            }
                        } else {
                            connectedModesSection {
                                ForEach(Array(model.customModes.enumerated()), id: \.element.name) { index, m in
                                    modeRowButton(
                                        selection: .custom(m.name),
                                        contextMenu: {
                                            Button("Activate") { model.activateCustom(name: m.name) }
                                            Divider()
                                            Button("Delete", role: .destructive) { model.deleteCustom(name: m.name) }
                                        }
                                    ) {
                                        ModeRow(
                                            scheme: scheme,
                                            title: m.name,
                                            subtitle: m.description.isEmpty ? model.builtinSubtitle(id: m.baseMode) : m.description,
                                            icon: "slider.horizontal.3",
                                            active: model.isCustomActive(name: m.name),
                                            isSelected: model.selection == .custom(m.name)
                                        )
                                    }
                                    if index < model.customModes.count - 1 {
                                        Divider().padding(.leading, 36)
                                    }
                                }
                            }
                        }
                    }
                }

                activeModeBanner
            }
            .junoSplitPanePadding()
            .junoSubpaneSurface()
            .navigationSplitViewColumnWidth(
                min: JunoTheme.SplitColumns.primaryListMin,
                ideal: JunoTheme.SplitColumns.primaryListIdeal,
                max: JunoTheme.SplitColumns.primaryListMax
            )
        } detail: {
            detailPane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .tint(JunoDesignTokens.accent)
        .onAppear { model.refreshAll() }
        .sheet(isPresented: $model.showCreateSheet) {
            CustomModeEditorSheet(
                scheme: scheme,
                title: "New custom style",
                builtinModes: model.builtinModeIds,
                draft: model.makeNewDraft(),
                onSave: { draft, done in
                    model.upsertCustom(draft) { result in
                        done(result)
                    }
                }
            )
        }
    }

    @ViewBuilder
    private func connectedModesSection<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        VStack(spacing: 0) {
            content()
        }
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.64 : 0.14), lineWidth: 0.6)
        )
    }

    @ViewBuilder
    private func modeRowButton<Label: View>(
        selection: ModesViewModel.Selection,
        @ViewBuilder contextMenu: () -> some View,
        @ViewBuilder label: () -> Label
    ) -> some View {
        Button {
            model.selection = selection
        } label: {
            label()
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .focusable(false)
        .contextMenu { contextMenu() }
    }

    private var activeModeBanner: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(model.currentModeIsCustom ? JunoDesignTokens.accent : JunoDesignTokens.meadow)
                .frame(width: 7, height: 7)
            Text("Active: \(model.currentModeLabel)")
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.65 : 0.14), lineWidth: 0.6)
        )
    }

    @ViewBuilder
    private var detailPane: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: JunoTheme.PageInsets.sectionGap) {
                if let err = model.errorMessage, !err.isEmpty {
                    JunoInlineStatusBanner(
                        kind: .danger,
                        title: "Couldn’t update styles",
                        message: err,
                        systemImage: "exclamationmark.triangle"
                    ) {
                        Button("Dismiss") { model.dismissError() }
                            .junoTertiaryActionButton()
                    }
                }

                switch model.selection {
                case .builtin(let id):
                    BuiltinModeDetail(
                        scheme: scheme,
                        modeId: id,
                        title: JunoUserFacingCopy.builtinModeTitle(id: id),
                        subtitle: model.builtinSubtitle(id: id),
                        active: model.isBuiltinActive(id: id),
                        onActivate: { model.activateBuiltin(id: id) }
                    )
                case .custom(let name):
                    if let m = model.customModes.first(where: { $0.name == name }) {
                        CustomModeDetail(
                            scheme: scheme,
                            builtinModes: model.builtinModeIds,
                            mode: m,
                            active: model.isCustomActive(name: m.name),
                            onActivate: { model.activateCustom(name: m.name) },
                            onSave: { draft in model.upsertCustom(draft) },
                            onDelete: { model.deleteCustom(name: m.name) }
                        )
                    } else {
                        JunoChromeEmptyState(
                            title: "Select a style",
                            message: "Choose a built-in style or create a custom one.",
                            symbol: "sparkles"
                        )
                    }
                case .none:
                    JunoChromeEmptyState(
                        title: "Select a style",
                        message: "Choose a built-in style or create a custom one.",
                        symbol: "sparkles"
                    )
                }
            }
            .junoDetailPagePadding()
        }
    }
}

// MARK: - View-model

@MainActor
final class ModesViewModel: ObservableObject {
    enum Selection: Hashable {
        case builtin(String)
        case custom(String)
    }

    struct CustomModeDraft: Equatable {
        var name: String
        var baseMode: String
        var description: String
        var promptPrefix: String
        var enabled: Bool

        var itnOverride: String
        var cleanupOverride: String
        var styleCardName: String
        var snippetScope: String
        var commandPolicy: String
        var autoTransformId: String

        /// The mode's name when the editor opened. Empty string for brand-new
        /// drafts. The upsert path uses this to detect renames so the
        /// payload can include `previous_name` — the broker then deletes
        /// the orphan row in the same lock as the upsert. See fix plan
        /// issue #23.
        var originalName: String = ""
    }

    /// Error surface delivered through `upsertCustom`'s completion. The
    /// broker's `error` string is passed through verbatim so the inline
    /// sheet UI can render the same value the legacy `errorMessage` toast
    /// used to show. See fix plan issue #22.
    struct UpsertError: Error, Equatable {
        let message: String
    }

    /// Function signature for the broker POST transport. Default value
    /// uses `JunoBroker.postJSON`; tests inject a stub that captures
    /// requests and returns scripted responses. Keeping this as a closure
    /// (rather than a protocol) is the smallest possible seam: every
    /// production caller still goes through `JunoBroker`. See fix plan
    /// issues #22 and #23.
    typealias PostTransport = (_ path: String, _ payload: [String: Any], _ completion: @escaping ([String: Any]) -> Void) -> Void

    struct CustomMode: Equatable {
        let name: String
        let baseMode: String
        let description: String
        let promptPrefix: String
        let enabled: Bool

        let itnOverride: String?
        let cleanupOverride: String?
        let styleCardName: String?
        let snippetScope: String?
        let commandPolicy: String?
        let autoTransformId: String?

        func toDraft() -> CustomModeDraft {
            CustomModeDraft(
                name: name,
                baseMode: baseMode,
                description: description,
                promptPrefix: promptPrefix,
                enabled: enabled,
                itnOverride: itnOverride ?? "",
                cleanupOverride: cleanupOverride ?? "",
                styleCardName: styleCardName ?? "",
                snippetScope: snippetScope ?? "",
                commandPolicy: commandPolicy ?? "",
                autoTransformId: autoTransformId ?? "",
                originalName: name
            )
        }
    }

    @Published var builtinModeIds: [String] = []
    @Published var customModes: [CustomMode] = []
    @Published var selection: Selection? = nil
    @Published var errorMessage: String? = nil

    func dismissError() { errorMessage = nil }
    @Published var showCreateSheet: Bool = false

    /// Broker POST transport. Production wires this to
    /// `JunoBroker.postJSON`; tests inject a stub.
    private let postTransport: PostTransport

    init(postTransport: @escaping PostTransport = JunoBroker.postJSON) {
        self.postTransport = postTransport
    }

    // Current mode (from /modes/current)
    @Published private(set) var activeEffectiveMode: String = "default_surface"
    @Published private(set) var activeSource: String = "auto"
    @Published private(set) var activeCustomName: String? = nil
    @Published private(set) var activeManualName: String? = nil

    var currentModeIsCustom: Bool { (activeSource == "custom") && !(activeCustomName ?? "").isEmpty }

    var currentModeLabel: String {
        if currentModeIsCustom, let n = activeCustomName, !n.isEmpty { return n }
        return JunoUserFacingCopy.builtinModeTitle(id: activeEffectiveMode)
    }

    func refreshAll() {
        refreshBuiltin()
        refreshCustom()
        refreshCurrent()
    }

    func refreshBuiltin() {
        JunoBroker.getJSON(path: "api/broker/modes/builtin") { [weak self] obj in
            guard let self else { return }
            let ok = (obj["ok"] as? Bool) ?? true
            if !ok {
                self.errorMessage = (obj["error"] as? String) ?? "Could not load modes"
                self.builtinModeIds = []
                return
            }
            let items = (obj["modes"] as? [[String: Any]]) ?? []
            // Issue #5: even if an older broker returns ``default_surface``, never
            // surface it as a clickable manual built-in. AUTO is exposed via the
            // separate ``activateAuto()`` affordance.
            let ids = items.compactMap { $0["id"] as? String }.filter { $0 != "default_surface" }
            self.builtinModeIds = ids
            if self.selection == nil, let first = self.builtinModeIds.first {
                self.selection = .builtin(first)
            }
        }
    }

    func refreshCustom() {
        JunoBroker.getJSON(path: "api/broker/modes/custom") { [weak self] obj in
            guard let self else { return }
            let ok = (obj["ok"] as? Bool) ?? true
            if !ok {
                self.errorMessage = (obj["error"] as? String) ?? "Could not load custom modes"
                self.customModes = []
                return
            }
            let items = (obj["modes"] as? [[String: Any]]) ?? []
            self.customModes = items.compactMap { raw in
                guard let name = raw["name"] as? String, !name.isEmpty else { return nil }
                return CustomMode(
                    name: name,
                    baseMode: (raw["base_mode"] as? String) ?? "default_surface",
                    description: (raw["description"] as? String) ?? "",
                    promptPrefix: (raw["prompt_prefix"] as? String) ?? "",
                    enabled: (raw["enabled"] as? Bool) ?? true,
                    itnOverride: raw["itn_override"] as? String,
                    cleanupOverride: raw["cleanup_override"] as? String,
                    styleCardName: raw["style_card_name"] as? String,
                    snippetScope: raw["snippet_scope"] as? String,
                    commandPolicy: raw["command_policy"] as? String,
                    autoTransformId: raw["auto_transform_id"] as? String
                )
            }
            self.customModes.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        }
    }

    func refreshCurrent() {
        JunoBroker.getJSON(path: "api/broker/modes/current") { [weak self] obj in
            guard let self else { return }
            let ok = (obj["ok"] as? Bool) ?? true
            if !ok {
                self.errorMessage = (obj["error"] as? String) ?? "Could not load current style"
                return
            }
            let sel = (obj["selection"] as? [String: Any]) ?? [:]
            self.activeEffectiveMode = (sel["effective_mode"] as? String) ?? "default_surface"
            self.activeSource = (sel["mode_source"] as? String) ?? "auto"
            self.activeCustomName = sel["custom_mode_name"] as? String
            self.activeManualName = sel["manual_mode_name"] as? String
        }
    }

    func makeNewDraft() -> CustomModeDraft {
        let base = (builtinModeIds.first { $0 != "default_surface" }) ?? "formal_email"
        return CustomModeDraft(
            name: "",
            baseMode: base,
            description: "",
            promptPrefix: "",
            enabled: true,
            itnOverride: "",
            cleanupOverride: "",
            styleCardName: "",
            snippetScope: "",
            commandPolicy: "",
            autoTransformId: ""
        )
    }

    func builtinSubtitle(id: String) -> String {
        switch id {
        case "verbatim": return "Type what you say; explicit edit commands still work"
        case "casual_chat": return "Conversational, light cleanup"
        case "formal_email": return "Professional, polished sentences"
        case "structured_notes": return "Bullet‑organized structure"
        case "explicit_rewrite": return "Full rewrite for clarity"
        case "command_mode": return "Voice commands to control Juno"
        default: return "Built-in style"
        }
    }

    func isBuiltinActive(id: String) -> Bool {
        !currentModeIsCustom && activeEffectiveMode == id
    }

    func isCustomActive(name: String) -> Bool {
        currentModeIsCustom && (activeCustomName ?? "") == name
    }

    /// Issue #5: explicit "Auto (per app)" affordance — clears both manual and
    /// custom mode pins so surface presets resolve normally. The UI shows this
    /// as a row above the built-in list (or as a "Use auto" button when a
    /// non-auto mode is currently active).
    func activateAuto() {
        errorMessage = nil
        JunoBroker.postJSON(path: "api/broker/modes/custom/activate", payload: ["name": ""]) { _ in
            JunoBroker.postJSON(path: "api/broker/modes/manual/clear", payload: [:]) { [weak self] resp in
                guard let self else { return }
                let ok = (resp["ok"] as? Bool) ?? false
                if !ok {
                    self.errorMessage = (resp["error"] as? String) ?? "Switch to auto failed"
                }
                self.refreshCurrent()
            }
        }
    }

    func activateBuiltin(id: String) {
        errorMessage = nil
        // Built-in activation: set manual mode, clear any custom mode.
        JunoBroker.postJSON(path: "api/broker/modes/custom/activate", payload: ["name": ""]) { _ in
            JunoBroker.postJSON(path: "api/broker/modes/manual/set", payload: ["mode": id]) { [weak self] resp in
                guard let self else { return }
                let ok = (resp["ok"] as? Bool) ?? false
                if !ok {
                    self.errorMessage = (resp["error"] as? String) ?? "Activate failed"
                }
                self.refreshCurrent()
            }
        }
    }

    func activateCustom(name: String) {
        errorMessage = nil
        // Custom activation: clear manual mode, activate custom.
        JunoBroker.postJSON(path: "api/broker/modes/manual/clear", payload: [:]) { _ in
            JunoBroker.postJSON(path: "api/broker/modes/custom/activate", payload: ["name": name]) { [weak self] resp in
                guard let self else { return }
                let ok = (resp["ok"] as? Bool) ?? false
                if !ok {
                    self.errorMessage = (resp["error"] as? String) ?? "Activate failed"
                }
                self.refreshCurrent()
            }
        }
    }

    func upsertCustom(
        _ draft: CustomModeDraft,
        completion: ((Result<Void, UpsertError>) -> Void)? = nil
    ) {
        errorMessage = nil
        let name = draft.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            errorMessage = "Name is required"
            completion?(.failure(UpsertError(message: "Name is required")))
            return
        }
        var payload: [String: Any] = [
            "name": name,
            "base_mode": draft.baseMode,
            "description": draft.description,
            "prompt_prefix": draft.promptPrefix,
            "enabled": draft.enabled,
            "itn_override": draft.itnOverride.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? NSNull() : draft.itnOverride,
            "cleanup_override": draft.cleanupOverride.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? NSNull() : draft.cleanupOverride,
            "style_card_name": draft.styleCardName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? NSNull() : draft.styleCardName,
            "snippet_scope": draft.snippetScope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? NSNull() : draft.snippetScope,
            "command_policy": draft.commandPolicy.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? NSNull() : draft.commandPolicy,
            "auto_transform_id": draft.autoTransformId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? NSNull() : draft.autoTransformId,
        ]
        // #23: rename hint. Only attach when the user actually changed the
        // name on an existing mode; brand-new drafts (originalName == "")
        // and unchanged-name saves do not send `previous_name`.
        let original = draft.originalName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !original.isEmpty && original != name {
            payload["previous_name"] = original
        }
        postTransport("api/broker/modes/custom/upsert", payload) { [weak self] resp in
            guard let self else { return }
            let ok = (resp["ok"] as? Bool) ?? false
            if !ok {
                let message = (resp["error"] as? String) ?? "Save failed"
                self.errorMessage = message
                self.refreshCustom()
                self.refreshCurrent()
                completion?(.failure(UpsertError(message: message)))
                return
            }
            self.refreshCustom()
            self.refreshCurrent()
            completion?(.success(()))
        }
    }

    func deleteCustom(name: String) {
        errorMessage = nil
        JunoBroker.postJSON(path: "api/broker/modes/custom/delete", payload: ["name": name]) { [weak self] resp in
            guard let self else { return }
            let ok = (resp["ok"] as? Bool) ?? false
            if !ok {
                self.errorMessage = (resp["error"] as? String) ?? "Delete failed"
            }
            self.refreshCustom()
            self.refreshCurrent()
            if case .custom(let selected) = self.selection, selected == name {
                if let first = self.builtinModeIds.first {
                    self.selection = .builtin(first)
                } else {
                    self.selection = nil
                }
            }
        }
    }
}

// MARK: - Advanced picker option sets
//
// These feed `GenericGlyphPopoverPicker`. Each picker uses one uniform
// leading glyph so the rows look cohesive without inventing per-option
// semantics that the original native Picker never communicated. Subtitles
// are intentionally empty — the native source had none, and adding them
// here would be a polish change.

private func kAdvancedOption(_ value: String, _ title: String, glyph: String) -> GenericGlyphPopoverPicker.Option {
    GenericGlyphPopoverPicker.Option(value: value, title: title, subtitle: "", systemName: glyph)
}

private let kItnOptions: [GenericGlyphPopoverPicker.Option] = [
    kAdvancedOption("", "From base style", glyph: "textformat.123"),
    kAdvancedOption("surface_default", "Surface default", glyph: "textformat.123"),
    kAdvancedOption("standard", "Standard", glyph: "textformat.123"),
    kAdvancedOption("messaging", "Messaging style", glyph: "textformat.123"),
    kAdvancedOption("literal_minimal", "Literal — minimal", glyph: "textformat.123"),
]

private let kCleanupOptions: [GenericGlyphPopoverPicker.Option] = [
    kAdvancedOption("", "From base style", glyph: "wand.and.sparkles"),
    kAdvancedOption("minimal_only", "Minimal", glyph: "wand.and.sparkles"),
    kAdvancedOption("light_only", "Light cleanup", glyph: "wand.and.sparkles"),
    kAdvancedOption("messaging_oriented", "Messaging", glyph: "wand.and.sparkles"),
    kAdvancedOption("grammar_and_paragraphs", "Grammar + paragraphs", glyph: "wand.and.sparkles"),
    kAdvancedOption("structure_forward", "Structure-forward", glyph: "wand.and.sparkles"),
    kAdvancedOption("full", "Full cleanup", glyph: "wand.and.sparkles"),
    kAdvancedOption("minimal_for_commands", "Commands minimal", glyph: "wand.and.sparkles"),
]

private let kSnippetScopeOptions: [GenericGlyphPopoverPicker.Option] = [
    kAdvancedOption("", "From base style", glyph: "scope"),
    kAdvancedOption("none", "None", glyph: "scope"),
    kAdvancedOption("none_unless_invoked", "Vocab only (on demand)", glyph: "scope"),
    kAdvancedOption("surface_plus_global", "Surface + Global", glyph: "scope"),
    kAdvancedOption("messaging_plus_global", "Messaging + Global", glyph: "scope"),
    kAdvancedOption("email_plus_global", "Email + Global", glyph: "scope"),
    kAdvancedOption("docs_plus_global", "Docs + Global", glyph: "scope"),
    kAdvancedOption("all_scopes", "All scopes", glyph: "scope"),
]

private let kCommandPolicyOptions: [GenericGlyphPopoverPicker.Option] = [
    kAdvancedOption("", "From base style", glyph: "command"),
    kAdvancedOption("strict_threshold", "Strict", glyph: "command"),
    kAdvancedOption("narrow_threshold", "Narrow", glyph: "command"),
    kAdvancedOption("moderate_threshold", "Moderate", glyph: "command"),
    kAdvancedOption("inline_moderate_ambiguity", "Inline moderate", glyph: "command"),
    kAdvancedOption("low_threshold", "Low", glyph: "command"),
    kAdvancedOption("confidence_gated", "Confidence-gated", glyph: "command"),
]

private let kAutoTransformOptions: [GenericGlyphPopoverPicker.Option] = [
    kAdvancedOption("", "None", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("polish", "Polish", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("fix_grammar", "Fix grammar", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("make_shorter", "Make shorter", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("make_longer", "Make longer", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("make_clearer", "Make clearer", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("make_more_formal", "More formal", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("make_more_casual", "More casual", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("bulletize", "Bullets", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("numbered_list", "Numbered list", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("summarize", "Summarize", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("simplify", "Simplify", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("translate_preserve_meaning", "Translate", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("email_rewrite", "Email rewrite", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("slack_rewrite", "Slack rewrite", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("notes_rewrite", "Notes rewrite", glyph: "arrow.triangle.2.circlepath"),
    kAdvancedOption("checklist_rewrite", "Checklist", glyph: "arrow.triangle.2.circlepath"),
]

// MARK: - UI building blocks

private struct ModeRow: View {
    let scheme: ColorScheme
    let title: String
    let subtitle: String
    let icon: String
    let active: Bool
    var isSelected: Bool = false

    private var titleColor: Color {
        return JunoTheme.primaryText(scheme)
    }

    private var subtitleColor: Color {
        if isSelected { return JunoTheme.primaryText(scheme).opacity(0.72) }
        return JunoTheme.secondaryText(scheme)
    }

    private var iconColor: Color {
        return JunoDesignTokens.accent
    }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(iconColor)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text(title)
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(titleColor)
                    if active {
                        Text("ACTIVE")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .tracking(0.8)
                            .foregroundStyle(isSelected ? JunoDesignTokens.accent : JunoDesignTokens.meadow)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                Capsule().fill(
                                    (isSelected ? JunoDesignTokens.accent.opacity(0.14) : JunoDesignTokens.meadow.opacity(0.12))
                                )
                            )
                    }
                }
                Text(subtitle)
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(subtitleColor)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(isSelected ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.12 : 0.07) : Color.clear)
        )
    }
}

private struct BuiltinModeDetail: View {
    let scheme: ColorScheme
    let modeId: String
    let title: String
    let subtitle: String
    let active: Bool
    let onActivate: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            JunoPageHeader(
                eyebrow: active ? "Active style" : "Writing style",
                title: title,
                subtitle: subtitle,
                trailing: {
                    Button(active ? "Active" : "Use this style") { onActivate() }
                        .junoPrimaryActionButton()
                        .disabled(active)
                }
            )

            VStack(alignment: .leading, spacing: 10) {
                JunoSectionLabel(text: "How it writes")
                Text(JunoUserFacingCopy.builtinModeDetail(id: modeId))
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .junoPageCard(padding: 16)
        }
    }
}

private struct CustomModeDetail: View {
    let scheme: ColorScheme
    let builtinModes: [String]
    let mode: ModesViewModel.CustomMode
    let active: Bool
    let onActivate: () -> Void
    let onSave: (ModesViewModel.CustomModeDraft) -> Void
    let onDelete: () -> Void

    @State private var draft: ModesViewModel.CustomModeDraft
    @State private var dirty: Bool = false
    @State private var showAdvanced: Bool = false

    init(
        scheme: ColorScheme,
        builtinModes: [String],
        mode: ModesViewModel.CustomMode,
        active: Bool,
        onActivate: @escaping () -> Void,
        onSave: @escaping (ModesViewModel.CustomModeDraft) -> Void,
        onDelete: @escaping () -> Void
    ) {
        self.scheme = scheme
        self.builtinModes = builtinModes
        self.mode = mode
        self.active = active
        self.onActivate = onActivate
        self.onSave = onSave
        self.onDelete = onDelete
        _draft = State(initialValue: mode.toDraft())
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    JunoEyebrow(text: active ? "Active custom style" : "Custom style")
                    Text(mode.name)
                        .font(.system(.title2, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                }
                Spacer()
                Button(active ? "Active" : "Use this style") { onActivate() }
                    .junoPrimaryActionButton()
                    .disabled(active)
            }

            Divider().opacity(0.35)

            VStack(alignment: .leading, spacing: 10) {
                fieldLabel("Name")
                TextField("Style name", text: $draft.name)
                    .textFieldStyle(.roundedBorder)
                    .focusEffectDisabled()
                    .onChange(of: draft.name) { _ in dirty = true }
            }

            VStack(alignment: .leading, spacing: 10) {
                fieldLabel("Base style")
                GenericGlyphPopoverPicker(
                    selection: $draft.baseMode,
                    options: builtinModes.map {
                        GenericGlyphPopoverPicker.Option(
                            value: $0,
                            title: JunoUserFacingCopy.builtinModeTitle(id: $0),
                            subtitle: "",
                            systemName: "doc.richtext"
                        )
                    }
                ) { _ in dirty = true }
            }

            VStack(alignment: .leading, spacing: 10) {
                fieldLabel("Description (optional)")
                TextField("What is this style for?", text: $draft.description)
                    .textFieldStyle(.roundedBorder)
                    .focusEffectDisabled()
                    .onChange(of: draft.description) { _ in dirty = true }
            }

            VStack(alignment: .leading, spacing: 10) {
                fieldLabel("Instructions")
                TextEditor(text: $draft.promptPrefix)
                    .font(.system(.body, design: .rounded))
                    .focusEffectDisabled()
                    .frame(minHeight: 200)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(JunoTheme.border(scheme).opacity(0.5), lineWidth: 0.5)
                    )
                    .onChange(of: draft.promptPrefix) { _ in dirty = true }
                Text("Tip: Be specific. If you leave instructions blank, results can be unpredictable.")
                    .font(.caption2)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }

            DisclosureGroup(isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 14) {
                    Toggle("Enabled", isOn: $draft.enabled)
                        .onChange(of: draft.enabled) { _ in dirty = true }

                    advancedPicker("ITN", options: kItnOptions, selection: $draft.itnOverride)
                    advancedPicker("Cleanup", options: kCleanupOptions, selection: $draft.cleanupOverride)
                    advancedTextField("Style card name", text: $draft.styleCardName)
                    advancedPicker("Snippet scope", options: kSnippetScopeOptions, selection: $draft.snippetScope)
                    advancedPicker("Command detection", options: kCommandPolicyOptions, selection: $draft.commandPolicy)
                    advancedPicker("Auto-transform", options: kAutoTransformOptions, selection: $draft.autoTransformId)
                }
                .padding(.top, 8)
            } label: {
                Text("Advanced")
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
            }

            Divider().opacity(0.35)

            HStack(spacing: 10) {
                Button("Delete", role: .destructive) { onDelete() }
                    .junoSecondaryActionButton()
                Spacer()
                Button("Save") {
                    onSave(draft)
                    dirty = false
                }
                .junoPrimaryActionButton()
                .disabled(draft.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !dirty)
            }
        }
        .padding(18)
        .premiumCard()
        .onChange(of: mode) { _ in
            draft = mode.toDraft()
            dirty = false
        }
    }

    private func fieldLabel(_ s: String) -> some View {
        Text(s)
            .font(.caption)
            .foregroundStyle(JunoTheme.secondaryText(scheme))
    }

    private func advancedTextField(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            fieldLabel(label)
            TextField("", text: text)
                .textFieldStyle(.roundedBorder)
                .focusEffectDisabled()
                .onChange(of: text.wrappedValue) { _ in dirty = true }
        }
    }

    private func advancedPicker(_ label: String, options: [GenericGlyphPopoverPicker.Option], selection: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            fieldLabel(label)
            GenericGlyphPopoverPicker(
                selection: selection,
                options: options
            ) { _ in dirty = true }
        }
    }
}

private struct CustomModeEditorSheet: View {
    let scheme: ColorScheme
    let title: String
    let builtinModes: [String]
    @State var draft: ModesViewModel.CustomModeDraft
    /// Save callback. Invokes `done(result)` once the broker has replied.
    /// On `.success` the sheet dismisses; on `.failure` it stays open and
    /// renders an inline error so the user can correct and retry. See
    /// fix plan issue #22.
    let onSave: (ModesViewModel.CustomModeDraft, @escaping (Result<Void, ModesViewModel.UpsertError>) -> Void) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var showAdvanced: Bool = false
    @State private var isSaving: Bool = false
    @State private var inlineError: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text(title)
                        .font(.system(.title2, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))

                    Text("Custom styles layer your instructions on top of a built-in base. Pick a base that matches most of your writing, then spell out how Juno should behave in Instructions.")
                        .font(.callout)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Name")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                        TextField("Style name", text: $draft.name)
                            .textFieldStyle(.roundedBorder)
                            .focusEffectDisabled()
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Base style")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                        GenericGlyphPopoverPicker(
                            selection: $draft.baseMode,
                            options: builtinModes.map {
                                GenericGlyphPopoverPicker.Option(
                                    value: $0,
                                    title: JunoUserFacingCopy.builtinModeTitle(id: $0),
                                    subtitle: "",
                                    systemName: "doc.richtext"
                                )
                            }
                        )
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Description (optional)")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                        TextField("What is this style for?", text: $draft.description)
                            .textFieldStyle(.roundedBorder)
                            .focusEffectDisabled()
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Instructions")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                        TextEditor(text: $draft.promptPrefix)
                            .font(.system(.body, design: .rounded))
                            .focusEffectDisabled()
                            .frame(minHeight: 200)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(JunoTheme.border(scheme).opacity(0.5), lineWidth: 0.5)
                            )
                    }

                    DisclosureGroup(isExpanded: $showAdvanced) {
                        VStack(alignment: .leading, spacing: 14) {
                            Toggle("Enabled", isOn: $draft.enabled)
                            sheetAdvancedPicker("ITN", options: kItnOptions, selection: $draft.itnOverride)
                            sheetAdvancedPicker("Cleanup", options: kCleanupOptions, selection: $draft.cleanupOverride)
                            sheetAdvancedTextField("Style card name", text: $draft.styleCardName)
                            sheetAdvancedPicker("Snippet scope", options: kSnippetScopeOptions, selection: $draft.snippetScope)
                            sheetAdvancedPicker("Command detection", options: kCommandPolicyOptions, selection: $draft.commandPolicy)
                            sheetAdvancedPicker("Auto-transform", options: kAutoTransformOptions, selection: $draft.autoTransformId)
                        }
                        .padding(.top, 8)
                    } label: {
                        Text("Advanced")
                            .font(.system(.callout, design: .rounded).weight(.semibold))
                            .foregroundStyle(JunoTheme.primaryText(scheme))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 520)

            Divider().opacity(0.35)
                .padding(.top, 12)

            if let inlineError, !inlineError.isEmpty {
                Text(inlineError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.top, 6)
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .junoSecondaryActionButton()
                    .disabled(isSaving)
                Spacer()
                Button(isSaving ? "Saving…" : "Create") {
                    isSaving = true
                    inlineError = nil
                    onSave(draft) { result in
                        isSaving = false
                        switch result {
                        case .success:
                            dismiss()
                        case .failure(let err):
                            inlineError = err.message
                        }
                    }
                }
                .junoPrimaryActionButton()
                .disabled(isSaving || draft.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.top, 12)
        }
        .padding(18)
        .frame(minWidth: 640)
    }

    private func sheetAdvancedTextField(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            TextField("", text: text)
                .textFieldStyle(.roundedBorder)
                .focusEffectDisabled()
        }
    }

    private func sheetAdvancedPicker(_ label: String, options: [GenericGlyphPopoverPicker.Option], selection: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            GenericGlyphPopoverPicker(
                selection: selection,
                options: options
            )
        }
    }
}
