import AppKit
import Combine
import JunoHotkeyCore
import SwiftUI

final class JunoOverlayPanel: NSPanel {
    private static let escapeKeyCode: UInt16 = 53
    private static let dragThreshold: CGFloat = 4
    /// Conservative prewarm size: matches the smallest resting HUD shape so
    /// that if SwiftUI's first layout pass briefly reports the prewarm frame
    /// (instead of the intrinsic island size), nothing oversized renders. The
    /// island grows to its natural width on the first real `positionAndShow`
    /// once the SwiftUI tree has settled. Previously this was 464×88, which
    /// could leak as a stretched-comma flash on the first show.
    static let defaultPrewarmSize = NSSize(width: 192, height: 56)

    var onEscape: (() -> Void)?
    var onCopy: (() -> Bool)?
    var onDragFrameChanged: ((NSRect) -> NSRect)?
    private var dragStartMouseLocation: NSPoint?
    private var dragStartFrame: NSRect?
    private var draggingHUD: Bool = false

    init(rootView: AnyView) {
        super.init(
            contentRect: NSRect(origin: .zero, size: Self.defaultPrewarmSize),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = false
        // Stay above normal document windows so the island reads like a compact
        // overlay near the upper screen edge.
        self.level = .floating
        self.hidesOnDeactivate = false
        // Allow taps on the island (e.g. copy draft) while it is visible.
        self.ignoresMouseEvents = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        let hosting = NSHostingView(rootView: rootView)
        hosting.wantsLayer = true
        self.contentView = hosting
    }

    override var canBecomeKey: Bool {
        // CRITICAL: gate key-eligibility on the panel actually being
        // visible. If the panel is offscreen (post orderOut) or fully
        // transparent (mid-fade or just-prewarmed), AppKit must NOT
        // be allowed to pick it as the app's key window — otherwise
        // an invisible HUD island silently steals every keyboard /
        // responder event from the visible main window. This is the
        // root cause of "main window visible but unclickable after
        // any background/restore cycle" — the prewarm leaves a
        // canBecomeKey=true panel at level=.floating, alpha=0, off-
        // screen in the window list. Returning false here means even
        // if the orderOut path has any leak (Sequoia AppKit bug or
        // a SwiftUI hosting-view interaction), the panel still cannot
        // become key.
        guard isVisible else { return false }
        return alphaValue > 0.05
    }
    override var canBecomeMain: Bool { false }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == Self.escapeKeyCode {
            handleEscape()
            return
        }
        super.keyDown(with: event)
    }

    override func cancelOperation(_ sender: Any?) {
        handleEscape()
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        if event.keyCode == Self.escapeKeyCode {
            handleEscape()
            return true
        }
        if JunoHotkeyEventLine.isCommandCopy(
            keyCode: event.keyCode,
            modifierFlags: event.modifierFlags,
            isRepeat: event.isARepeat
        ), onCopy?() == true {
            return true
        }
        return super.performKeyEquivalent(with: event)
    }

    override func sendEvent(_ event: NSEvent) {
        switch event.type {
        case .leftMouseDown:
            dragStartMouseLocation = NSEvent.mouseLocation
            dragStartFrame = frame
            draggingHUD = false
            super.sendEvent(event)
        case .leftMouseDragged:
            guard let startMouse = dragStartMouseLocation, let startFrame = dragStartFrame else {
                super.sendEvent(event)
                return
            }
            let currentMouse = NSEvent.mouseLocation
            let dx = currentMouse.x - startMouse.x
            let dy = currentMouse.y - startMouse.y
            if !draggingHUD {
                let distance = hypot(dx, dy)
                if distance < Self.dragThreshold {
                    super.sendEvent(event)
                    return
                }
                draggingHUD = true
            }
            var next = startFrame
            next.origin.x += dx
            next.origin.y += dy
            if let onDragFrameChanged {
                next = onDragFrameChanged(next)
            }
            setFrame(next, display: true, animate: false)
        case .leftMouseUp:
            let consumedDrag = draggingHUD
            dragStartMouseLocation = nil
            dragStartFrame = nil
            draggingHUD = false
            if !consumedDrag {
                super.sendEvent(event)
            }
        default:
            super.sendEvent(event)
        }
    }

    private func handleEscape() {
        DispatchQueue.main.async { [weak self] in
            self?.onEscape?()
        }
    }
}

// MARK: - Overlay root

struct JunoOverlayView: View {
    @ObservedObject var controller: DictationController

    /// Mirrors `JunoUserDefaults.hudLiveTranscriptionsEnabled` and refreshes
    /// when the user flips the toggle so the HUD swaps surfaces live, without
    /// requiring the overlay coordinator to be re-installed.
    @State private var liveTranscriptionsEnabled: Bool = JunoUserDefaults.hudLiveTranscriptionsEnabled

    var body: some View {
        Group {
            if liveTranscriptionsEnabled {
                JunoBrandIslandStack(controller: controller)
            } else {
                JunoBrandIslandCompact(controller: controller)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .onReceive(NotificationCenter.default.publisher(for: UserDefaults.didChangeNotification)) { _ in
            let next = JunoUserDefaults.hudLiveTranscriptionsEnabled
            if next != liveTranscriptionsEnabled {
                liveTranscriptionsEnabled = next
            }
        }
    }
}

// MARK: - Coordinator

final class JunoOverlayCoordinator {
    private var panel: JunoOverlayPanel?
    private var cancellables = Set<AnyCancellable>()
    private var hideWorkItem: DispatchWorkItem?
    private weak var controllerRef: DictationController?
    /// Tracks whether the panel is currently visible so we play the open sound
    /// only on the idle→visible transition. Escape is handled by ``juno-hotkey``
    /// (same global event path as push-to-talk) so it works while another app is focused.
    private var isCurrentlyVisible: Bool = false
    private var fadeGeneration: UInt64 = 0
    private var sessionDraggedCenter: NSPoint?
    private var resignActiveObserver: NSObjectProtocol?
    private static let fadeOutSeconds: TimeInterval = 0.26
    private static let visibleHideDelay: TimeInterval = 0.9

    deinit {
        if let resignActiveObserver {
            NotificationCenter.default.removeObserver(resignActiveObserver)
        }
    }

    @MainActor
    func install(controller: DictationController) {
        cancellables.removeAll()
        controllerRef = controller
        // Defensive cleanup hook (paired with the fadeOutAndHide fix):
        // when the app deactivates mid-fade, Core Animation pauses and
        // the runAnimationGroup completion can fire with alpha != 0.
        // Force the panel offscreen + orderOut now so we never leave a
        // floating-level canBecomeKey=true panel in the window list at
        // intermediate alpha — that's the invisible-key-window bug that
        // ate every click after the user backgrounded Juno.
        if resignActiveObserver == nil {
            resignActiveObserver = NotificationCenter.default.addObserver(
                forName: NSApplication.willResignActiveNotification,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                self?.forceHidePanelImmediate()
            }
        }
        let root = AnyView(
            JunoOverlayView(controller: controller)
                .fixedSize(horizontal: true, vertical: true)
        )
        let created = JunoOverlayPanel(rootView: root)
        created.onEscape = { [weak controller] in
            controller?.cancelDictation()
        }
        created.onCopy = { [weak controller] in
            guard let controller else { return false }
            guard JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
                hotkeyLine: JunoHotkeyEventLine.copy,
                copyableTranscript: controller.copyableTranscript,
                hudStateWire: controller.state
            ) else {
                return false
            }
            controller.copyCopyableTranscriptToClipboard()
            return true
        }
        created.onDragFrameChanged = { [weak self] proposedFrame in
            guard let self else { return proposedFrame }
            let clamped = self.clampedFrame(proposedFrame, preferredScreen: self.screenUnderMouse())
            self.sessionDraggedCenter = NSPoint(x: clamped.midX, y: clamped.midY)
            return clamped
        }
        panel = created
        // Pre-warm only the hosting view layout. Do NOT order this panel
        // during app bootstrap: it is a floating .nonactivatingPanel, and
        // making it the app's first ordered window on relaunch leaves the
        // visible main window rendered but not receiving AppKit mouse
        // events. Even after orderOut, AppKit's internal window-ordering
        // history retains the floating panel as the most-recent ordered
        // window — and on relaunch (where the main window opens in a
        // tight sequence right after bootstrap) that history wedges
        // makeKeyAndOrderFront so the main window never wins clean
        // key/main status. The first real orderFront happens in
        // positionAndShow() on actual dictation.
        //
        // This was the root cause behind the AX-works-mouse-fails relaunch
        // regression: older builds ordered the floating panel during bootstrap
        // and left the main window unable to win key/main status.
        if let screen = NSScreen.main {
            var prewarmFrame = created.frame
            prewarmFrame.size = JunoOverlayPanel.defaultPrewarmSize
            prewarmFrame.origin = NSPoint(x: screen.frame.maxX + 4000, y: screen.frame.maxY + 4000)
            created.setFrame(prewarmFrame, display: false)
            created.alphaValue = 0
            created.contentView?.layoutSubtreeIfNeeded()
        }

        let layoutTick = Publishers.MergeMany(
            controller.$state.map { _ in () }.eraseToAnyPublisher(),
            controller.$liveDisplayTranscript.map { _ in () }.eraseToAnyPublisher(),
            controller.$liveSpeechHint.map { _ in () }.eraseToAnyPublisher(),
            controller.$transientDoneWordCount.map { _ in () }.eraseToAnyPublisher(),
            controller.$transientActionHUDResult.map { _ in () }.eraseToAnyPublisher(),
            controller.$copyableTranscript.map { _ in () }.eraseToAnyPublisher(),
            controller.$writerDegradedNotice.map { _ in () }.eraseToAnyPublisher(),
            controller.$refiningStartedAt.map { _ in () }.eraseToAnyPublisher(),
            controller.$targetApp.map { _ in () }.eraseToAnyPublisher(),
            JunoMilestoneNotifier.shared.$active.map { _ in () }.eraseToAnyPublisher(),
            // Background sink work — drives the "Saving…" pill and the
            // hold-visible decision in ``updateVisibility``.
            JunoActionExecutor.shared.$inFlight.map { _ in () }.eraseToAnyPublisher()
        )

        let contentTick = Publishers.MergeMany(
            controller.$livePartialText.map { _ in () }.eraseToAnyPublisher(),
            controller.$draftFlashActive.map { _ in () }.eraseToAnyPublisher(),
            controller.$transientCopyToast.map { _ in () }.eraseToAnyPublisher(),
            controller.$delightSweepActive.map { _ in () }.eraseToAnyPublisher()
        )

        // Layout-driving state can change visibility or measured size, so it
        // still goes through `positionAndShow()`. Content-only pulses should
        // not re-measure and reposition the floating panel on every word.
        layoutTick
            .throttle(for: .milliseconds(50), scheduler: DispatchQueue.main, latest: true)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateVisibility()
            }
            .store(in: &cancellables)

        contentTick
            .throttle(for: .milliseconds(33), scheduler: DispatchQueue.main, latest: true)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateContentIfNeeded()
            }
            .store(in: &cancellables)

        updateVisibility()
    }

    @MainActor
    private func updateContentIfNeeded() {
        guard isCurrentlyVisible, let panel else { return }
        panel.contentView?.layoutSubtreeIfNeeded()
    }

    @MainActor
    private func updateVisibility() {
        guard let controller = controllerRef else { return }
        // Only show the floating HUD during an active dictation attempt (or
        // brief post-dictation / milestone flourishes)—never as an idle chip.
        // The predicate lives on `HUDState` itself so the producer and the
        // overlay can never drift apart (Issue #8).
        let inDictationFlow = controller.hudState.isInDictationFlow
        let showTransient = controller.transientDoneWordCount != nil
        let hasActionResult = controller.transientActionHUDResult != nil
        // Voice Action sinks (Notes / Reminders / Calendar) run on a
        // background queue *after* dictation has settled to idle. Keeping
        // the HUD up while ``inFlight`` is set bridges that 0.5–6 s gap so
        // the user sees a "Saving…" pill instead of an empty screen and
        // wondering whether the action ran at all.
        let actionsRunning = JunoActionExecutor.shared.inFlight != nil
        let milestone = JunoMilestoneNotifier.shared.active != nil
        let hasCopyReady = controller.copyableTranscript != nil
        if inDictationFlow || showTransient || hasActionResult || actionsRunning || milestone || hasCopyReady {
            cancelPendingHide()
            positionAndShow()
        } else {
            scheduleHide()
        }
    }

    private func positionAndShow() {
        guard let panel else { return }
        guard let controller = controllerRef else { return }
        guard let screen = screenForOverlay() else { return }
        let vf = screen.visibleFrame
        let sz = measuredPanelSize(for: panel, controller: controller)
        // Clamp width is content-driven so the compact pill (~168pt) and the
        // full island (~440pt) both fit naturally. The 120pt floor and 520pt
        // ceiling exist only as guardrails against degenerate fittingSize
        // values; under normal layout the SwiftUI content drives the frame.
        let w = max(120, min(sz.width, 520))
        let h = max(40, min(sz.height, 320))
        var frame = panel.frame
        frame.size = NSSize(width: w, height: h)
        if let center = sessionDraggedCenter {
            frame.origin.x = center.x - w / 2
            frame.origin.y = center.y - h / 2
        } else {
            let pos = JunoUserDefaults.hudPosition
            switch pos {
            case .topCenter:
                frame.origin.x = vf.midX - w / 2
                let topInset: CGFloat = 10
                frame.origin.y = vf.origin.y + vf.height - h - topInset
            case .bottomCenter:
                frame.origin.x = vf.midX - w / 2
                let bottomInset: CGFloat = 18
                frame.origin.y = vf.origin.y + bottomInset
            }
        }
        let nextFrame = clampedFrame(frame, preferredScreen: screen)
        let wasVisible = isCurrentlyVisible
        // Tolerance gate: micro-resizes (≤2pt in either dimension) come
        // from text wrapping shifting by a single line height when a
        // preview→final correction lands on a word boundary. Treating
        // them as no-ops eliminates the shake-on-replace the user
        // reported without changing the open/grow feel — anything
        // larger than 2pt still resizes (and now animates).
        let frameDelta = max(
            abs(nextFrame.size.width - panel.frame.size.width),
            abs(nextFrame.size.height - panel.frame.size.height)
        )
        let originDelta = max(
            abs(nextFrame.origin.x - panel.frame.origin.x),
            abs(nextFrame.origin.y - panel.frame.origin.y)
        )
        let needsFrameUpdate = !wasVisible || frameDelta > 2.0 || originDelta > 2.0

        if needsFrameUpdate {
            // Always snap. `positionAndShow` is wired to ~15 Combine
            // publishers (live partial text, display transcript, hint,
            // draft flash, copy toast, action in-flight, etc.) and during
            // dictation can fire 5–30× per second. Animating the panel
            // frame on every frame change > 2pt — which 43d1028 did to
            // smooth preview→final corrections — turns each new word
            // into a 0.20 s NSAnimation that overlaps the next, and the
            // user sees the HUD continuously shaking up and down. The
            // tolerance gate above already eliminates the no-op resizes
            // it was meant to fix; SwiftUI handles the visual smoothness
            // inside the panel via `.contentTransition(.opacity)` plus
            // the shimmer pulse on `controller.correctionGeneration`.
            // The panel frame itself just needs to track the content —
            // snapping is cheap and visually correct.
            applyPanelFrame(nextFrame, to: panel, display: true, animate: false)
        }

        if !wasVisible {
            // Idle → visible transition: fade in.
            panel.alphaValue = 0
            panel.orderFrontRegardless()
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.22
                ctx.timingFunction = CAMediaTimingFunction(controlPoints: 0.34, 1.0, 0.64, 1.0)
                panel.animator().alphaValue = 1
            }
            isCurrentlyVisible = true
        } else {
            fadeGeneration &+= 1
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0
                ctx.allowsImplicitAnimation = false
                panel.animator().alphaValue = 1
            }
            panel.orderFrontRegardless()
        }
    }

    /// On the first-ever HUD show, `NSHostingView` can briefly report the
    /// oversized offscreen prewarm frame instead of the island's intrinsic
    /// size. Falling back to a sane HUD size avoids the stretched-logo flash
    /// until SwiftUI finishes its first live layout pass.
    private func measuredPanelSize(for panel: JunoOverlayPanel, controller: DictationController) -> NSSize {
        // Force the hosting view to settle before we sample fittingSize.
        // Setting a sane intermediate frame prevents the first-pass measurement
        // from echoing a stale prewarm size.
        panel.contentView?.invalidateIntrinsicContentSize()
        if !isCurrentlyVisible {
            let probeFrame = NSRect(
                origin: panel.frame.origin,
                size: NSSize(width: JunoDesignTokens.islandWidth, height: 100)
            )
            panel.setFrame(probeFrame, display: false, animate: false)
        }
        panel.contentView?.layoutSubtreeIfNeeded()
        let measured = panel.contentView?.fittingSize ?? .zero
        // Tighter upper bound (was 520) — anything wider than the island plus
        // generous chrome is a layout-glitch echo, not real content. Fall back
        // to the deterministic per-state size in that case.
        if measured.width >= 120, measured.width <= JunoDesignTokens.islandWidth + 32,
           measured.height >= 40, measured.height <= 320 {
            return measured
        }
        return fallbackVisibleSize(for: controller)
    }

    private func fallbackVisibleSize(for controller: DictationController) -> NSSize {
        if controller.copyableTranscript != nil {
            return NSSize(width: JunoDesignTokens.copyReadyIslandSize.width + 24, height: 156)
        }
        if controller.transientDoneWordCount != nil, controller.hudState == .idle {
            return NSSize(width: JunoDesignTokens.doneSize.width + 24, height: 64)
        }
        if controller.transientActionHUDResult != nil, controller.hudState == .idle {
            return NSSize(width: 260, height: 72)
        }
        if !JunoUserDefaults.hudLiveTranscriptionsEnabled {
            return NSSize(width: 192, height: 56)
        }
        if controller.hudState == .refining {
            return NSSize(width: JunoDesignTokens.islandWidth + 24, height: 72)
        }
        return JunoOverlayPanel.defaultPrewarmSize
    }

    private func applyPanelFrame(_ frame: NSRect, to panel: JunoOverlayPanel, display: Bool, animate: Bool) {
        panel.setFrame(frame, display: display, animate: animate)
    }

    private func clampedFrame(_ frame: NSRect, preferredScreen: NSScreen?) -> NSRect {
        guard let screen = preferredScreen ?? screenForFrame(frame) ?? NSScreen.main else { return frame }
        let vf = screen.visibleFrame
        var next = frame
        if next.width > vf.width {
            next.size.width = vf.width
        }
        if next.height > vf.height {
            next.size.height = vf.height
        }
        next.origin.x = min(max(next.origin.x, vf.minX), vf.maxX - next.width)
        next.origin.y = min(max(next.origin.y, vf.minY), vf.maxY - next.height)
        return next
    }

    /// Prefer the screen under the cursor so multi-monitor setups feel correct
    /// on each new HUD session. While the HUD is already visible, preserve the
    /// dragged panel's screen instead.
    private func screenForOverlay() -> NSScreen? {
        if let panel, isCurrentlyVisible, let screen = screenForPanel(panel) {
            return screen
        }
        return screenUnderMouse() ?? NSScreen.main
    }

    private func screenUnderMouse() -> NSScreen? {
        let p = NSEvent.mouseLocation
        for s in NSScreen.screens where s.frame.contains(p) {
            return s
        }
        return nil
    }

    private func screenForPanel(_ panel: JunoOverlayPanel) -> NSScreen? {
        screenForFrame(panel.frame)
    }

    private func screenForFrame(_ frame: NSRect) -> NSScreen? {
        var best: (screen: NSScreen, area: CGFloat)?
        for screen in NSScreen.screens {
            let area = frame.intersection(screen.frame).width * frame.intersection(screen.frame).height
            if area > (best?.area ?? 0) {
                best = (screen, area)
            }
        }
        if let best, best.area > 0 {
            return best.screen
        }
        let center = NSPoint(x: frame.midX, y: frame.midY)
        return NSScreen.screens.first { $0.frame.contains(center) }
    }

    private func scheduleHide() {
        cancelPendingHide()
        let work = DispatchWorkItem { [weak self] in
            self?.fadeOutAndHide()
        }
        hideWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.visibleHideDelay, execute: work)
    }

    private func fadeOutAndHide() {
        guard let panel, isCurrentlyVisible else { return }
        fadeGeneration &+= 1
        let gen = fadeGeneration
        // Single dismiss cue, fired at the start of the fade so it lands in
        // sync with the visual disappearance. Gated by the same defaults
        // toggle as the open cue so the user has one switch for both ends.
        if JunoUserDefaults.hudOpenSoundEnabled {
            JunoHUDSound.playClose()
        }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = Self.fadeOutSeconds
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            guard let self, let panel = self.panel else { return }
            // If a newer fade-out / re-show superseded us, that path
            // owns the next orderOut. Bail.
            guard self.fadeGeneration == gen else { return }
            // CRITICAL: orderOut must run unconditionally when our
            // generation is still current. The previous code gated this
            // on `panel.alphaValue > 0.05` to avoid orderOut'ing a
            // half-faded panel — but if the app deactivates mid-fade,
            // Core Animation pauses, the animator never reaches 0, real
            // time elapses, and this completion fires with alpha=~0.5.
            // The guard then SKIPPED orderOut, leaving a floating-level
            // canBecomeKey=true panel in the window list at intermediate
            // alpha. On next reactivation that panel could win key
            // status invisibly and steal every keyboard / responder
            // event from the main window. The user-reported "click
            // doesn't work after backgrounding" symptom matches exactly.
            // Force the panel to alpha 0 + orderOut here regardless;
            // the next show path (positionAndShow / wasVisible=false
            // branch) restores alpha to 1 before orderFrontRegardless.
            self.isCurrentlyVisible = false
            self.sessionDraggedCenter = nil
            panel.alphaValue = 0
            panel.orderOut(nil)
            panel.alphaValue = 1
        })
    }

    private func cancelPendingHide() {
        hideWorkItem?.cancel()
        hideWorkItem = nil
    }

    /// Force the HUD panel out of the window list immediately, bypassing
    /// fade animation. Called from the willResignActive notification so a
    /// half-faded panel never gets stuck at intermediate alpha when Core
    /// Animation pauses on app deactivation. Idempotent and safe to call
    /// when the panel is already hidden.
    private func forceHidePanelImmediate() {
        guard let panel else { return }
        cancelPendingHide()
        // Bump generation so any in-flight fadeOutAndHide completion
        // bails when it finally fires; we own the orderOut now.
        fadeGeneration &+= 1
        // Snap alpha to 0 with a zero-duration animation so any pending
        // animator's "current value" is also 0 (avoids next show inheriting
        // a stale half-fade state).
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0
            ctx.allowsImplicitAnimation = false
            panel.animator().alphaValue = 0
        }
        isCurrentlyVisible = false
        sessionDraggedCenter = nil
        panel.alphaValue = 0
        panel.orderOut(nil)
        panel.alphaValue = 1
    }
}
