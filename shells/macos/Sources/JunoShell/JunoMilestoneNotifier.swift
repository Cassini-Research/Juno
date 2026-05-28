import Combine
import SwiftUI

/// Fires brand-kit milestone UI when lifetime word totals cross 100-word boundaries.
final class JunoMilestoneNotifier: ObservableObject {
    static let shared = JunoMilestoneNotifier()

    enum Variant {
        case signatureComma
        case signatureFull
        /// First-run onboarding hero (comma + Juno, no milestone copy).
        case onboardingHero
    }

    @Published private(set) var active: Variant?

    private var dismissWorkItem: DispatchWorkItem?

    private init() {}

    func notifyIfMilestone(crossed: Bool, useFullLockup: Bool) {
        // The mid-dictation milestone overlay (every 100 words) was rendering
        // inside the HUD's small panel frame and the centered comma at scale
        // 1.12 visually overflowed — read by users as a glitchy expanded-logo
        // flash. Disabling here is the conservative fix: milestones still
        // exist conceptually (the lifetime word counter advances), they just
        // don't take over the HUD any more. The deliberate one-shot
        // onboarding-hero moment is preserved via `playOnboardingHeroIfNeeded`.
        _ = crossed
        _ = useFullLockup
    }

    /// One-shot full lockup after onboarding (brand kit Delight B).
    func playOnboardingHeroIfNeeded() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard !JunoUserDefaults.onboardingBrandDelightShown else { return }
            JunoUserDefaults.onboardingBrandDelightShown = true
            self.dismissWorkItem?.cancel()
            self.active = .onboardingHero
            let work = DispatchWorkItem { [weak self] in self?.active = nil }
            self.dismissWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.6, execute: work)
        }
    }
}
