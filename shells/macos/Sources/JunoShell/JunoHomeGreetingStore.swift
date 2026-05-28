import Combine
import Foundation

/// Owns home hero greeting copy across `JunoHomeView` teardown so sidebar navigation
/// does not replay the full local + broker bootstrap. Refreshes broker text only when
/// the staleness identity actually changes; reachability flapping never re-paints.
@MainActor
final class JunoHomeGreetingStore: ObservableObject {
    enum Source: String {
        case local
        case broker
    }

    enum HydrationOutcome: Equatable {
        /// Hero should snap `shown*` to ``headline``/``subline`` (e.g. return to Home).
        case snapToStore
        /// First session paint: store holds local lines; hero may run typewriter.
        case initialLocalTypewriter
        /// A background broker fetch was scheduled; copy may update later.
        case brokerFetchScheduled
    }

    @Published private(set) var headline: String = ""
    @Published private(set) var subline: String = ""
    @Published private(set) var source: Source = .local
    /// Bumps whenever ``headline``/``subline`` change so the hero can sync or animate.
    @Published private(set) var contentRevision: Int = 0
    @Published private(set) var isFetching: Bool = false
    /// Current 3-hour staleness identity used for the hero greeting prompt.
    @Published private(set) var greetingIdentity: String = ""
    /// Bumps whenever the hero should run the typewriter reveal for the local greeting.
    @Published private(set) var typewriterToken: Int = 0

    private var debounceTask: Task<Void, Never>?
    /// Identity at which we last got a successful broker greeting. Once set, we will
    /// not refetch for the same identity even on reachability flaps or revisits.
    private var lastSuccessfulBrokerIdentity: String?
    /// Identity at which the last broker fetch failed. Suppresses retries until the
    /// identity rolls (3-hour bucket or display-name change) — protects against
    /// thrash when the broker is down.
    private var lastFailedBrokerIdentity: String?
    /// Identity the currently-shown local lines were generated for. Used to keep
    /// the local subline stable across navigations within the same bucket.
    private var currentLocalIdentity: String?
    private let brokerDebounceNs: UInt64 = 400_000_000

    deinit {
        debounceTask?.cancel()
    }

    /// Call from Home hero when context changes (appear, revisit, broker reachability).
    func hydrate(brokerReachable: Bool) -> HydrationOutcome {
        let identity = JunoHomeGreeting.greetingStalenessIdentity()
        greetingIdentity = identity
        let hasLines = !headline.isEmpty || !subline.isEmpty

        // First paint: render local immediately, schedule broker if eligible.
        if !hasLines {
            applyPair(JunoHomeGreeting.heroLines(), source: .local, bumpRevision: true)
            currentLocalIdentity = identity
            typewriterToken += 1
            if brokerReachable && shouldFetchBroker(for: identity) {
                scheduleDebouncedBrokerFetch()
                return .initialLocalTypewriter
            }
            return .initialLocalTypewriter
        }

        // Identity rolled (3-hour bucket or display name changed): refresh local
        // copy unless we already have a fresh broker result for the new identity.
        if currentLocalIdentity != identity {
            if lastSuccessfulBrokerIdentity != identity {
                applyPair(JunoHomeGreeting.heroLines(), source: .local, bumpRevision: true)
                currentLocalIdentity = identity
                typewriterToken += 1
                if brokerReachable && shouldFetchBroker(for: identity) {
                    scheduleDebouncedBrokerFetch()
                }
                return .initialLocalTypewriter
            }
            return .snapToStore
        }

        // Identity unchanged. Only fetch if we have neither a success nor a recent
        // failure for this identity — flaps and revisits are no-ops.
        if brokerReachable && shouldFetchBroker(for: identity) {
            scheduleDebouncedBrokerFetch()
            return .brokerFetchScheduled
        }

        return .snapToStore
    }

    func cancelDebouncedFetch() {
        debounceTask?.cancel()
        debounceTask = nil
    }

    private func shouldFetchBroker(for identity: String) -> Bool {
        if isFetching { return false }
        if lastSuccessfulBrokerIdentity == identity { return false }
        if lastFailedBrokerIdentity == identity { return false }
        return true
    }

    private func scheduleDebouncedBrokerFetch() {
        debounceTask?.cancel()
        debounceTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: brokerDebounceNs)
            guard !Task.isCancelled else { return }
            requestBrokerGreeting()
        }
    }

    private func requestBrokerGreeting() {
        let identityAtStart = JunoHomeGreeting.greetingStalenessIdentity()
        isFetching = true
        JunoBroker.fetchHomeGreeting { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                self.isFetching = false
                let identityNow = JunoHomeGreeting.greetingStalenessIdentity()
                // If identity rolled mid-flight, the response is for a stale bucket; drop it.
                guard identityAtStart == identityNow else { return }
                switch result {
                case .success(let pair):
                    let changed = pair.headline != self.headline || pair.subline != self.subline
                    self.applyPair(pair, source: .broker, bumpRevision: changed)
                    self.lastSuccessfulBrokerIdentity = identityNow
                    self.lastFailedBrokerIdentity = nil
                case .failure:
                    // Keep the local lines on screen; record failure so we don't retry
                    // for this identity until the bucket rolls.
                    self.lastFailedBrokerIdentity = identityNow
                }
            }
        }
    }

    private func applyPair(_ pair: (headline: String, subline: String), source: Source, bumpRevision: Bool) {
        headline = pair.headline
        subline = pair.subline
        self.source = source
        if bumpRevision {
            contentRevision += 1
        }
    }
}
