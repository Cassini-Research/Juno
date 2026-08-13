import Combine
import Foundation

enum JunoStatsRange: String, CaseIterable, Identifiable {
    case sevenDays = "7d"
    case thirtyDays = "30d"
    case allTime = "all"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sevenDays: return "7 Days"
        case .thirtyDays: return "30 Days"
        case .allTime: return "All Time"
        }
    }
}

struct JunoStatsDisplaySnapshot: Equatable {
    let totalWords: Int
    let dictations: Int
    let timeSavedSeconds: Int

    init(
        period: StatsPeriodResponse,
        range: JunoStatsRange,
        lifetimeWords: Int,
        lifetimeDictations: Int
    ) {
        if range == .allTime {
            totalWords = max(period.totalWords, lifetimeWords)
            dictations = max(period.dictations, lifetimeDictations)
            timeSavedSeconds = Int((Double(totalWords) / 133.0 * 60.0).rounded())
        } else {
            totalWords = period.totalWords
            dictations = period.dictations
            timeSavedSeconds = period.timeSavedS
        }
    }
}

@MainActor
final class JunoStatsModel: ObservableObject {
    @Published private(set) var wordsToday: Int?
    @Published private(set) var wordsWeek: Int?
    @Published private(set) var appsToday: Int?
    @Published private(set) var timeSavedSec: Int?
    @Published private(set) var timeSavedMin: Int?
    /// Words per day for the last 7 days, oldest → today. `nil` while
    /// loading or if the broker is too old to emit the field.
    @Published private(set) var wordsByDay: [Int]?
    /// App names used today, sorted by descending frequency.
    @Published private(set) var appsTodayTop: [String]?
    /// Most-used app today, for the "most in X" caption.
    @Published private(set) var topAppToday: String?
    /// Range summaries used by the dedicated Stats page.
    @Published private(set) var periods: [StatsPeriodResponse] = []
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var lastError: String?

    func period(for range: JunoStatsRange) -> StatsPeriodResponse? {
        periods.first { $0.id == range.rawValue }
    }

    func refresh() {
        isLoading = true
        JunoBroker.fetchStatsSummary { [weak self] result in
            guard let self else { return }
            self.isLoading = false
            switch result {
            case .success(let resp):
                self.wordsToday = resp.wordsToday
                self.wordsWeek = resp.wordsWeek
                self.appsToday = resp.appsToday
                self.timeSavedSec = resp.timeSavedS
                self.timeSavedMin = resp.timeSavedMin
                self.wordsByDay = resp.wordsByDay
                self.appsTodayTop = resp.appsTodayTop
                self.topAppToday = resp.topAppToday
                self.periods = resp.periods ?? []
                self.lastError = nil
            case .failure(let err):
                self.lastError = err.localizedDescription
                self.wordsToday = nil
                self.wordsWeek = nil
                self.appsToday = nil
                self.timeSavedSec = nil
                self.timeSavedMin = nil
                self.wordsByDay = nil
                self.appsTodayTop = nil
                self.topAppToday = nil
                self.periods = []
            }
        }
    }
}
