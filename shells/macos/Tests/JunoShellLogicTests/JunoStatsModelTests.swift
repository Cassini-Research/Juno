import XCTest
@testable import JunoShell

final class JunoStatsModelTests: XCTestCase {
    func testExtendedStatsSummaryDecodesPeriodPayload() throws {
        let json = """
        {
          "ok": true,
          "words_today": 120,
          "words_week": 720,
          "apps_today": 2,
          "time_saved_s": 54,
          "time_saved_min": 0,
          "computed_at_unix_ms": 1785265200000,
          "words_by_day": [0, 100, 80, 90, 110, 220, 120],
          "apps_today_top": ["Notes", "Mail"],
          "top_app_today": "Notes",
          "periods": [{
            "id": "all",
            "total_words": 1200,
            "dictations": 18,
            "time_saved_s": 541,
            "bucket_start_dates": ["2026-06-01", "2026-07-01"],
            "bucket_end_dates": ["2026-06-30", "2026-07-28"],
            "words_by_bucket": [500, 700],
            "top_apps": [{"name": "Notes", "words": 800}]
          }]
        }
        """

        let response = try BrokerDecode.decoder.decode(
            StatsSummaryResponse.self,
            from: Data(json.utf8)
        )

        let period = try XCTUnwrap(response.periods?.first)
        XCTAssertEqual(period.id, "all")
        XCTAssertEqual(period.totalWords, 1_200)
        XCTAssertEqual(period.wordsByBucket, [500, 700])
        XCTAssertEqual(period.topApps.first?.name, "Notes")
    }

    func testAllTimeSnapshotKeepsTrueLifetimeCounters() {
        let period = StatsPeriodResponse(
            id: "all",
            totalWords: 1_200,
            dictations: 18,
            timeSavedS: 541,
            bucketStartDates: ["2026-06-01"],
            bucketEndDates: ["2026-07-28"],
            wordsByBucket: [1_200],
            topApps: []
        )

        let snapshot = JunoStatsDisplaySnapshot(
            period: period,
            range: .allTime,
            lifetimeWords: 2_000,
            lifetimeDictations: 24
        )

        XCTAssertEqual(snapshot.totalWords, 2_000)
        XCTAssertEqual(snapshot.dictations, 24)
        XCTAssertEqual(snapshot.timeSavedSeconds, 902)
    }

    func testShortRangesUseBrokerValues() {
        let period = StatsPeriodResponse(
            id: "7d",
            totalWords: 300,
            dictations: 4,
            timeSavedS: 135,
            bucketStartDates: [],
            bucketEndDates: [],
            wordsByBucket: [],
            topApps: []
        )

        let snapshot = JunoStatsDisplaySnapshot(
            period: period,
            range: .sevenDays,
            lifetimeWords: 9_999,
            lifetimeDictations: 999
        )

        XCTAssertEqual(snapshot.totalWords, 300)
        XCTAssertEqual(snapshot.dictations, 4)
        XCTAssertEqual(snapshot.timeSavedSeconds, 135)
    }
}
