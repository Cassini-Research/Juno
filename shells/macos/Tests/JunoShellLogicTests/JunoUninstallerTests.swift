import XCTest
@testable import JunoShell

final class JunoUninstallerTests: XCTestCase {
    private let home = URL(fileURLWithPath: "/Users/example")

    func testPlanCoversAllDataLocations() {
        let plan = JunoUninstaller.plan(
            extraRepoIds: [],
            home: home,
            bundleURL: URL(fileURLWithPath: "/Applications/Juno.app"),
            hfHome: nil
        )
        let paths = plan.dataPaths.map(\.path)
        XCTAssertTrue(paths.contains("/Users/example/Library/Application Support/com.juno.shell"))
        XCTAssertTrue(paths.contains("/Users/example/Library/Application Support/Juno"))
        XCTAssertTrue(paths.contains("/Users/example/Library/Logs/Juno"))
        XCTAssertTrue(paths.contains("/Users/example/Library/Caches/com.juno.shell"))
        XCTAssertTrue(paths.contains("/Users/example/Library/Saved Application State/com.juno.shell.savedState"))
    }

    func testModelPathsUseHFCacheNamingAndDefaultRepos() {
        let plan = JunoUninstaller.plan(
            extraRepoIds: [],
            home: home,
            bundleURL: URL(fileURLWithPath: "/Applications/Juno.app"),
            hfHome: nil
        )
        XCTAssertTrue(plan.modelPaths.map(\.path).contains(
            "/Users/example/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo"
        ))
        XCTAssertEqual(plan.modelPaths.count, JunoUninstaller.knownModelRepoIds.count)
    }

    func testExtraRepoIdsMergeWithoutDuplicatesAndIgnoreJunk() {
        let plan = JunoUninstaller.plan(
            extraRepoIds: [
                "mlx-community/whisper-large-v3-turbo", // duplicate of a known repo
                "custom-org/custom-model",
                "",            // junk
                "not-a-repo",  // junk: no org/name separator
            ],
            home: home,
            bundleURL: URL(fileURLWithPath: "/Applications/Juno.app"),
            hfHome: nil
        )
        let paths = plan.modelPaths.map(\.path)
        XCTAssertEqual(plan.modelPaths.count, JunoUninstaller.knownModelRepoIds.count + 1)
        XCTAssertTrue(paths.contains(
            "/Users/example/.cache/huggingface/hub/models--custom-org--custom-model"
        ))
    }

    func testHFHomeOverrideIsRespected() {
        let dir = JunoUninstaller.hfModelCacheDir(
            repoId: "org/model",
            hfHome: "/Volumes/Big/hf",
            home: home
        )
        XCTAssertEqual(dir.path, "/Volumes/Big/hf/hub/models--org--model")
    }

    func testAppBundleOnlyRemovableFromApplications() {
        let installed = JunoUninstaller.plan(
            extraRepoIds: [],
            home: home,
            bundleURL: URL(fileURLWithPath: "/Applications/Juno.app"),
            hfHome: nil
        )
        XCTAssertEqual(installed.appBundle?.path, "/Applications/Juno.app")

        // Dev builds (repo checkout, dist/, mounted DMG) must never
        // delete themselves.
        for devPath in [
            "/Users/example/code/lola/Juno/dist/Juno.app",
            "/Volumes/Juno/Juno.app",
        ] {
            let plan = JunoUninstaller.plan(
                extraRepoIds: [],
                home: home,
                bundleURL: URL(fileURLWithPath: devPath),
                hfHome: nil
            )
            XCTAssertNil(plan.appBundle, "unexpected self-removal for \(devPath)")
        }
    }

    func testLaunchAgentPlistsCoverCurrentAndLegacyLabels() {
        let plan = JunoUninstaller.plan(
            extraRepoIds: [],
            home: home,
            bundleURL: URL(fileURLWithPath: "/Applications/Juno.app"),
            hfHome: nil
        )
        let names = plan.launchAgentPlists.map(\.lastPathComponent)
        XCTAssertEqual(
            Set(names),
            Set(["com.juno.voice-engine.plist", "com.juno.launch.plist", "com.juno.shell.agent.plist"])
        )
    }
}
