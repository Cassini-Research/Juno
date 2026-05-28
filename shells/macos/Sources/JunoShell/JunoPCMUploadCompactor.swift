import Foundation

struct JunoPCMCompactionResult {
    let pcm: Data
    let originalBytes: Int
    let compactedBytes: Int
    let retainedRegionCount: Int
    let activeWindowCount: Int
    let sampleRate: Double

    var droppedBytes: Int {
        max(0, originalBytes - compactedBytes)
    }

    var didCompact: Bool {
        droppedBytes > 0
    }

    var originalDurationSeconds: Double {
        durationSeconds(forBytes: originalBytes)
    }

    var compactedDurationSeconds: Double {
        durationSeconds(forBytes: compactedBytes)
    }

    var droppedDurationSeconds: Double {
        durationSeconds(forBytes: droppedBytes)
    }

    private func durationSeconds(forBytes bytes: Int) -> Double {
        guard sampleRate > 0 else { return 0 }
        return Double(max(0, bytes)) / (sampleRate * 2.0)
    }
}

enum JunoPCMUploadCompactor {
    static func compact(
        _ pcm: Data,
        sampleRate: Double = 16_000,
        windowSeconds: Double = 0.10,
        preRollSeconds: Double = 0.80,
        postRollSeconds: Double = 1.80,
        bridgeGapSeconds: Double = 2.50,
        rmsThreshold: Double = 0.002,
        peakThreshold: Double = 0.018,
        minCompactedSeconds: Double = 0.35
    ) -> JunoPCMCompactionResult {
        let originalBytes = pcm.count
        let bytesPerSample = 2
        let validBytes = originalBytes - (originalBytes % bytesPerSample)
        guard validBytes > 0, sampleRate > 0, windowSeconds > 0 else {
            return unchanged(pcm, sampleRate: sampleRate)
        }

        let sampleCount = validBytes / bytesPerSample
        let windowSampleCount = max(160, Int((sampleRate * windowSeconds).rounded()))
        let windowCount = Int(ceil(Double(sampleCount) / Double(windowSampleCount)))
        guard windowCount > 0 else {
            return unchanged(pcm, sampleRate: sampleRate)
        }

        var activeWindows = Array(repeating: false, count: windowCount)
        var activeWindowCount = 0

        pcm.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            for windowIndex in 0..<windowCount {
                let sampleStart = windowIndex * windowSampleCount
                let sampleEnd = min(sampleCount, sampleStart + windowSampleCount)
                guard sampleStart < sampleEnd else { continue }

                var sumSquares = 0.0
                var peak = 0.0
                for sampleIndex in sampleStart..<sampleEnd {
                    let byteIndex = sampleIndex * bytesPerSample
                    let word = UInt16(bytes[byteIndex]) | (UInt16(bytes[byteIndex + 1]) << 8)
                    let normalized = Double(Int16(bitPattern: word)) / 32768.0
                    let magnitude = abs(normalized)
                    sumSquares += normalized * normalized
                    peak = max(peak, magnitude)
                }

                let rms = sqrt(sumSquares / Double(sampleEnd - sampleStart))
                if rms >= rmsThreshold || peak >= peakThreshold {
                    activeWindows[windowIndex] = true
                    activeWindowCount += 1
                }
            }
        }

        guard activeWindowCount > 0 else {
            return JunoPCMCompactionResult(
                pcm: pcm,
                originalBytes: originalBytes,
                compactedBytes: originalBytes,
                retainedRegionCount: 0,
                activeWindowCount: 0,
                sampleRate: sampleRate
            )
        }

        let preRollWindows = max(0, Int(ceil(preRollSeconds / windowSeconds)))
        let postRollWindows = max(0, Int(ceil(postRollSeconds / windowSeconds)))
        let bridgeGapWindows = max(0, Int(ceil(bridgeGapSeconds / windowSeconds)))
        var keepWindows = Array(repeating: false, count: windowCount)

        for (index, isActive) in activeWindows.enumerated() where isActive {
            let keepStart = max(0, index - preRollWindows)
            let keepEnd = min(windowCount - 1, index + postRollWindows)
            for keepIndex in keepStart...keepEnd {
                keepWindows[keepIndex] = true
            }
        }

        keepWindows = bridgeShortGaps(in: keepWindows, maxGapWindows: bridgeGapWindows)
        let keptRanges = contiguousKeptByteRanges(
            keepWindows: keepWindows,
            windowSampleCount: windowSampleCount,
            validBytes: validBytes
        )
        guard !keptRanges.isEmpty else {
            return unchanged(pcm, sampleRate: sampleRate)
        }

        var compacted = Data()
        compacted.reserveCapacity(keptRanges.reduce(0) { $0 + ($1.upperBound - $1.lowerBound) })
        for range in keptRanges {
            compacted.append(pcm.subdata(in: range))
        }

        let minCompactedBytes = Int((sampleRate * minCompactedSeconds).rounded()) * bytesPerSample
        guard compacted.count >= minCompactedBytes, compacted.count < originalBytes else {
            return JunoPCMCompactionResult(
                pcm: pcm,
                originalBytes: originalBytes,
                compactedBytes: originalBytes,
                retainedRegionCount: keptRanges.count,
                activeWindowCount: activeWindowCount,
                sampleRate: sampleRate
            )
        }

        return JunoPCMCompactionResult(
            pcm: compacted,
            originalBytes: originalBytes,
            compactedBytes: compacted.count,
            retainedRegionCount: keptRanges.count,
            activeWindowCount: activeWindowCount,
            sampleRate: sampleRate
        )
    }

    private static func unchanged(_ pcm: Data, sampleRate: Double) -> JunoPCMCompactionResult {
        JunoPCMCompactionResult(
            pcm: pcm,
            originalBytes: pcm.count,
            compactedBytes: pcm.count,
            retainedRegionCount: 0,
            activeWindowCount: 0,
            sampleRate: sampleRate
        )
    }

    private static func bridgeShortGaps(in keepWindows: [Bool], maxGapWindows: Int) -> [Bool] {
        guard maxGapWindows > 0 else { return keepWindows }
        var bridged = keepWindows
        var index = 0
        var previousEnd: Int?

        while index < bridged.count {
            while index < bridged.count, !bridged[index] {
                index += 1
            }
            guard index < bridged.count else { break }

            let regionStart = index
            while index < bridged.count, bridged[index] {
                index += 1
            }
            let regionEnd = index - 1

            if let previousEnd {
                let gap = regionStart - previousEnd - 1
                if gap > 0, gap <= maxGapWindows {
                    for gapIndex in (previousEnd + 1)..<regionStart {
                        bridged[gapIndex] = true
                    }
                }
            }
            previousEnd = regionEnd
        }

        return bridged
    }

    private static func contiguousKeptByteRanges(
        keepWindows: [Bool],
        windowSampleCount: Int,
        validBytes: Int
    ) -> [Range<Int>] {
        let bytesPerSample = 2
        var ranges: [Range<Int>] = []
        var index = 0

        while index < keepWindows.count {
            while index < keepWindows.count, !keepWindows[index] {
                index += 1
            }
            guard index < keepWindows.count else { break }

            let regionStartWindow = index
            while index < keepWindows.count, keepWindows[index] {
                index += 1
            }
            let regionEndWindow = index

            let startByte = min(validBytes, regionStartWindow * windowSampleCount * bytesPerSample)
            let endByte = min(validBytes, regionEndWindow * windowSampleCount * bytesPerSample)
            if startByte < endByte {
                ranges.append(startByte..<endByte)
            }
        }

        return ranges
    }
}
