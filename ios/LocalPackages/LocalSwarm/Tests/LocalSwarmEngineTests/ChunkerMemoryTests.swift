import XCTest
import Darwin
@testable import LocalSwarmEngine

/// Guards against the jetsam (out-of-memory) regression: chunking a large file
/// must stream, not accumulate the whole file in RAM.
final class ChunkerMemoryTests: XCTestCase {
    private func residentBytes() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }

    func testChunkingLargeFileStaysMemoryBounded() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ls-mem-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("big.bin")

        // Build a 256 MB file by appending 1 MB blocks (never allocate it whole).
        let megabytes = 256
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let writer = try FileHandle(forWritingTo: url)
        let block = Data(repeating: 0xAB, count: 1 << 20)
        for _ in 0..<megabytes { try writer.write(contentsOf: block) }
        try writer.close()

        let before = residentBytes()
        let (manifest, _) = try Chunker.buildManifest(name: "big", fileURLs: [url], chunkSize: 1 << 20)
        let growthMB = Double(residentBytes() - before) / 1_000_000

        XCTAssertEqual(manifest.chunkCount, megabytes)
        // With per-chunk autorelease draining, growth is a few MB. Without it,
        // growth would be ~256 MB. 100 MB threshold cleanly separates the two.
        XCTAssertLessThan(growthMB, 100,
                          "buildManifest grew RSS by \(Int(growthMB)) MB — autorelease accumulation regressed")
    }

    func testProgressReportsMonotonicallyAndCompletes() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ls-prog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("file.bin")

        // 200 chunks so the every-64th throttle still emits several callbacks.
        let megabytes = 200
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let writer = try FileHandle(forWritingTo: url)
        let block = Data(repeating: 0xCD, count: 1 << 20)
        for _ in 0..<megabytes { try writer.write(contentsOf: block) }
        try writer.close()

        // Thread-safe collector: the progress callback is @Sendable, so it can't
        // capture a mutable local (an error under Swift 6).
        let collector = FractionCollector()
        _ = try Chunker.buildManifest(name: "f", fileURLs: [url], chunkSize: 1 << 20,
                                      progress: { collector.add($0) })
        let fractions = collector.snapshot

        XCTAssertFalse(fractions.isEmpty, "progress was never reported")
        XCTAssertEqual(fractions, fractions.sorted(), "progress went backwards")
        XCTAssertEqual(fractions.last ?? 0, 1.0, accuracy: 0.0001,
                       "progress did not reach 100% (\(fractions.last ?? -1))")
    }
}

/// Thread-safe accumulator for values delivered through a `@Sendable` closure.
private final class FractionCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []
    func add(_ value: Double) { lock.lock(); values.append(value); lock.unlock() }
    var snapshot: [Double] { lock.lock(); defer { lock.unlock() }; return values }
}
