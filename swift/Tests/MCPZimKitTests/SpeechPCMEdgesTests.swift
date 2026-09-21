import XCTest
@testable import MCPZimKit

final class SpeechPCMEdgesTests: XCTestCase {
    func testNonzeroEdgesBecomeSilentWithoutChangingInteriorOrDuration() {
        let original = (0..<4410).map { Float(sin(Double($0) * 0.031) * 0.4 + 0.1) }
        var samples = original
        samples.withUnsafeMutableBufferPointer { SpeechPCMEdges.taper($0, sampleRate: 44100) }
        XCTAssertEqual(samples.count, original.count)
        XCTAssertEqual(samples.first, 0)
        XCTAssertEqual(samples.last, 0)
        XCTAssertEqual(Array(samples[220..<(samples.count - 220)]), Array(original[220..<(samples.count - 220)]))
        for (before, after) in zip(original, samples) { XCTAssertLessThanOrEqual(abs(after), abs(before)) }
    }
    func testAdjacentChunksHaveNoSampleStepAtJoin() {
        var a = [Float](repeating: 0.3, count: 1000)
        var b = [Float](repeating: -0.4, count: 1000)
        a.withUnsafeMutableBufferPointer { SpeechPCMEdges.taper($0, sampleRate: 44100) }
        b.withUnsafeMutableBufferPointer { SpeechPCMEdges.taper($0, sampleRate: 44100) }
        XCTAssertEqual(a.last, b.first)
        XCTAssertEqual(a[500], 0.3)
        XCTAssertEqual(b[500], -0.4)
    }
    func testSilenceTinyBuffersAndInvalidRate() {
        for size in 0...7 {
            var a = [Float](repeating: 0, count: size)
            a.withUnsafeMutableBufferPointer { SpeechPCMEdges.taper($0, sampleRate: 44100) }
            XCTAssertEqual(a, [Float](repeating: 0, count: size))
        }
        var a: [Float] = [0.2, -0.3]
        a.withUnsafeMutableBufferPointer { SpeechPCMEdges.taper($0, sampleRate: 0) }
        XCTAssertEqual(a, [0.2, -0.3])
        a.withUnsafeMutableBufferPointer { SpeechPCMEdges.taper($0, sampleRate: 44100) }
        XCTAssertEqual(a, [0, 0])
    }
}
