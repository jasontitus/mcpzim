// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class StreamingSpeechPolicyTests: XCTestCase {
    func testCompleteSentenceStartsBeforeLaterText() {
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            "The first sentence is ready. The second is still growing",
            generating: true,
            allowEarlyClause: true)
        XCTAssertEqual(result?.text, "The first sentence is ready.")
        XCTAssertEqual(result?.boundary, .sentence)
        XCTAssertEqual(result?.consumedCharacters,
                       "The first sentence is ready.".count)
    }

    func testEarlyClauseStartsLongSentence() {
        let first = String(repeating: "word ", count: 14) + ","
        let text = first + " and the answer continues with more detail"
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: true, allowEarlyClause: true)
        XCTAssertEqual(result?.text, first)
        XCTAssertEqual(result?.boundary, .clause)
    }

    func testShortCommaDoesNotCreateTinyUtterance() {
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            "Putin, a Russian politician whose career continued",
            generating: true,
            allowEarlyClause: true)
        XCTAssertNil(result)
    }

    func testAbbreviationDoesNotCreateFalseSentence() {
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            "Dr. Vladimir Putin served in the intelligence service",
            generating: true,
            allowEarlyClause: true)
        XCTAssertNil(result)

        let initial = StreamingSpeechPolicy.takeSpeakablePrefix(
            "V. Putin served in the intelligence service",
            generating: true,
            allowEarlyClause: true)
        XCTAssertNil(initial)
    }

    func testSoftWrapBoundsUnpunctuatedProse() {
        let text = String(repeating: "generated prose keeps growing ", count: 10)
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: true, allowEarlyClause: true,
            minimumClause: 64, maximumClause: 160)
        XCTAssertEqual(result?.boundary, .softWrap)
        XCTAssertGreaterThanOrEqual(result?.consumedCharacters ?? 0, 64)
        XCTAssertLessThanOrEqual(result?.consumedCharacters ?? .max, 160)
    }

    func testPressureCanDisableOnlyEagerOverlap() {
        XCTAssertFalse(StreamingSpeechPolicy.allowsEagerSynthesis(
            availableMemoryMB: 3_100, estimatedTTSMemoryMB: 2_800))
        XCTAssertTrue(StreamingSpeechPolicy.allowsEagerSynthesis(
            availableMemoryMB: 3_200, estimatedTTSMemoryMB: 2_800))
        XCTAssertTrue(StreamingSpeechPolicy.allowsEagerSynthesis(
            availableMemoryMB: 0, estimatedTTSMemoryMB: 2_800))
    }

    func testThermalPressureDoesNotReplaceLowMemoryANEVoice() {
        XCTAssertFalse(StreamingSpeechPolicy.requiresLightweightVoiceFallback(
            availableMemoryMB: 2_300,
            estimatedTTSMemoryMB: 96,
            thermallyConstrained: true))
    }

    func testHeavyMLXVoiceFallsBackUnderThermalOrMemoryPressure() {
        XCTAssertTrue(StreamingSpeechPolicy.requiresLightweightVoiceFallback(
            availableMemoryMB: 5_000,
            estimatedTTSMemoryMB: 2_800,
            thermallyConstrained: true))
        XCTAssertTrue(StreamingSpeechPolicy.requiresLightweightVoiceFallback(
            availableMemoryMB: 3_100,
            estimatedTTSMemoryMB: 2_800,
            thermallyConstrained: false))
        XCTAssertFalse(StreamingSpeechPolicy.requiresLightweightVoiceFallback(
            availableMemoryMB: 3_600,
            estimatedTTSMemoryMB: 2_800,
            thermallyConstrained: false))
    }

    func testCompletedGenerationFlushesWholeTail() {
        let text = "A final fragment without punctuation"
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: false, allowEarlyClause: false)
        XCTAssertEqual(result?.text, text)
        XCTAssertEqual(result?.consumedCharacters, text.count)
        XCTAssertEqual(result?.boundary, .final)
    }

    func testParagraphOfferGetsItsOwnProsodyWindow() {
        let answer = "His ideas influenced global revolutions and political thought."
        let offer = "Want to hear about revolutionary socialism or class struggle?"
        let text = answer + "\n\n" + offer
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: false, allowEarlyClause: true,
            minimumClause: 180, maximumClause: 360)

        XCTAssertEqual(result?.text, answer)
        XCTAssertEqual(result?.consumedCharacters, answer.count + 2)
        XCTAssertEqual(result?.boundary, .sentence)
        XCTAssertEqual(
            String(text.dropFirst(result?.consumedCharacters ?? 0)),
            offer)
    }

    func testCompletedSentencesInSameParagraphKeepLongWindow() {
        let text = "The first sentence is complete. The second stays in the same paragraph."
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: false, allowEarlyClause: true,
            minimumClause: 180, maximumClause: 360)

        XCTAssertEqual(result?.text, text)
        XCTAssertEqual(result?.consumedCharacters, text.count)
        XCTAssertEqual(result?.boundary, .final)
    }

    func testLongSentenceHonorsBackendCharacterCap() {
        let text = "Doctor Vladimir Putin attended Leningrad State University before beginning a long career in government service."
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: true, allowEarlyClause: true,
            minimumClause: 40, maximumClause: 68)
        XCTAssertEqual(result?.boundary, .softWrap)
        XCTAssertLessThanOrEqual(result?.consumedCharacters ?? .max, 68)
        XCTAssertFalse(result?.text.hasSuffix(".") ?? true)
    }

    func testCompletedLongTailDrainsInBoundedChunks() {
        let text = String(repeating: "unpunctuated historical detail ", count: 8)
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: false, allowEarlyClause: true,
            minimumClause: 40, maximumClause: 68)
        XCTAssertEqual(result?.boundary, .softWrap)
        XCTAssertGreaterThan(result?.consumedCharacters ?? 0, 0)
        XCTAssertLessThanOrEqual(result?.consumedCharacters ?? .max, 68)
        XCTAssertLessThan(result?.consumedCharacters ?? .max, text.count)
    }

    func testSingleLongTokenStillHonorsBackendCharacterCap() {
        let text = String(repeating: "x", count: 100)
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            text, generating: false, allowEarlyClause: true,
            minimumClause: 40, maximumClause: 68)
        XCTAssertEqual(result?.boundary, .softWrap)
        XCTAssertEqual(result?.consumedCharacters, 68)
    }

    func testWashingtonExcerptUsesValidatedSupertonicWindow() {
        var remaining = "George Washington (February 22, 1732 – December 14, 1799) was a Founding Father and the first president of the United States, serving from 1789 to 1797. As commander of the Continental Army, he led Patriot forces to victory in the American Revolutionary War against the British Empire."
        var chunks: [StreamingSpeechPrefix] = []
        while !remaining.isEmpty {
            guard let prefix = StreamingSpeechPolicy.takeSpeakablePrefix(
                remaining, generating: false, allowEarlyClause: true,
                minimumClause: 56, maximumClause: 94)
            else {
                XCTFail("Expected a bounded prefix for the remaining excerpt")
                return
            }
            chunks.append(prefix)
            remaining = String(remaining.dropFirst(prefix.consumedCharacters))
        }

        XCTAssertEqual(chunks.count, 4)
        XCTAssertTrue(chunks.allSatisfy { $0.text.count <= 94 })
        XCTAssertEqual(chunks.map(\.text).joined(),
                       "George Washington (February 22, 1732 – December 14, 1799) was a Founding Father and the first president of the United States, serving from 1789 to 1797. As commander of the Continental Army, he led Patriot forces to victory in the American Revolutionary War against the British Empire.")
    }

    func testLongKokoroWindowKeepsCompleteSentenceTogether() {
        let sentence = "George Washington was a Founding Father and the first president of the United States, serving from 1789 to 1797."
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            sentence + " More text is still being generated",
            generating: true,
            allowEarlyClause: true,
            minimumClause: 180,
            maximumClause: 360)

        XCTAssertEqual(result?.text, sentence)
        XCTAssertEqual(result?.boundary, .sentence)
    }

    func testLongKokoroWindowIgnoresEarlyMidSentenceComma() {
        let early = String(repeating: "word ", count: 22) + ","
        let later = String(repeating: "detail ", count: 12) + ","
        let result = StreamingSpeechPolicy.takeSpeakablePrefix(
            early + " " + later + " and the sentence is still growing",
            generating: true,
            allowEarlyClause: true,
            minimumClause: 180,
            maximumClause: 360)

        XCTAssertEqual(result?.text, early + " " + later)
        XCTAssertEqual(result?.boundary, .clause)
        XCTAssertGreaterThan(result?.consumedCharacters ?? 0, early.count)
    }
}
