// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class AnswerAttributionTests: XCTestCase {

    private let passages = [
        AnswerAttribution.Passage(
            article: "Apple TV (device)", section: "lead",
            text: """
            Apple TV is a digital media player and microconsole developed by \
            Apple. It is a small network appliance that sends received media \
            data such as video and audio to a television. The first \
            generation was announced in September 2006 and shipped in \
            March 2007. The device runs the tvOS operating system.
            """),
        AnswerAttribution.Passage(
            article: "Apple TV (device)", section: "4K (third generation)",
            text: """
            The third-generation Apple TV 4K was announced on October 18, \
            2022 and released on November 4, 2022, powered by the A15 \
            Bionic chip.
            """),
    ]

    func testSupportedSentencesAttributeToRightPassage() {
        let answer = "Apple TV is a digital media player and microconsole developed by Apple. "
            + "The third-generation Apple TV 4K was announced on October 18, 2022 and released on November 4, 2022, powered by the A15 Bionic chip."
        let attrs = AnswerAttribution.attribute(answer: answer, passages: passages)
        XCTAssertEqual(attrs.count, 2)
        XCTAssertEqual(attrs[0].passageIndex, 0)
        XCTAssertEqual(attrs[1].passageIndex, 1, "date+chip sentence must bind the 4K section")
        XCTAssertTrue(attrs.allSatisfy(\.isSupported))
    }

    func testHallucinatedDateIsUnsupported() {
        // "launched September 15, 2011" — the real capture 2026-07-19
        // confabulation. Nouns match the lead but the numbers don't exist
        // in any passage; the 3× numeric weighting must sink it.
        let answer = "The Apple TV was launched on September 15, 2011."
        let attrs = AnswerAttribution.attribute(answer: answer, passages: passages)
        XCTAssertEqual(attrs.count, 1)
        XCTAssertFalse(attrs[0].isSupported,
                       "invented date must not attribute (support=\(attrs[0].support))")
    }

    func testHallucinatedSingleDigitCountIsUnsupported() {
        let evidence = [AnswerAttribution.Passage(
            article: "Trip", section: "lead", text: "The trip took 7 days.")]
        let attrs = AnswerAttribution.attribute(
            answer: "The trip took 5 days.", passages: evidence)
        XCTAssertEqual(attrs.count, 1)
        XCTAssertFalse(attrs[0].isSupported,
                       "single-digit numeric contradictions must affect grounding")
    }

    func testTrainedDataFactIsUnsupported() {
        // True in the world, absent from the passages — exactly what the
        // user wants surfaced as "not from the offline library".
        let answer = "Tim Cook announced the product at a keynote in Cupertino alongside the iPhone 15."
        let attrs = AnswerAttribution.attribute(answer: answer, passages: passages)
        XCTAssertFalse(attrs[0].isSupported)
    }

    func testShortUnsourcedPhrasesAreNotGivenAFalseSourceMatch() {
        let attrs = AnswerAttribution.attribute(
            answer: "Sure! Of course. Happy to help.", passages: passages)
        XCTAssertTrue(attrs.allSatisfy { !$0.isSupported })
    }

    func testEvenPlausibleParaphraseRequiresSourceRendering() {
        let answer = "The first generation shipped in March 2007 after being announced in September 2006."
        let attrs = AnswerAttribution.attribute(answer: answer, passages: passages)
        XCTAssertNil(attrs[0].passageIndex)
    }

    func testLogLineFormat() {
        let answer = "The device runs the tvOS operating system. It was invented in 1877 by Thomas Edison."
        let attrs = AnswerAttribution.attribute(answer: answer, passages: passages)
        let line = AnswerAttribution.logLine(attrs, passages: passages)
        XCTAssertTrue(line.contains("s1→Apple TV (device)§lead"))
        XCTAssertTrue(line.contains("s2→UNSUPPORTED"))
    }

    func testEmptyPassagesCannotApproveAnAnswer() {
        XCTAssertFalse(AnswerAttribution.attribute(answer: "Anything.", passages: [])[0].isSupported)
    }

    func testNounOverlapCannotApproveChangedMeaning() {
        let evidence = [AnswerAttribution.Passage(article: "Synthetic fixture", section: "lead",
            text: "Ada did not approve Ben's plan. Ada paid Ben 10 dollars and Ben paid Cara 20 dollars. The treatment may help some patients. The duet features two singers performing simultaneously.")]
        for answer in [
            "Ada did approve Ben's plan.",
            "Ben did not approve Ada's plan.",
            "Ada paid Ben 20 dollars and Ben paid Cara 10 dollars.",
            "The treatment will help some patients.",
            "The duet features two singers taking turns performing solo sections.",
        ] {
            XCTAssertTrue(AnswerAttribution.attribute(answer: answer, passages: evidence)
                .allSatisfy { !$0.isSupported }, answer)
        }
    }
}
