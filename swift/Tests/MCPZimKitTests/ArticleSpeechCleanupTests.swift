// SPDX-License-Identifier: MIT
//
// `ArticleSections.stripHTML` must remove pronunciation/coordinate noise
// that reads as symbol-by-symbol gibberish through TTS. These fixtures are
// the real Kiwix/Wikipedia lead markup shapes (verified against
// wikipedia_en_top_nopic): IPA wrapped in `rt-commentedText` with a
// per-character `<span title>` tree, the `ext-phonos` ⓘ "listen" button,
// inline `geo` coordinate spans, and bracketed foreign-language IPA.
//
// Contract: pure phonetic marks (ˈ ˌ ː, foreign IPA letters), the ⓘ glyph,
// and inline coordinates are gone; the running prose and the human-readable
// pronunciation RESPELLING (e.g. "nə-VAD-ə") survive — TTS reads those fine.

import XCTest
@testable import MCPZimKit

final class ArticleSpeechCleanupTests: XCTestCase {

    private func stripIPAchars(_ s: String) -> Bool {
        // Pure-phonetic marks that never appear in a respelling.
        for ch in ["ˈ", "ˌ", "ː", "ʃ", "ʒ", "θ", "ð", "β", "ʁ", "ɲ"] {
            if s.contains(ch) { return false }
        }
        return true
    }

    func testNestedIPAClusterRemoved() {
        // California: IPA as a per-character span tree inside rt-commentedText.
        let html = """
        <p><b>California</b> (<span class="rt-commentedText nowrap">\
        <span class="IPA nopopups noexcerpt" lang="en-fonipa">/\
        <span style="border-bottom:1px dotted">\
        <span title="a">ˌ</span><span title="b">k</span><span title="c">æ</span>\
        <span title="d">l</span><span title="e">ɪ</span><span title="f">ˈ</span>\
        <span title="g">f</span><span title="h">ɔːr</span><span title="i">n</span>\
        <span title="j">i</span><span title="k">ə</span></span>/</span></span>) \
        is a state in the Western United States.</p>
        """
        let out = ArticleSections.stripHTML(html)
        XCTAssertTrue(out.contains("is a state in the Western United States"),
                      "real prose must survive: \(out)")
        XCTAssertTrue(stripIPAchars(out), "IPA stress/length marks must be gone: \(out)")
        XCTAssertFalse(out.contains("( )"), "emptied pronunciation parens tidied: \(out)")
        XCTAssertFalse(out.contains("ɔːr"), "IPA glyphs gone: \(out)")
    }

    func testListenButtonForeignIPAAndRespelling() {
        // Nevada: IPA + ext-phonos ⓘ + bracketed Spanish IPA + respelling.
        let html = """
        <p><b>Nevada</b> (<span class="rt-commentedText nowrap">\
        <span class="IPA" lang="en-fonipa">/nəˈvædə/</span> \
        <span class="ext-phonos"><span class="ext-phonos-PhonosButton">ⓘ</span></span>\
        </span> nə-VAD-ə; Spanish: [neˈβaða]) is a landlocked state.</p>
        """
        let out = ArticleSections.stripHTML(html)
        XCTAssertTrue(out.contains("is a landlocked state"), "prose survives: \(out)")
        XCTAssertFalse(out.contains("ⓘ"), "listen glyph gone: \(out)")
        XCTAssertFalse(out.contains("/nəˈvædə/"), "slash IPA gone: \(out)")
        XCTAssertFalse(out.contains("neˈβaða"), "bracketed foreign IPA gone: \(out)")
        XCTAssertFalse(out.contains("Spanish:"), "dangling language label tidied: \(out)")
        // The human respelling is acceptable for TTS and may remain.
        XCTAssertTrue(out.contains("VAD"), "respelling preserved: \(out)")
    }

    func testInlineCoordinatesRemoved() {
        let html = "<p>The summit sits at <span class=\"geo\">37.77750; -122.41639</span> above the bay.</p>"
        let out = ArticleSections.stripHTML(html)
        XCTAssertFalse(out.contains("37.77750"), "inline coords gone: \(out)")
        XCTAssertTrue(out.contains("above the bay"), "prose survives: \(out)")
    }

    func testGeoTokenDoesNotEatGeography() {
        // Token match, not substring — a "geography" class span must stay.
        let html = "<p>California has diverse <span class=\"geography-note\">geography spanning coast to desert</span>.</p>"
        let out = ArticleSections.stripHTML(html)
        XCTAssertTrue(out.contains("geography spanning coast to desert"),
                      "non-geo class must NOT be removed: \(out)")
    }

    func testListenWordAndPronunciationLabel() {
        let html = "<p><b>Oahu</b> (English pronunciation: oh-AH-hoo (listen)) is an island.</p>"
        let out = ArticleSections.stripHTML(html)
        XCTAssertFalse(out.contains("(listen)"), "(listen) gone: \(out)")
        XCTAssertFalse(out.lowercased().contains("english pronunciation"), "label gone: \(out)")
        XCTAssertTrue(out.contains("is an island"), "prose survives: \(out)")
    }
}
