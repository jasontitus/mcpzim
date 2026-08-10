// SPDX-License-Identifier: MIT
//
// Regression coverage for discuss-mode passage retrieval, built from the
// REAL failure on device (2026-07-01, LFM2.5 FT + full English Wikipedia):
// discussing "Putin", the question "What about his parents?" retrieved
// Pets | Sports | Cult of personality, and "How about his time in
// Germany?" missed the KGB/Dresden section. Root cause: ranking was pure
// HashingEmbedder n-gram cosine — hash noise on short strings — and
// `questionKeywords` let "his" through as a keyword.

import XCTest
@testable import MCPZimKit

final class DiscussRetrievalTests: XCTestCase {
    func testChronologicalContinuationInheritsPreviousFacet() {
        let resolved = ArticleHeuristics.contextualizedDiscussionQuestion(
            "Then what happened in Soviet times",
            previousQuestion: "Tell me about Buddhism there")
        XCTAssertTrue(resolved.contains("Soviet times"))
        XCTAssertTrue(resolved.hasSuffix("— buddhism"))
        XCTAssertEqual(
            Set(ArticleHeuristics.questionKeywords(resolved)),
            Set(["soviet", "times", "buddhism"]))
    }

    func testExplicitTopicHandoffDoesNotInheritPreviousFacet() {
        XCTAssertEqual(
            ArticleHeuristics.contextualizedDiscussionQuestion(
                "Then tell me about Donald Trump",
                previousQuestion: "Tell me about Buddhism there"),
            "Then tell me about Donald Trump")
        XCTAssertEqual(
            ArticleHeuristics.contextualizedDiscussionQuestion(
                "What is the population?",
                previousQuestion: "Tell me about Buddhism there"),
            "What is the population?")
    }


    /// Section titles from the actual Putin article (as logged on
    /// device), with condensed but realistic bodies.
    private var putinSections: [ArticleSection] {
        func s(_ title: String, _ text: String) -> ArticleSection {
            ArticleSection(title: title, level: 2, text: text)
        }
        return [
            ArticleSection(title: "", level: 0, text:
                "Vladimir Vladimirovich Putin is a Russian politician and former "
                + "intelligence officer who has served as President of Russia since 2012."),
            s("Early life", """
                Putin was born on 7 October 1952 in Leningrad. His mother, Maria \
                Ivanovna Putina, was a factory worker, and his father, Vladimir \
                Spiridonovich Putin, was a conscript in the Soviet Navy. His \
                grandfather was a personal cook to Lenin and Stalin. Two brothers \
                died before his birth, one of diphtheria during the siege of \
                Leningrad. He studied law at Leningrad State University.
                """),
            s("Intelligence career", """
                Putin joined the KGB in 1975. From 1985 to 1990 he served in \
                Dresden, East Germany, using a cover identity as a translator. \
                After the fall of the Berlin Wall he returned from Germany to \
                Leningrad. His KGB career ended with the rank of lieutenant colonel.
                """),
            s("Political career", """
                In 1991 Putin joined the office of the Saint Petersburg mayor. He \
                rose through the Kremlin administration, became acting president \
                in 1999, and won the 2000 presidential election.
                """),
            s("After the 2022 invasion of Ukraine", """
                Putin launched a full-scale invasion of Ukraine in February 2022. \
                He had annexed Crimea in 2014, declaring in a speech that Crimea \
                has always been an inseparable part of Russia. Sanctions followed \
                the annexation of Crimea and the invasion.
                """),
            s("Personal life", """
                Putin married Lyudmila Shkrebneva in 1983; they have two daughters, \
                Maria and Katerina. The couple announced their divorce in 2013.
                """),
            s("Pets", """
                Putin has received several dogs from foreign leaders, including a \
                Bulgarian shepherd and an Akita. His labrador Koni attended meetings.
                """),
            s("Sports", """
                Putin practises judo and ice hockey. He holds a black belt and has \
                co-authored a book on judo.
                """),
            s("Cult of personality", """
                Russian media have cultivated an image of Putin as a strong \
                outdoorsman, publishing photographs of shirtless horseback riding.
                """),
            s("Assessments", """
                Scholars describe Putin's rule as authoritarian, marked by the \
                suppression of political opposition and independent media.
                """),
            s("Awards and honours", """
                Putin has received numerous state decorations from Russia and \
                foreign governments.
                """),
        ]
    }

    private func topTitles(_ question: String, k: Int = 3) -> [String] {
        ArticleHeuristics.rankSectionsMultiSource(
            question, sources: [(title: "Putin", sections: putinSections)], k: k
        ).map { $0.section.title.isEmpty ? "lead" : $0.section.title }
    }

    func testParentsQuestionRetrievesEarlyLife() {
        let top = topTitles("What about his parents?")
        XCTAssertEqual(top.first, "Early life", "got: \(top)")
        XCTAssertFalse(top.contains("Pets"), "Pets outranked family: \(top)")
        XCTAssertFalse(top.contains("Sports"), "got: \(top)")
    }

    func testGermanyQuestionRetrievesIntelligenceCareer() {
        let top = topTitles("How about his time in Germany?")
        XCTAssertEqual(top.first, "Intelligence career", "got: \(top)")
    }

    func testCrimeaQuestionRetrievesUkraineSection() {
        let top = topTitles("What has he said about crimea?")
        XCTAssertEqual(top.first, "After the 2022 invasion of Ukraine", "got: \(top)")
    }

    func testWifeQuestionRetrievesPersonalLife() {
        let top = topTitles("Was he ever married?")
        XCTAssertEqual(top.first, "Personal life", "got: \(top)")
    }

    func testDogQuestionStillFindsPets() {
        // Keyword scoring must not break on-topic quirky questions.
        let top = topTitles("Does he have any dogs?")
        XCTAssertEqual(top.first, "Pets", "got: \(top)")
    }

    func testKeywordlessQuestionStillReturnsSomething() {
        let top = topTitles("Tell me more")
        XCTAssertFalse(top.isEmpty)
    }

    func testQuestionKeywordsDropPronounsAndFiller() {
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("What about his parents?"),
            ["parents"])
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("How about his time in Germany?"),
            ["germany"])
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("What has he said about crimea?"),
            ["crimea"])
    }

    func testNestedNavboxTablesFullyStripped() {
        // Maxi builds nest tables 3 deep inside navbox/sidebar templates.
        // The old non-greedy strip cut at the FIRST </table>, leaking the
        // inner link farm into prose (device capture 2026-07-02: a
        // "gravitational waves" answer that was 2 kB of looping physicist
        // names). Depth-aware removal must eat the whole thing.
        let html = """
        <p>Gravitational waves are ripples in spacetime.</p>
        <div class="navbox"><table class="navbox-inner"><tr><td>
        <table class="navbox-subgroup"><tr><td>
        <a href="Kerr_metric">Kerr</a> <a href="Taub">Taub</a>
        </td></tr></table>
        <table class="navbox-subgroup"><tr><td>
        <a href="Hulse">Hulse</a> <a href="Wheeler">Wheeler</a>
        </td></tr></table>
        </td></tr></table></div>
        <p>They were predicted by Einstein in 1916.</p>
        """
        let text = ArticleSections.stripHTML(html)
        XCTAssertTrue(text.contains("ripples in spacetime"), "got: \(text)")
        XCTAssertTrue(text.contains("predicted by Einstein in 1916"), "got: \(text)")
        XCTAssertFalse(text.contains("Kerr"), "navbox leaked: \(text)")
        XCTAssertFalse(text.contains("Wheeler"), "navbox leaked: \(text)")
    }

    func testHatnoteDisambiguationExtraction() {
        // Real markup shape from A/Gravity_wave (full-wiki nopic build):
        // the FIRST hatnote is the cross-meaning; "Further information:"
        // and "See also:" hatnotes are section cross-refs, not meanings.
        let html = """
        <div class="hatnote navigation-not-searchable">For the phenomenon \
        of general relativity, see <a href="Gravitational_wave">Gravitational \
        wave</a>.</div>
        <p>In fluid dynamics, gravity waves are waves…</p>
        <div class="hatnote">Further information: <a href="Atmospheric_wave">\
        Atmospheric wave</a></div>
        <div class="hatnote">See also: <a href="Undular_bore">Undular bore</a></div>
        """
        let alts = ArticleHeuristics.disambiguationHatnotes(html: html)
        XCTAssertEqual(alts.map(\.title), ["Gravitational wave"])
        XCTAssertEqual(alts.first?.path, "Gravitational_wave")
    }

    func testCorrectionKeywordsDropDeicticFiller() {
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("the ones einstein predicted"),
            ["einstein", "predicted"])
    }

    func testParentsCountsAsCoveredByEarlyLife() {
        // Synonym-aware coverage: "parents" is answered by mother/father
        // prose, so no corpus pull should fire.
        XCTAssertTrue(ArticleHeuristics.sectionsCoverQuestion(
            putinSections, "What about his parents?"))
    }

    func testEvidenceDepthTracksQuestionComplexity() {
        XCTAssertEqual(ArticleHeuristics.groundedPassageLimit(
            for: "Where did he go to school?"), 2)
        XCTAssertEqual(ArticleHeuristics.groundedPassageLimit(
            for: "What about his parents?"), 2)
        XCTAssertEqual(ArticleHeuristics.groundedPassageLimit(
            for: "Explain how gravitational waves are created."), 4)
        XCTAssertEqual(ArticleHeuristics.groundedPassageCharacterLimit(
            for: "Where did he go to school?"), 1_200)
        XCTAssertEqual(ArticleHeuristics.groundedPassageCharacterLimit(
            for: "How many people died there?"), 1_100)
    }

    func testGroundedWindowFindsParentsInMiddleInsteadOfPrefix() {
        let padding = String(repeating:
            "Putin held a number of public offices during his career. ", count: 18)
        let text = padding
            + "He grew up in Leningrad. His mother Maria was a factory worker, "
            + "and his father Vladimir served in the Soviet Navy. Two older brothers died young. "
            + padding
        let window = ArticleHeuristics.groundedPassageWindow(
            text, question: "What about his parents?", maxChars: 420)
        XCTAssertTrue(window.contains("mother Maria"), "got: \(window)")
        XCTAssertTrue(window.contains("father Vladimir"), "got: \(window)")
        XCTAssertLessThanOrEqual(window.count, 420)
    }

    func testGroundedWindowDoesNotSplitSchoolNumberAbbreviation() {
        let text = String(repeating: "Earlier context about his childhood. ", count: 12)
            + "He attended School No. 193 and later High School 281. "
            + String(repeating: "Later context about politics. ", count: 12)
        let window = ArticleHeuristics.groundedPassageWindow(
            text, question: "Where did he go to school?", maxChars: 240)
        XCTAssertTrue(window.contains("School No. 193"), "got: \(window)")
        XCTAssertTrue(window.contains("High School 281"), "got: \(window)")
    }

    func testDeathQuestionRanksCasualtiesAndFindsCountWindow() {
        let sections = [
            ArticleSection(title: "", level: 0,
                           text: "The battle was fought in 1836."),
            ArticleSection(title: "Interior fighting", level: 2,
                           text: "The defenders withdrew into barracks rooms."),
            ArticleSection(title: "Casualties", level: 2, text:
                "Historians disagree on some details. Between 182 and 257 Texians died, while Mexican casualty estimates vary. Later reports gave other totals."),
        ]
        let ranked = ArticleHeuristics.rankSectionsMultiSource(
            "How many people died there?",
            sources: [(title: "Battle of the Alamo", sections: sections)], k: 2)
        XCTAssertEqual(ranked.first?.section.title, "Casualties")
        let window = ArticleHeuristics.groundedPassageWindow(
            sections[2].text,
            question: "How many people died there?", maxChars: 150)
        XCTAssertTrue(window.contains("182"), "got: \(window)")
    }

    func testDeathQuestionPrefersConsensusRangeOverOpeningClaim() {
        let text = """
        Santa Anna claimed that 600 Texians had been killed, with only 70 Mexican soldiers killed and 300 wounded. His secretary reported 400 killed. Other estimates of Mexican soldiers killed ranged from 60 to 200, with 250–300 wounded. Most Alamo historians place Mexican casualties at 400–600. This represented about one quarter of the assault force. Most eyewitnesses counted between 182 and 257 Texians killed. Some historians believe one Texian escaped but later died of his wounds.
        """
        let window = ArticleHeuristics.groundedPassageWindow(
            text, question: "How many people died there?", maxChars: 300)
        XCTAssertTrue(window.contains("182"), "got: \(window)")
        XCTAssertTrue(window.contains("257"), "got: \(window)")
        XCTAssertFalse(window.hasPrefix("Santa Anna claimed"), "got: \(window)")
    }

    func testExtractiveParentAnswerPreservesNamesAndRoles() {
        let evidence = """
        Vladimir Putin was the youngest of three children born to Vladimir Spiridonovich Putin and Maria Ivanovna Putina. His mother was a factory worker. His father was a conscript in the Soviet Navy who served in the submarine fleet.
        """
        let answer = ArticleHeuristics.groundedExtractiveAnswer(
            question: "What about his parents?", passages: [evidence])
        XCTAssertNotNil(answer)
        XCTAssertTrue(answer!.contains("Vladimir Spiridonovich Putin"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("Maria Ivanovna Putina"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("factory worker"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("Soviet Navy"), "got: \(answer!)")
        XCTAssertFalse(answer!.contains("grandfather"), "got: \(answer!)")
    }

    func testExtractiveParentAnswerPrefersCleanFamilySentenceOverDamagedLead() {
        let damagedLead = """
        February 11, 1731] – December 14, 1799) was a Founding Father and the first president. George Washington was the first of six children of Augustine and Mary Ball Washington.
        """
        let family = """
        He was the first of six children of Augustine and Mary Ball Washington. His father was Augustine Washington. His mother was Mary Ball Washington.
        """
        let answer = ArticleHeuristics.groundedExtractiveAnswer(
            question: "Who were his parents?", passages: [damagedLead, family])
        XCTAssertNotNil(answer)
        XCTAssertTrue(answer!.contains("Augustine"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("Mary Ball"), "got: \(answer!)")
        XCTAssertFalse(answer!.contains("February 11"), "got: \(answer!)")
    }

    func testExtractiveDeathAnswerKeepsKilledSeparateFromCasualties() {
        let evidence = """
        Santa Anna claimed that 600 Texians had been killed, with only 70 Mexican soldiers killed and 300 wounded. His secretary reported 400 killed. Other estimates of Mexican soldiers killed ranged from 60 to 200, with 250–300 wounded. Most Alamo historians place Mexican casualties at 400–600. Most eyewitnesses counted between 182 and 257 Texians killed.
        """
        let answer = ArticleHeuristics.groundedExtractiveAnswer(
            question: "How many people died there?", passages: [evidence])
        XCTAssertNotNil(answer)
        XCTAssertTrue(answer!.contains("60 to 200"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("182 and 257"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("casualties at 400–600"), "got: \(answer!)")
        XCTAssertFalse(answer!.contains("400–600 dead"), "got: \(answer!)")
        XCTAssertFalse(answer!.hasPrefix("Santa Anna claimed"), "got: \(answer!)")
    }

    func testDeathCountRejectsBiographyDeathDates() {
        let evidence = """
        He died on 2 November 2012, in Harrisburg, Pennsylvania. He died on 7 August 1994. He died on 27 February 2010.
        """
        XCTAssertNil(ArticleHeuristics.groundedExtractiveAnswer(
            question: "How many people died in the Pearl Harbor attack?",
            passages: [evidence],
            passageLabels: [
                "Radar warning of Pearl Harbor attack Key participants",
            ]))
    }

    func testDeathCountUsesNamedEventLabelAndRejectsLaterIncident() {
        let later = "On December 4, 2019, a US Navy sailor killed two civilian workers at the Pearl Harbor Naval Shipyard."
        let attack = "A total of 2,403 Americans were killed and 1,178 others were wounded."
        let answer = ArticleHeuristics.groundedExtractiveAnswer(
            question: "How many people died in the Japanese attack?",
            passages: [later, attack],
            passageLabels: [
                "Pearl Harbor Naval presence",
                "Attack on Pearl Harbor Casualties",
            ])
        XCTAssertEqual(answer, attack)
    }

    func testExtractiveAnswerLeavesOpenEndedTurnForBonsai() {
        XCTAssertNil(ArticleHeuristics.groundedExtractiveAnswer(
            question: "Why was the battle important?",
            passages: ["The battle became a symbol of resistance."]))
    }

    func testExtractivePostGraduationAnswerKeepsCareerTimeline() {
        let evidence = """
        Main article: Intelligence career of Vladimir Putin

        In 1975, Putin joined the KGB and trained at the 401st KGB School in Okhta, Leningrad. In 1996, he moved to Moscow and joined the administration of President Boris Yeltsin.
        """
        let answer = ArticleHeuristics.groundedExtractiveAnswer(
            question: "What did he do after graduating?",
            passages: [evidence])
        XCTAssertNotNil(answer)
        XCTAssertTrue(answer!.contains("1975"), "got: \(answer!)")
        XCTAssertTrue(answer!.contains("joined the KGB"), "got: \(answer!)")
        XCTAssertFalse(answer!.contains("1996"), "got: \(answer!)")
        XCTAssertFalse(answer!.contains("Main article"), "got: \(answer!)")
    }

    func testCollegeQuestionStaysInBiographyEarlyLife() {
        let sections = [
            ArticleSection(title: "", level: 0,
                           text: "George Washington was the first president."),
            ArticleSection(title: "Early life (1732–1752)", level: 2,
                           text: "Washington did not have the formal education his brothers received at Appleby Grammar School in England."),
            ArticleSection(title: "Presidency", level: 2,
                           text: "He served two terms as president."),
        ]
        XCTAssertTrue(ArticleHeuristics.sectionsCoverQuestion(
            sections, "Did he go to college?"))
        let ranked = ArticleHeuristics.rankSectionsMultiSource(
            "Did he go to college?",
            sources: [(title: "George Washington", sections: sections)], k: 2)
        XCTAssertEqual(ranked.first?.section.title, "Early life (1732–1752)")
    }

    func testAfterGraduatingPrefersCareerSection() {
        let sections = [
            ArticleSection(title: "", level: 0,
                           text: "Vladimir Putin is a Russian politician."),
            ArticleSection(title: "After the 2022 invasion of Ukraine", level: 2,
                           text: "Events after the invasion affected Russian politics."),
            ArticleSection(title: "Intelligence career", level: 2,
                           text: "After graduating in 1975, Putin joined the KGB."),
        ]
        let ranked = ArticleHeuristics.rankSectionsMultiSource(
            "What did he do after graduating?",
            sources: [(title: "Vladimir Putin", sections: sections)], k: 2)
        XCTAssertEqual(ranked.first?.section.title, "Intelligence career")
    }

    func testCombatantsQuestionRanksArmyOrForcesHeading() {
        let sections = [
            ArticleSection(title: "", level: 0, text: "A major battle."),
            ArticleSection(title: "Legacy", level: 2, text: "Remembered later."),
            ArticleSection(title: "Opposing armies", level: 2,
                           text: "Texian defenders faced Mexican troops."),
        ]
        let ranked = ArticleHeuristics.rankSectionsMultiSource(
            "Who were the combatants?",
            sources: [(title: "Battle", sections: sections)], k: 2)
        XCTAssertEqual(ranked.first?.section.title, "Opposing armies")
    }

    func testFirstDetectionQuestionPrefersHistoryOverDetectorDesign() {
        let sections = [
            ArticleSection(title: "", level: 0,
                           text: "Gravitational waves are ripples in spacetime."),
            ArticleSection(title: "Ground-based detectors", level: 2,
                           text: "Ground-based detectors use laser interferometers to detect tiny strains."),
            ArticleSection(title: "History", level: 2,
                           text: "The first indirect observation came from a binary pulsar. The first direct observation was GW150914 in 2015."),
        ]
        let ranked = ArticleHeuristics.rankSectionsMultiSource(
            "How were they first detected?",
            sources: [(title: "Gravitational waves", sections: sections)], k: 2)
        XCTAssertEqual(ranked.first?.section.title, "History")
    }

    func testFounderQuestionRetrievesGoogleBrainOriginEvidence() {
        let sections = [
            ArticleSection(title: "", level: 0, text:
                "Google Brain began in 2011 as a part-time research collaboration "
                + "between Google Fellow Jeff Dean, researcher Greg Corrado, and "
                + "Stanford professor Andrew Ng."),
            ArticleSection(title: "Google Translate", level: 2, text:
                "Google Brain developed neural machine translation systems."),
            ArticleSection(title: "Controversies", level: 2, text:
                "Google Brain co-founder Samy Bengio announced his resignation in 2021."),
        ]

        XCTAssertTrue(ArticleHeuristics.sectionsCoverQuestion(
            sections, "Who are the founders of Google Brain?",
            articleTitle: "Google Brain"))
        let ranked = ArticleHeuristics.rankSectionsMultiSource(
            "Who are the founders of Google Brain?",
            sources: [(title: "Google Brain", sections: sections)], k: 2)
        XCTAssertTrue(ranked.contains { item in
            item.section.title.isEmpty || item.section.title == "Controversies"
        }, "got: \(ranked.map { $0.section.title })")
        XCTAssertFalse(ranked.contains {
            $0.section.title == "Google Translate"
        }, "irrelevant product section should not displace origin evidence")
    }

    func testGroundedWindowCapsLongUnpunctuatedEvidenceBlock() {
        let text = String(repeating: "opening context ", count: 70)
            + "Mexican reports listed 70 killed and 300 wounded while Texian reports gave different estimates "
            + String(repeating: "later context ", count: 70)
        let window = ArticleHeuristics.groundedPassageWindow(
            text, question: "How many people died?", maxChars: 300)
        XCTAssertLessThanOrEqual(window.count, 300)
        XCTAssertTrue(window.contains("70 killed"), "got: \(window)")
    }
}

// MARK: - Question-flow retrieval (device capture 2026-08-02)

extension DiscussRetrievalTests {

    func testKeywordPoorFollowUpInheritsPreviousQuestionContext() {
        let out = ArticleHeuristics.contextualizedDiscussionQuestion(
            "What year?", previousQuestion: "When did it join nato?")
        XCTAssertTrue(out.lowercased().contains("nato"), "got: \(out)")
        // Keyword-rich turns stay untouched.
        let rich = ArticleHeuristics.contextualizedDiscussionQuestion(
            "What is education like in Bulgaria?",
            previousQuestion: "When did it join nato?")
        XCTAssertEqual(rich, "What is education like in Bulgaria?")
    }

    func testNatoQuestionRanksForeignRelationsOverGeography() {
        let sections = [
            ArticleSection(title: "", level: 0, text:
                "Bulgaria is a country in Southeast Europe on the Balkan peninsula."),
            ArticleSection(title: "Geography", level: 2, text: String(repeating:
                "The west of the country borders the western mountains. To the west lies Serbia. ", count: 4)),
            ArticleSection(title: "Foreign relations", level: 2, text:
                "Bulgaria became a member of NATO in 2004 and of the European Union in 2007. Its foreign relations align with the alliance."),
        ]
        let ranked = ArticleHeuristics.rankSectionsForQuestion(
            "How has Bulgaria dealt with the West and NATO?", sections: sections)
        let nonLead = ranked.filter { !$0.title.isEmpty }
        XCTAssertEqual(nonLead.first?.title, "Foreign relations",
                       "got: \(ranked.map(\.title))")
    }

    func testIsFactoidShaped() {
        XCTAssertTrue(ArticleHeuristics.isFactoidShaped("When did it join nato?"))
        XCTAssertTrue(ArticleHeuristics.isFactoidShaped("What year?"))
        XCTAssertTrue(ArticleHeuristics.isFactoidShaped("How many people live there?"))
        XCTAssertFalse(ArticleHeuristics.isFactoidShaped("Tell me about Bulgaria"))
        XCTAssertFalse(ArticleHeuristics.isFactoidShaped("What is education like?"))
    }

    func testKeyFactSentenceFindsDateSentenceAnywhere() {
        let sources: [(title: String, sections: [ArticleSection])] = [
            ("Lithuania", [
                ArticleSection(title: "", level: 0, text:
                    "Lithuania is a country in the Baltic region of Europe."),
                ArticleSection(title: "Demographics", level: 2, text:
                    "The median age was 44 years in 2022. The fertility rate was 1.34 in 2021."),
                ArticleSection(title: "Foreign relations", level: 2, text:
                    "Lithuania is a member of the OSCE. Lithuania joined NATO on 29 March 2004. It joined the European Union that May."),
            ]),
        ]
        let fact = ArticleHeuristics.keyFactSentence(
            question: "When did it join nato? — join nato lithuania",
            sources: sources)
        XCTAssertNotNil(fact)
        XCTAssertTrue(fact?.sentence.contains("2004") ?? false, "got: \(fact?.sentence ?? "nil")")
        XCTAssertTrue(fact?.sentence.lowercased().contains("nato") ?? false)
    }
}
