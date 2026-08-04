// SPDX-License-Identifier: MIT

import Foundation

/// One downloadable offline archive the in-app downloader can fetch: a
/// Wikipedia edition (with or without images) or a StreetZIM offline map.
/// The catalogs resolve the *latest* files at runtime — Kiwix and StreetZIM
/// republish archives regularly, so a shipped URL would eventually strand
/// new users on an old or removed file — and fall back to a list pinned at
/// build time when the index is unreachable (offline-first app, after all).
struct ZimCatalogItem: Identifiable, Hashable, Sendable {
    enum Kind: Hashable, Sendable {
        case wikipedia(images: Bool)
        case map
    }

    /// Stable identity independent of the dated filename, so "already
    /// downloading / already downloaded" survives a catalog refresh that
    /// bumps the file date.
    let id: String
    let title: String
    let detail: String
    let sizeLabel: String
    /// Approximate size for free-space prechecks (labels round; that's fine).
    let sizeBytes: Int64
    let url: URL
    let kind: Kind
    /// Section grouping for maps ("Continents…", "Individual countries", …).
    var tier: String? = nil
    var recommended: Bool = false

    var filename: String { url.lastPathComponent }
}

// MARK: - Wikipedia (Kiwix)

/// Resolves the newest official English Wikipedia archives — both the
/// no-picture (`nopic`) and with-picture (`maxi`) flavors — from the Kiwix
/// directory index.
enum WikipediaZimCatalog {
    static let indexURL = URL(string: "https://download.kiwix.org/zim/wikipedia/")!

    enum Edition: String, CaseIterable, Sendable {
        case top, simple, complete
    }

    /// Pinned to the newest files at the time this build shipped.
    static let fallback: [ZimCatalogItem] = [
        item(edition: .top, images: false,
             filename: "wikipedia_en_top_nopic_2026-06.zim", sizeLabel: "2.1 GB"),
        item(edition: .simple, images: false,
             filename: "wikipedia_en_simple_all_nopic_2026-05.zim", sizeLabel: "937 MB"),
        item(edition: .complete, images: false,
             filename: "wikipedia_en_all_nopic_2026-06.zim", sizeLabel: "49 GB"),
        item(edition: .top, images: true,
             filename: "wikipedia_en_top_maxi_2026-06.zim", sizeLabel: "7.8 GB"),
        item(edition: .simple, images: true,
             filename: "wikipedia_en_simple_all_maxi_2026-05.zim", sizeLabel: "3.2 GB"),
        item(edition: .complete, images: true,
             filename: "wikipedia_en_all_maxi_2026-02.zim", sizeLabel: "115 GB"),
    ]

    static func load() async throws -> [ZimCatalogItem] {
        var request = URLRequest(url: indexURL)
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw URLError(.cannotDecodeContentData)
        }
        let parsed = parse(html: html)
        // A page that yields fewer families than the fallback knows about is
        // a format change, not an emptier catalog — keep the pinned list.
        guard parsed.count >= fallback.count else { throw URLError(.cannotParseResponse) }
        return parsed
    }

    /// Pure parser over the nginx autoindex, kept static/side-effect-free so
    /// the Mac unit target can pin the directory format without network.
    static func parse(html: String) -> [ZimCatalogItem] {
        let pattern = #"(wikipedia_en_(all|top|simple_all)_(nopic|maxi)_(\d{4}-\d{2})\.zim)</a>\s+[0-9-]+\s+[0-9:]+\s+([0-9.]+\s*[KMGTP])"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(html.startIndex..., in: html)

        struct Key: Hashable { let edition: Edition; let images: Bool }
        var newest: [Key: (filename: String, size: String)] = [:]

        for match in regex.matches(in: html, range: range) {
            guard match.numberOfRanges == 6,
                  let filenameRange = Range(match.range(at: 1), in: html),
                  let familyRange = Range(match.range(at: 2), in: html),
                  let flavorRange = Range(match.range(at: 3), in: html),
                  let sizeRange = Range(match.range(at: 5), in: html)
            else { continue }

            let edition: Edition
            switch String(html[familyRange]) {
            case "top": edition = .top
            case "simple_all": edition = .simple
            case "all": edition = .complete
            default: continue
            }
            let key = Key(edition: edition, images: String(html[flavorRange]) == "maxi")
            let filename = String(html[filenameRange])
            // Dated YYYY-MM filenames within a family sort lexicographically.
            if newest[key].map({ $0.filename < filename }) ?? true {
                newest[key] = (filename, String(html[sizeRange]))
            }
        }

        var items: [ZimCatalogItem] = []
        for images in [false, true] {
            for edition in Edition.allCases {
                guard let found = newest[Key(edition: edition, images: images)] else { continue }
                items.append(item(edition: edition, images: images,
                                  filename: found.filename,
                                  sizeLabel: expandSize(found.size)))
            }
        }
        return items
    }

    private static func item(edition: Edition, images: Bool,
                             filename: String, sizeLabel: String) -> ZimCatalogItem {
        let title: String
        let detail: String
        switch edition {
        case .top:
            title = "Popular English Wikipedia"
            detail = images
                ? "The most useful articles with images. Best all-round choice when you have the space."
                : "The most useful articles, full text, no pictures. Best starting point for a phone."
        case .simple:
            title = "Simple English Wikipedia"
            detail = images
                ? "Every Simple English article with images — plain language, compact."
                : "Every Simple English article, full text, no pictures. The smallest complete encyclopedia."
        case .complete:
            title = "Complete English Wikipedia"
            detail = images
                ? "Every English Wikipedia article with images. Requires very large free storage."
                : "Every English Wikipedia article, full text, no pictures. Requires substantial free storage."
        }
        return ZimCatalogItem(
            id: "wikipedia.\(edition.rawValue).\(images ? "maxi" : "nopic")",
            title: title,
            detail: detail,
            sizeLabel: sizeLabel,
            sizeBytes: approximateBytes(sizeLabel),
            url: indexURL.appendingPathComponent(filename),
            kind: .wikipedia(images: images),
            recommended: edition == .top && !images)
    }

    /// "2.1G" → "2.1 GB" (nginx autoindex prints compact suffixes).
    static func expandSize(_ compact: String) -> String {
        let trimmed = compact.replacingOccurrences(of: " ", with: "")
        guard let suffix = trimmed.last else { return compact }
        let number = trimmed.dropLast()
        let unit: String
        switch suffix {
        case "K": unit = "KB"
        case "M": unit = "MB"
        case "G": unit = "GB"
        case "T": unit = "TB"
        case "P": unit = "PB"
        default: return compact
        }
        return "\(number) \(unit)"
    }
}

/// "2.1 GB" / "937 MB" → approximate byte count for free-space checks.
/// Binary multipliers deliberately overestimate slightly, which errs on the
/// safe side of "will it fit".
func approximateBytes(_ sizeLabel: String) -> Int64 {
    let scanner = Scanner(string: sizeLabel)
    guard let number = scanner.scanDouble() else { return 0 }
    let unit = sizeLabel.uppercased()
    let multiplier: Double
    if unit.contains("T") { multiplier = 1024 * 1024 * 1024 * 1024 }
    else if unit.contains("G") { multiplier = 1024 * 1024 * 1024 }
    else if unit.contains("M") { multiplier = 1024 * 1024 }
    else if unit.contains("K") { multiplier = 1024 }
    else { multiplier = 1 }
    return Int64(number * multiplier)
}

// MARK: - StreetZIM maps

/// Resolves the current StreetZIM region list from streetzim.web.app (the
/// same cards the website shows, including size and coverage description).
enum StreetZimCatalog {
    static let indexURL = URL(string: "https://streetzim.web.app/")!

    static func load() async throws -> [ZimCatalogItem] {
        var request = URLRequest(url: indexURL)
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw URLError(.cannotDecodeContentData)
        }
        let parsed = parse(html: html)
        // The site lists dozens of regions; a near-empty parse means the page
        // format changed and the pinned fallback is more trustworthy.
        guard parsed.count >= 10 else { throw URLError(.cannotParseResponse) }
        return parsed
    }

    /// Pure parser over the StreetZIM landing page. The page is organized as
    /// tier headers (`<h3 class="maps-tier-header">`) followed by
    /// `map-card` blocks carrying title, size, description, and the
    /// archive.org download link.
    static func parse(html: String) -> [ZimCatalogItem] {
        var items: [ZimCatalogItem] = []
        // Split into tier sections. Content before the first header belongs
        // to the page's opening tier of continent-scale maps.
        let headerPattern = #"<h3 class="maps-tier-header">([^<]*)</h3>"#
        guard let headerRegex = try? NSRegularExpression(pattern: headerPattern) else { return [] }
        let all = NSRange(html.startIndex..., in: html)
        var sections: [(tier: String, body: Substring)] = []
        var cursor = html.startIndex
        var currentTier = "Continents & regions"
        for match in headerRegex.matches(in: html, range: all) {
            guard let whole = Range(match.range, in: html),
                  let name = Range(match.range(at: 1), in: html) else { continue }
            sections.append((currentTier, html[cursor..<whole.lowerBound]))
            currentTier = decodeEntities(String(html[name]))
            cursor = whole.upperBound
        }
        sections.append((currentTier, html[cursor...]))

        for (tier, body) in sections {
            for card in body.components(separatedBy: "<div class=\"map-card\">").dropFirst() {
                guard let title = firstMatch(#"<div class="map-card-title">(.*?)</div>"#, in: card),
                      let size = firstMatch(#"<div class="map-card-size">(.*?)</div>"#, in: card),
                      let url = firstMatch(#"href="(https://archive\.org/download/streetzim-[a-z0-9-]+/[^"]+\.zim)""#, in: card),
                      let downloadURL = URL(string: url)
                else { continue }
                let desc = firstMatch(#"<p class="map-card-desc">(.*?)</p>"#, in: card).map {
                    truncate(decodeEntities(stripTags($0)), to: 140)
                } ?? ""
                let region = downloadURL.deletingLastPathComponent().lastPathComponent
                    .replacingOccurrences(of: "streetzim-", with: "")
                items.append(ZimCatalogItem(
                    id: "map.\(region)",
                    title: decodeEntities(title),
                    detail: desc,
                    sizeLabel: size.trimmingCharacters(in: .whitespaces),
                    sizeBytes: approximateBytes(size),
                    url: downloadURL,
                    kind: .map,
                    tier: tier))
            }
        }
        return items
    }

    private static func firstMatch(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern,
                                                   options: [.dotMatchesLineSeparators]) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              let captured = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[captured])
    }

    private static func stripTags(_ html: String) -> String {
        html.replacingOccurrences(of: #"<[^>]+>"#, with: "", options: .regularExpression)
    }

    private static func decodeEntities(_ text: String) -> String {
        var result = text
        for (entity, plain) in [("&amp;", "&"), ("&mdash;", "—"), ("&ndash;", "–"),
                                ("&lsquo;", "\u{2018}"), ("&rsquo;", "\u{2019}"),
                                ("&ldquo;", "\u{201C}"), ("&rdquo;", "\u{201D}"),
                                ("&quot;", "\""), ("&#39;", "'"), ("&nbsp;", " ")] {
            result = result.replacingOccurrences(of: entity, with: plain)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func truncate(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        let cut = String(text.prefix(limit))
        let trimmed = cut.contains(" ")
            ? cut[..<(cut.lastIndex(of: " ") ?? cut.endIndex)]
            : cut[...]
        return String(trimmed).trimmingCharacters(in: CharacterSet(charactersIn: ",;—– ")) + "…"
    }

    /// Pinned to the site's region list at the time this build shipped.
    static let fallback: [ZimCatalogItem] = [
        .init(id: "map.europe", title: "Europe",
              detail: "United Kingdom, Ireland, France, Spain, Portugal, Germany, Italy, Netherlands, Belgium, Switzerland, Austria…",
              sizeLabel: "62.1 GB", sizeBytes: 66679367270,
              url: URL(string: "https://archive.org/download/streetzim-europe/osm-europe-2026-05-06.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.united-states", title: "United States",
              detail: "Continental United States — all 48 contiguous states and Washington, D.C.",
              sizeLabel: "38.0 GB", sizeBytes: 40802189312,
              url: URL(string: "https://archive.org/download/streetzim-united-states/osm-united-states-2026-05-05.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.south-america", title: "South America",
              detail: "Continental South America — Colombia, Venezuela, Guyana, Suriname, French Guiana, Ecuador, Peru, Bolivia…",
              sizeLabel: "20.7 GB", sizeBytes: 22226455756,
              url: URL(string: "https://archive.org/download/streetzim-south-america/osm-south-america-2026-05-08.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.africa", title: "Africa",
              detail: "All of Africa — Algeria, Egypt, Ethiopia, Kenya, Morocco, Nigeria, South Africa, Tanzania, and 40+ more…",
              sizeLabel: "20.4 GB", sizeBytes: 21904333209,
              url: URL(string: "https://archive.org/download/streetzim-africa/osm-africa-2026-04-29.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.indian-subcontinent", title: "Indian Subcontinent",
              detail: "India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan, and the Maldives.",
              sizeLabel: "12.0 GB", sizeBytes: 12884901888,
              url: URL(string: "https://archive.org/download/streetzim-indian-subcontinent/osm-indian-subcontinent-2026-05-08.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.china", title: "China",
              detail: "China — Beijing, Shanghai, Guangzhou, Shenzhen, Chengdu, Chongqing, Xi’an, Wuhan, Hangzhou, Hong Kong, and…",
              sizeLabel: "19.5 GB", sizeBytes: 20937965568,
              url: URL(string: "https://archive.org/download/streetzim-china/osm-china-2026-05-09.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.russia", title: "Russia",
              detail: "Russia — Moscow, Saint Petersburg, Novosibirsk, Yekaterinburg, Kazan, Nizhny Novgorod, Sochi, Vladivostok…",
              sizeLabel: "24.5 GB", sizeBytes: 26306674688,
              url: URL(string: "https://archive.org/download/streetzim-russia/osm-russia-2026-05-12c.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.australia-nz", title: "Australia & New Zealand",
              detail: "Australia and New Zealand — Sydney, Melbourne, Brisbane, Perth, Auckland, Wellington, the Great Barrier Reef…",
              sizeLabel: "6.3 GB", sizeBytes: 6764573491,
              url: URL(string: "https://archive.org/download/streetzim-australia-nz/osm-australia-nz-2026-04-28.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.southeast-asia", title: "Southeast Asia",
              detail: "Thailand, Vietnam, Myanmar, Cambodia, Laos, Malaysia, Singapore, Indonesia, the Philippines, Brunei, and…",
              sizeLabel: "9.7 GB", sizeBytes: 10415295692,
              url: URL(string: "https://archive.org/download/streetzim-southeast-asia/osm-southeast-asia-2026-05-09.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.canada", title: "Canada",
              detail: "All of Canada — Toronto, Montreal, Vancouver, Calgary, Ottawa, Quebec City, Edmonton, Halifax, the Rockies…",
              sizeLabel: "25.7 GB", sizeBytes: 27595164876,
              url: URL(string: "https://archive.org/download/streetzim-canada/osm-canada-2026-04-27d.zim")!,
              kind: .map, tier: "Continents & continent-scale regions"),
        .init(id: "map.west-asia", title: "West Asia",
              detail: "Turkey, Syria, Lebanon, Israel, Palestine, Jordan, Iraq, Iran, Kuwait, Saudi Arabia, Bahrain, Qatar, UAE…",
              sizeLabel: "10.3 GB", sizeBytes: 11059540787,
              url: URL(string: "https://archive.org/download/streetzim-west-asia/osm-west-asia-2026-04-27d.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.central-asia", title: "Central Asia",
              detail: "Central Asia — Kazakhstan, Uzbekistan, Turkmenistan, Tajikistan, Kyrgyzstan, Afghanistan, and the Caucasus.…",
              sizeLabel: "4.4 GB", sizeBytes: 4724464025,
              url: URL(string: "https://archive.org/download/streetzim-central-asia/osm-central-asia-2026-04-26c.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.korea-mongolia", title: "Korea & Mongolia",
              detail: "The Korean Peninsula and Mongolia — Seoul, Busan, Daegu, Incheon, Pyongyang, Ulaanbaatar, Darkhan; the Gobi…",
              sizeLabel: "5.2 GB", sizeBytes: 5583457484,
              url: URL(string: "https://archive.org/download/streetzim-korea-mongolia/osm-korea-mongolia-2026-05-11c.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.caucasus", title: "Caucasus",
              detail: "The Caucasus — Georgia, Armenia, and Azerbaijan, with the southern Russian Caucasus and eastern Turkey.…",
              sizeLabel: "1.1 GB", sizeBytes: 1181116006,
              url: URL(string: "https://archive.org/download/streetzim-caucasus/osm-caucasus-2026-05-11c.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.central-america-caribbean", title: "Central America & Caribbean",
              detail: "Central America and the Caribbean — Yucatán, Belize, Guatemala, Honduras, El Salvador, Nicaragua, Costa Rica…",
              sizeLabel: "2.2 GB", sizeBytes: 2362232012,
              url: URL(string: "https://archive.org/download/streetzim-central-america-caribbean/osm-central-america-caribbean-2026-05-11.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.himalayas", title: "Himalayas",
              detail: "The Himalayas, Karakoram, Hindu Kush, and Pamir — Nepal, Bhutan, Tibet, Sikkim, Ladakh, Kashmir, the Indus…",
              sizeLabel: "5.2 GB", sizeBytes: 5583457484,
              url: URL(string: "https://archive.org/download/streetzim-himalayas/osm-himalayas-2026-04-29.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.baltics", title: "Baltics",
              detail: "Estonia, Latvia, and Lithuania — Tallinn, Riga, Vilnius, and the Baltic Sea coast.",
              sizeLabel: "1.5 GB", sizeBytes: 1610612736,
              url: URL(string: "https://archive.org/download/streetzim-baltics/osm-baltics-2026-05-10.zim")!,
              kind: .map, tier: "Multi-country regions"),
        .init(id: "map.brazil", title: "Brazil",
              detail: "All of Brazil — São Paulo, Rio de Janeiro, Brasília, Salvador, Belo Horizonte, Fortaleza, Manaus, the Amazon…",
              sizeLabel: "15.9 GB", sizeBytes: 17072495001,
              url: URL(string: "https://archive.org/download/streetzim-brazil/osm-brazil-2026-05-07.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.argentina", title: "Argentina",
              detail: "Argentina — Buenos Aires, Córdoba, Rosario, Mendoza, Bariloche, the Pampas, the Andes, Patagonia, and Tierra…",
              sizeLabel: "3.1 GB", sizeBytes: 3328599654,
              url: URL(string: "https://archive.org/download/streetzim-argentina/osm-argentina-2026-05-08.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.ukraine", title: "Ukraine, Moldova & Western Russia",
              detail: "Ukraine and Moldova in full (including Crimea), plus the western Russian frontier — Volgograd, Rostov, Sochi…",
              sizeLabel: "4.2 GB", sizeBytes: 4509715660,
              url: URL(string: "https://archive.org/download/streetzim-ukraine/osm-ukraine-2026-05-05b.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.japan", title: "Japan",
              detail: "Japan — all four main islands (Honshu, Hokkaido, Kyushu, Shikoku), Okinawa, and the Ryukyu archipelago.…",
              sizeLabel: "4.8 GB", sizeBytes: 5153960755,
              url: URL(string: "https://archive.org/download/streetzim-japan/osm-japan-2026-04-28.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.iran", title: "Iran",
              detail: "Iran — from the Caspian Sea to the Persian Gulf, including Tehran, Isfahan, Shiraz, and Mashhad.",
              sizeLabel: "2.4 GB", sizeBytes: 2576980377,
              url: URL(string: "https://archive.org/download/streetzim-iran/osm-iran-2026-05-11.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.egypt", title: "Egypt",
              detail: "Egypt — Cairo, Alexandria, Giza, Luxor, Aswan, the Nile Valley, Sinai Peninsula, and the Red Sea coast.",
              sizeLabel: "1.2 GB", sizeBytes: 1288490188,
              url: URL(string: "https://archive.org/download/streetzim-egypt/osm-egypt-2026-04-26c.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.iceland", title: "Iceland",
              detail: "Iceland — Reykjavík, Akureyri, the Ring Road, Þingvellir, Geysir, Gullfoss, Vík, Jökulsárlón, Snæfellsnes…",
              sizeLabel: "366 MB", sizeBytes: 383778816,
              url: URL(string: "https://archive.org/download/streetzim-iceland/osm-iceland-2026-05-11c.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.turkey", title: "Turkey",
              detail: "Turkey — Istanbul, Ankara, İzmir, Bursa, Antalya, Konya, Gaziantep, Trabzon, Cappadocia, Pamukkale, Ephesus…",
              sizeLabel: "3.0 GB", sizeBytes: 3221225472,
              url: URL(string: "https://archive.org/download/streetzim-turkey/osm-turkey-2026-05-11c.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.south-korea", title: "South Korea",
              detail: "South Korea — Seoul, Busan, Incheon, Daegu, Daejeon, Gwangju, Suwon, Jeju Island, the DMZ, Seoraksan, and…",
              sizeLabel: "900 MB", sizeBytes: 943718400,
              url: URL(string: "https://archive.org/download/streetzim-south-korea/osm-south-korea-2026-05-11c.zim")!,
              kind: .map, tier: "Individual countries"),
        .init(id: "map.east-coast-us", title: "East Coast US",
              detail: "U.S. East Coast from Maine to Florida — New York, Boston, Philadelphia, Washington D.C., Atlanta, Miami, and…",
              sizeLabel: "8.9 GB", sizeBytes: 9556302233,
              url: URL(string: "https://archive.org/download/streetzim-east-coast-us/osm-east-coast-us-2026-04-27d.zim")!,
              kind: .map, tier: "United States — sub-regions"),
        .init(id: "map.midwest-us", title: "Midwest United States",
              detail: "Ohio, Indiana, Illinois, Michigan, Wisconsin, Minnesota, Iowa, Missouri, North Dakota, South Dakota…",
              sizeLabel: "7.2 GB", sizeBytes: 7730941132,
              url: URL(string: "https://archive.org/download/streetzim-midwest-us/osm-midwest-us-2026-04-29.zim")!,
              kind: .map, tier: "United States — sub-regions"),
        .init(id: "map.west-coast-us", title: "West Coast US",
              detail: "U.S. West Coast: Washington, Oregon, and California — Seattle, Portland, San Francisco, Los Angeles, San…",
              sizeLabel: "4.8 GB", sizeBytes: 5153960755,
              url: URL(string: "https://archive.org/download/streetzim-west-coast-us/osm-west-coast-us-2026-04-26c.zim")!,
              kind: .map, tier: "United States — sub-regions"),
        .init(id: "map.central-us", title: "Central US",
              detail: "The Mountain West and surrounds — Utah, Colorado, Wyoming, Montana, Idaho, Nevada, Arizona, and New Mexico.…",
              sizeLabel: "6.7 GB", sizeBytes: 7194070220,
              url: URL(string: "https://archive.org/download/streetzim-central-us/osm-central-us-2026-04-26c.zim")!,
              kind: .map, tier: "United States — sub-regions"),
        .init(id: "map.california", title: "California",
              detail: "All of California — from the Oregon border to Mexico, the Pacific coast to the Sierra Nevada.",
              sizeLabel: "3.1 GB", sizeBytes: 3328599654,
              url: URL(string: "https://archive.org/download/streetzim-california/osm-california-2026-06-02.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.hawaii", title: "Hawaii",
              detail: "The Hawaiian Islands — Honolulu, Hilo, Kahului; O‘ahu, Maui, Hawai‘i (Big Island), Kaua‘i, Moloka‘i, and…",
              sizeLabel: "119 MB", sizeBytes: 124780544,
              url: URL(string: "https://archive.org/download/streetzim-hawaii/osm-hawaii-2026-05-12.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.alaska", title: "Alaska",
              detail: "Alaska — Anchorage, Fairbanks, Juneau, Sitka, Ketchikan, Nome, Barrow; Denali, the Brooks Range, the…",
              sizeLabel: "3.0 GB", sizeBytes: 3221225472,
              url: URL(string: "https://archive.org/download/streetzim-alaska/osm-alaska-2026-05-11.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.new-york-state", title: "New York State",
              detail: "All of New York State — New York City, Albany, Buffalo, Rochester, Syracuse, Yonkers; Long Island, the Hudson…",
              sizeLabel: "2.5 GB", sizeBytes: 2684354560,
              url: URL(string: "https://archive.org/download/streetzim-new-york-state/osm-new-york-state-2026-05-12c.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.florida", title: "Florida",
              detail: "All of Florida — Miami, Orlando, Tampa, Jacksonville, St. Petersburg, Tallahassee, Fort Lauderdale; the…",
              sizeLabel: "1.9 GB", sizeBytes: 2040109465,
              url: URL(string: "https://archive.org/download/streetzim-florida/osm-florida-2026-05-11.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.carolinas", title: "The Carolinas",
              detail: "North and South Carolina — Charlotte, Raleigh, Durham, Greensboro, Winston-Salem, Columbia, Charleston…",
              sizeLabel: "1.9 GB", sizeBytes: 2040109465,
              url: URL(string: "https://archive.org/download/streetzim-carolinas/osm-carolinas-2026-07-06.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.nyc-metro", title: "New York Metro",
              detail: "The NYC tri-state metropolitan area — Manhattan, Brooklyn, Queens, the Bronx, Staten Island, Long Island…",
              sizeLabel: "536 MB", sizeBytes: 562036736,
              url: URL(string: "https://archive.org/download/streetzim-nyc-metro/osm-nyc-metro-2026-05-11c.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.chicago-metro", title: "Chicago Metro",
              detail: "Greater Chicago — the city and Cook, DuPage, Lake, Will, Kane, McHenry counties, plus northwest Indiana and…",
              sizeLabel: "389 MB", sizeBytes: 407896064,
              url: URL(string: "https://archive.org/download/streetzim-chicago-metro/osm-chicago-metro-2026-05-11c.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.greater-la", title: "Greater Los Angeles",
              detail: "The Los Angeles metropolitan area — LA, Long Beach, Anaheim, Santa Monica, Pasadena, Riverside, San…",
              sizeLabel: "599 MB", sizeBytes: 628097024,
              url: URL(string: "https://archive.org/download/streetzim-greater-la/osm-greater-la-2026-05-11c.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.texas", title: "Texas",
              detail: "Texas, USA — from the Gulf Coast to the Rio Grande, including Houston, Dallas, San Antonio, Austin, Fort…",
              sizeLabel: "3.6 GB", sizeBytes: 3865470566,
              url: URL(string: "https://archive.org/download/streetzim-texas/osm-texas-2026-05-11.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.colorado", title: "Colorado",
              detail: "The Rocky Mountain state — Denver, Aspen, Vail, Rocky Mountain National Park, and the Continental Divide.",
              sizeLabel: "1.1 GB", sizeBytes: 1181116006,
              url: URL(string: "https://archive.org/download/streetzim-colorado/osm-colorado-2026-05-10.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.silicon-valley", title: "Silicon Valley",
              detail: "San Francisco Bay Area — San Francisco, Oakland, Palo Alto, Mountain View, Stanford, Cupertino, San Jose, and…",
              sizeLabel: "311 MB", sizeBytes: 326107136,
              url: URL(string: "https://archive.org/download/streetzim-silicon-valley/osm-silicon-valley-2026-06-03.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.washington-dc", title: "Washington, D.C.",
              detail: "Washington, D.C. — the U.S. capital and surrounding metro area.",
              sizeLabel: "94 MB", sizeBytes: 98566144,
              url: URL(string: "https://archive.org/download/streetzim-washington-dc/osm-washington-dc-2026-05-10.zim")!,
              kind: .map, tier: "States, cities & islands"),
        .init(id: "map.hispaniola", title: "Hispaniola",
              detail: "The Caribbean island of Hispaniola — Haiti and the Dominican Republic.",
              sizeLabel: "187 MB", sizeBytes: 196083712,
              url: URL(string: "https://archive.org/download/streetzim-hispaniola/osm-hispaniola-2026-05-10.zim")!,
              kind: .map, tier: "States, cities & islands"),
    ]
}
