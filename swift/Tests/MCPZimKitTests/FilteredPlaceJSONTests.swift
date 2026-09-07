import XCTest
@testable import MCPZimKit

final class FilteredPlaceJSONTests: XCTestCase {
    func testMatchesFoundationForUnicodeEscapesAndNestedRecords() throws {
        let data = Data(#"[{"n":"Belize \"} [\"","a":17.2,"o":-88.7,"extra":{"values":[1,2]}},{"n":"B\u0065lize café","t":"place"},{"n":"Elsewhere"}]"#.utf8)
        let expected = (try JSONSerialization.jsonObject(with: data) as! [[String: Any]])
            .filter { ($0["n"] as! String).lowercased().contains("belize") }
        let actual = try FilteredPlaceJSON.matching(data, query: "BELIZE")
        XCTAssertEqual(try JSONSerialization.data(withJSONObject: actual, options: .sortedKeys),
                       try JSONSerialization.data(withJSONObject: expected, options: .sortedKeys))
    }
    func testManyNonmatchesAcrossBatchBoundaries() throws {
        let rows = (0..<10_000).map { ["n": $0 == 5555 ? "Belize" : "Berlin \($0)"] }
        let data = try JSONSerialization.data(withJSONObject: rows)
        XCTAssertEqual(try FilteredPlaceJSON.matching(data, query: "Belize").count, 1)
        XCTAssertTrue(try FilteredPlaceJSON.matching(data, query: "missing").isEmpty)
    }
    func testMalformedInputNeverReturnsPartialResults() {
        for raw in ["", "{}", "[1]", #"[{"n":"Belize"},]"#, #"[{"n":"Belize"}] junk"#, #"[{"n":"Belize"}"#, #"[{"n":"Belize"],{}]"#] {
            XCTAssertThrowsError(try FilteredPlaceJSON.matching(Data(raw.utf8), query: "Belize"), raw)
        }
    }
    func testOversizedRecordAndUnboundedMatchesFailExplicitly() throws {
        let large = try JSONSerialization.data(withJSONObject: [["n": String(repeating: "x", count: 1_048_576)]])
        XCTAssertThrowsError(try FilteredPlaceJSON.matching(large, query: "Belize"))
        let broad = try JSONSerialization.data(withJSONObject: Array(repeating: ["n":"Belize"], count: 5001))
        XCTAssertThrowsError(try FilteredPlaceJSON.matching(broad, query: "Belize"))
    }
}
