import Foundation
import Darwin
let mode = CommandLine.arguments[1]
let root = URL(fileURLWithPath: CommandLine.arguments[2])
let leaves = ["6", "0", "1", "2", "3", "4", "5", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
var rawCache: [Data] = []
var parsedCache: [([[String: Any]], Int)] = []
var bytes = 0
var matches: [[String: Any]] = []
let start = Date()
try autoreleasepool {
    for leaf in leaves {
        try autoreleasepool {
            let data = try Data(contentsOf: root.appendingPathComponent("be-\(leaf).json"))
            if mode == "old" {
                let rows = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
                parsedCache.append((rows, data.count)); bytes += data.count
                while bytes > 24 * 1024 * 1024 && parsedCache.count > 1 { bytes -= parsedCache.removeFirst().1 }
                matches += rows.filter { (($0["n"] as? String) ?? "").lowercased().contains("belize") }
            } else {
                while !rawCache.isEmpty && bytes + data.count > 24 * 1024 * 1024 { bytes -= rawCache.removeFirst().count }
                rawCache.append(data); bytes += data.count
                matches += try FilteredPlaceJSON.matching(data, query: "Belize")
            }
        }
    }
}
var usage = rusage(); getrusage(RUSAGE_SELF, &usage)
print("\(mode): peakRSS=\(usage.ru_maxrss) bytes elapsed=\(Date().timeIntervalSince(start))s matches=\(matches.count)")
print(matches.map { $0["n"] as? String ?? "" }.sorted())
