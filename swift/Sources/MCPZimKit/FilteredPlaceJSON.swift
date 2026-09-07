import Foundation

/// Scan a JSON record array in small batches. A shard's decoded Foundation
/// object graph must never be retained merely to search a few place names.
enum FilteredPlaceJSON {
    enum Failure: LocalizedError {
        case malformed, oversizedRecord, tooManyMatches
        var errorDescription: String? {
            switch self {
            case .malformed: return "The offline map search data is malformed."
            case .oversizedRecord: return "An offline map search record is too large to read safely."
            case .tooManyMatches: return "Too many offline map matches. Try a more specific place name."
            }
        }
    }

    static func matching(_ data: Data, query: String) throws -> [[String: Any]] {
        let q = query.lowercased()
        guard !q.isEmpty else { return [] }
        return try data.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            var cursor = 0
            func whitespace(_ b: UInt8) -> Bool { b == 32 || b == 9 || b == 10 || b == 13 }
            func skipSpace() { while cursor < bytes.count && whitespace(bytes[cursor]) { cursor += 1 } }
            skipSpace()
            guard cursor < bytes.count, bytes[cursor] == 91 else { throw Failure.malformed }
            cursor += 1
            var batch = Data([91])
            var count = 0
            var result: [[String: Any]] = []
            func flush() throws {
                guard count > 0 else { return }
                try Task.checkCancellation()
                batch.append(93)
                let matches: [[String: Any]] = try autoreleasepool {
                    guard let rows = try JSONSerialization.jsonObject(with: batch) as? [[String: Any]]
                    else { throw Failure.malformed }
                    return rows.filter { (($0["n"] as? String) ?? "").lowercased().contains(q) }
                }
                guard result.count + matches.count <= 5_000 else { throw Failure.tooManyMatches }
                result.append(contentsOf: matches)
                batch = Data([91]); count = 0
            }
            skipSpace()
            if cursor < bytes.count, bytes[cursor] == 93 {
                cursor += 1; skipSpace()
                guard cursor == bytes.count else { throw Failure.malformed }
                return []
            }
            while cursor < bytes.count {
                guard bytes[cursor] == 123 else { throw Failure.malformed }
                let start = cursor
                var depth = 0
                var inString = false
                var escaped = false
                repeat {
                    let b = bytes[cursor]
                    if inString {
                        if escaped { escaped = false }
                        else if b == 92 { escaped = true }
                        else if b == 34 { inString = false }
                    } else if b == 34 { inString = true }
                    else if b == 123 || b == 91 { depth += 1 }
                    else if b == 125 || b == 93 { depth -= 1 }
                    cursor += 1
                    guard cursor - start <= 1_048_576 else { throw Failure.oversizedRecord }
                } while cursor < bytes.count && (depth > 0 || inString)
                guard depth == 0, !inString else { throw Failure.malformed }
                if count > 0 { batch.append(44) }
                batch.append(contentsOf: bytes[start..<cursor]); count += 1
                if count == 128 || batch.count >= 262_144 { try flush() }
                skipSpace()
                guard cursor < bytes.count else { throw Failure.malformed }
                if bytes[cursor] == 93 {
                    cursor += 1; skipSpace()
                    guard cursor == bytes.count else { throw Failure.malformed }
                    try flush()
                    return result
                }
                guard bytes[cursor] == 44 else { throw Failure.malformed }
                cursor += 1; skipSpace()
            }
            throw Failure.malformed
        }
    }
}
