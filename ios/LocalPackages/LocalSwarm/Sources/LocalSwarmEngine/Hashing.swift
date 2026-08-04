import Foundation
import CryptoKit

enum Hashing {
    /// Lowercase hex string of a SHA-256 digest.
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(of digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

extension FileHandle {
    /// Reads exactly `count` bytes (or fewer only at end-of-file). `read(upToCount:)`
    /// may legally return a short read mid-file; chunk alignment depends on full
    /// reads, so we loop until satisfied.
    func readFully(_ count: Int) throws -> Data {
        var buffer = Data()
        buffer.reserveCapacity(count)
        while buffer.count < count {
            guard let next = try read(upToCount: count - buffer.count), !next.isEmpty else { break }
            buffer.append(next)
        }
        return buffer
    }
}
