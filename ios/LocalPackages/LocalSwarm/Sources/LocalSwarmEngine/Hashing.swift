import Foundation
import CryptoKit

enum Hashing {
    /// Lowercase hex string of a SHA-256 digest.
    static func sha256Hex(_ data: Data) -> String {
        hex(SHA256.hash(data: data))
    }

    static func sha256Hex(of digest: SHA256.Digest) -> String {
        hex(digest)
    }

    private static let hexDigits: [UInt8] = Array("0123456789abcdef".utf8)

    /// Table-driven hex — runs once per chunk during seeding, where a per-byte
    /// `String(format:)` was an order of magnitude slower.
    static func hex<Bytes: Sequence>(_ bytes: Bytes) -> String where Bytes.Element == UInt8 {
        var out = [UInt8]()
        out.reserveCapacity(64)
        for byte in bytes {
            out.append(hexDigits[Int(byte >> 4)])
            out.append(hexDigits[Int(byte & 0x0F)])
        }
        return String(decoding: out, as: UTF8.self)
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
