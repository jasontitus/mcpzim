import Foundation
import CryptoKit

/// Proof a leecher presents to pull a PIN-protected share: SHA-256 over
/// "<swarmID>:<pin>", hex. Binding to the content-addressed swarmID stops a
/// token captured for one share being replayed to another, and the PIN itself
/// never crosses the wire — only this digest, inside the pinned-cert TLS
/// channel. Byte-for-byte identical to peer-go's AuthToken.
public func swarmAuthToken(swarmID: String, pin: String) -> String {
    let digest = SHA256.hash(data: Data((swarmID + ":" + pin).utf8))
    return Hashing.sha256Hex(of: digest)
}

extension String {
    /// Constant-time equality for auth tokens (fixed-length hex digests), so a
    /// wrong PIN can't be narrowed down by response timing.
    func utf8CStringConstantTimeEquals(_ other: String) -> Bool {
        let a = Array(utf8), b = Array(other.utf8)
        guard a.count == b.count else { return false }
        var diff: UInt8 = 0
        for i in 0..<a.count { diff |= a[i] ^ b[i] }
        return diff == 0
    }
}
