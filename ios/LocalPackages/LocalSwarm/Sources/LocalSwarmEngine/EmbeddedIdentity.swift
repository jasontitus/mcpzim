import Foundation
import Security

/// A shared, self-signed P-256 identity embedded in the app. QUIC mandates TLS
/// 1.3 and a QUIC **listener requires a certificate** (external-PSK-only is
/// rejected), so both ends present this identity and pin it.
///
/// Trust model: identical to the shared TCP passcode — it gives **encryption**
/// plus **"LocalSwarm apps only" gating** (a peer must present this exact cert),
/// not per-device authentication. A future build can swap this for an ephemeral
/// per-launch identity verified out-of-band (pairing code / SAS).
///
/// **macOS uses a keychainless identity.** Importing the P12 into any keychain
/// makes the private key subject to an ACL + partition list, and Network.framework
/// signs the QUIC handshake from a *separate* system process the key doesn't
/// trust — so macOS prompts "LocalSwarm wants to use the … keychain" on first use.
/// Instead we build the SecIdentity from the raw cert + an **in-memory**
/// (non-permanent) key via `SecIdentityCreate`: no keychain, no ACL, no partition
/// list, no lock, and therefore no prompt. iOS has no such prompt and keeps the
/// straightforward P12 import.
enum EmbeddedIdentity {
    // Self-signed P-256 (CN=LocalSwarm), PKCS#12, password below (iOS path).
    private static let p12Base64 = "MIID0wIBAzCCA5kGCSqGSIb3DQEHAaCCA4oEggOGMIIDgjCCAk8GCSqGSIb3DQEHBqCCAkAwggI8AgEAMIICNQYJKoZIhvcNAQcBMBwGCiqGSIb3DQEMAQYwDgQIJZTq7rbXEEMCAggAgIICCELToGXj4pw2MDEWHjCvgk7iW98KSxg+oYrZroGdrsvoqi7dShVLe/cnKHT086Pttu2y0GX3Exk/nxfzFLCkiPdFb7fk9XzXg3FXnf5ioI2mz6JamPs3G/sdhp4JUDrLfc3W83HqydJ1ENcfzwADwajSAPrHOXcbHXGj/l+b1ceFG5TcPXARB87AkkwXGFonJZQwudBk0a41Sf1T671E13FZGPZyXJ+o9d+cGRpAXzvQdr9E2QA6TrsuM7Ym15OqIIE1zSL+YYhYS0CD8vDPFWCWL2yWtBVZ0zYUnfXkKPFEf6yb6Ko5uTJKtwzqnSfCbqusLMHakat6KbmYHig4+3+4Kmc7K0AAzyqimkHklnr6Sm7C1tp9z424iU8vogz+O/laibXDejqviArlP6INV71iiW2QngJcOMEX/URv+nEK6bPh3z6aMR9UaqHm+W438Am8cDB5cJpq9/GK/+Me5hjD3orl9Qn2XzPon+NxSAMEdMSjfOg344efMzv6mYs/4owUXtllO+gcTlmFBT6uw4xIrroBCMbUktutrUNp6WN2CegYdQWFDdSlAucgb9BgkeABHEAwcTjk/j+iwuq7MOhSRvf25y59zWJPPZmJc3Q1ErSPXGbAwv+0Tc+yPiSPAfLLmXuwkSvmWLBWdtTyQOYKdAig5IEaVJF51379ZPsV7dtsmfdMfvUwggErBgkqhkiG9w0BBwGgggEcBIIBGDCCARQwggEQBgsqhkiG9w0BDAoBAqCBtDCBsTAcBgoqhkiG9w0BDAEDMA4ECMdD9bA3IRViAgIIAASBkKSlIygpCvQNTAZUkuy1T/lTbr4MrD5Ge+obyVUscNJ8Sjl1v8aCqUqiYKaR5N3xuOvgHzOE6Vu6CLzZd8mmN8VMr2c+syFqgI4w9DR+F7fEGpnhBQBpDuNWRLnSMFoShqCbRmvYw7xlNvSbktae1pAhikbUWXfsmR3VA2iieanYWvvDQOjrrHvJkXLMOh7SSDFKMCMGCSqGSIb3DQEJFDEWHhQAbABvAGMAYQBsAHMAdwBhAHIAbTAjBgkqhkiG9w0BCRUxFgQUzS/bi9RtSY9cDH+jNeO9fdjK6ugwMTAhMAkGBSsOAwIaBQAEFAOcs2JEfgW0nCZAeQ7qJWaUIZQxBAiInVngoDGSSgICCAA="
    private static let password = "localswarm"

    // Raw building blocks for the keychainless macOS identity: the same
    // certificate (DER) and its P-256 private key in ANSI X9.63 form
    // (0x04 ‖ X ‖ Y ‖ K). Extracted from the P12 above, so the pinned cert is
    // identical across every peer.
    private static let certDERBase64 = "MIIBfzCCASWgAwIBAgIUVcMYtwENv1lsje2D+QKwvjosM8YwCgYIKoZIzj0EAwIwFTETMBEGA1UEAwwKTG9jYWxTd2FybTAeFw0yNjA2MjMxNjI5MTdaFw0zNjA2MjAxNjI5MTdaMBUxEzARBgNVBAMMCkxvY2FsU3dhcm0wWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATJigdPGR499+2Vb9a6rvpYhzpl5cY9zEw0eaMwpjxNEdcWmTY123ztkcP32ze1tEqyjhQ6FFSIlr3+YNBOACygo1MwUTAdBgNVHQ4EFgQUIVRdQk6nNjL6vBVZbZZTJJpxvW4wHwYDVR0jBBgwFoAUIVRdQk6nNjL6vBVZbZZTJJpxvW4wDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNIADBFAiEA+1uL75yF5z9AiBY4it2sYhopEj9dnlJleP1iwl1B/4YCIA8/Z5XDGYFQGwDunEmLpro5EC/QGPTk4Ly8fOiTxOsQ"
    private static let ecPrivateKeyBase64 = "BMmKB08ZHj337ZVv1rqu+liHOmXlxj3MTDR5ozCmPE0R1xaZNjXbfO2Rw/fbN7W0SrKOFDoUVIiWvf5g0E4ALKDz/tTOlQpFA+XiilAsaH8isuY+k95px2uGaP/1PRGTCw=="

    /// Loaded once, lazily and thread-safely.
    private static let cached: (identity: SecIdentity, certDER: Data)? = loadFromBundle()

    static var secIdentity: SecIdentity? { cached?.identity }
    static var pinnedCertDER: Data? { cached?.certDER }

    private static func loadFromBundle() -> (SecIdentity, Data)? {
#if os(macOS)
        return loadKeychainlessIdentity()
#else
        return loadFromP12()
#endif
    }

#if os(macOS)
    // `SecIdentityCreate(allocator, cert, key)` is SPI but present in the Security
    // framework on every Apple OS; it assembles a SecIdentity from a cert + key
    // with no keychain involvement.
    private typealias SecIdentityCreateFn =
        @convention(c) (CFAllocator?, SecCertificate, SecKey) -> Unmanaged<SecIdentity>?

    private static func loadKeychainlessIdentity() -> (SecIdentity, Data)? {
        guard let certDER = Data(base64Encoded: certDERBase64),
              let keyData = Data(base64Encoded: ecPrivateKeyBase64),
              let certificate = SecCertificateCreateWithData(nil, certDER as CFData) else {
            return nil
        }
        let attributes: [String: Any] = [
            kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
            kSecAttrKeyClass as String: kSecAttrKeyClassPrivate,
            kSecAttrIsPermanent as String: false, // in memory — never touches a keychain
        ]
        guard let privateKey = SecKeyCreateWithData(keyData as CFData, attributes as CFDictionary, nil),
              let handle = dlopen(nil, RTLD_NOW),
              let symbol = dlsym(handle, "SecIdentityCreate") else {
            return nil
        }
        let secIdentityCreate = unsafeBitCast(symbol, to: SecIdentityCreateFn.self)
        guard let identity = secIdentityCreate(nil, certificate, privateKey)?.takeRetainedValue() else {
            return nil
        }
        return (identity, certDER)
    }
#else
    private static func loadFromP12() -> (SecIdentity, Data)? {
        guard let data = Data(base64Encoded: p12Base64) else { return nil }
        let options: [String: Any] = [kSecImportExportPassphrase as String: password]
        var items: CFArray?
        guard SecPKCS12Import(data as CFData, options as CFDictionary, &items) == errSecSuccess,
              let array = items as? [[String: Any]],
              let identityRef = array.first?[kSecImportItemIdentity as String],
              CFGetTypeID(identityRef as CFTypeRef) == SecIdentityGetTypeID() else {
            return nil
        }
        let identity = identityRef as! SecIdentity
        var certificate: SecCertificate?
        guard SecIdentityCopyCertificate(identity, &certificate) == errSecSuccess,
              let certificate else {
            return nil
        }
        return (identity, SecCertificateCopyData(certificate) as Data)
    }
#endif
}
