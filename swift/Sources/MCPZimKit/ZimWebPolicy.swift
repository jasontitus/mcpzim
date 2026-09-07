// SPDX-License-Identifier: MIT
import Foundation

/// Offline archive pages may use local JavaScript/MapLibre and blob workers,
/// but no embedded resource is allowed to contact an Internet origin.
public enum ZimWebPolicy {
    public static let contentSecurityPolicy = "default-src zim: data: blob:; script-src zim: blob: 'unsafe-inline' 'unsafe-eval'; style-src zim: 'unsafe-inline'; connect-src zim: blob:; img-src zim: data: blob:; font-src zim: data:; media-src zim: data: blob:; worker-src zim: blob:; frame-src zim:; object-src 'none'; base-uri zim:; form-action 'none'"

    public static func escapeAttribute(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    public static var meta: String {
        "<meta http-equiv=\"Content-Security-Policy\" content=\"\(escapeAttribute(contentSecurityPolicy))\">"
    }
}
