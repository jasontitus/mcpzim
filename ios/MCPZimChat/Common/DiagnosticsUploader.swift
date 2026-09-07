// SPDX-License-Identifier: MIT
//
// Opt-in upload of finished session logs to Firebase Storage so they can be
// pulled into the off-device conversation corpus (tools/logpipe). Fires on
// backgrounding, over whatever network is available — a walking session lands
// without any tapping.
//
// PRIVACY: default OFF. The app's privacy policy promises that questions,
// transcripts, article titles, and GPS stay on device, and a debug log
// contains all of those. This uploader only runs when the user has explicitly
// turned on "Share debug logs for analysis" in Settings, so the default
// posture for TestFlight/App Store testers is unchanged. The toggle's label
// says exactly what leaves the device.

import Foundation
#if canImport(FirebaseStorage)
import FirebaseStorage
import UIKit

@MainActor
enum DiagnosticsUploader {
    /// User-facing opt-in. Read/written by the Settings toggle.
    static let optInKey = "diagnostics.uploadDebugLogs"

    static var isEnabled: Bool {
        UserDefaults.standard.bool(forKey: optInKey)
    }

    private static var activeUploads: [String: (id: UUID, task: StorageUploadTask)] = [:]

    static func setEnabled(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: optInKey)
        if !enabled {
            let tasks = activeUploads.values.map(\.task)
            activeUploads.removeAll()
            tasks.forEach { $0.cancel() }
        }
    }

    /// Stable-but-anonymous per-install id so the corpus can group a device's
    /// sessions without carrying anything that identifies the person. Reset if
    /// the app is reinstalled — that's fine, sessions are keyed by timestamp.
    private static var deviceTag: String {
        let k = "diagnostics.deviceTag"
        if let existing = UserDefaults.standard.string(forKey: k) { return existing }
        let tag = (UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString)
            .prefix(8).lowercased()
        UserDefaults.standard.set(String(tag), forKey: k)
        return String(tag)
    }

    /// Track which session files we've already uploaded so re-backgrounding
    /// doesn't re-send the whole archive every time.
    private static var uploadedKey: String { "diagnostics.uploadedLogs" }
    private static var uploaded: Set<String> {
        get { Set(UserDefaults.standard.stringArray(forKey: uploadedKey) ?? []) }
        set {
            // Bound the ledger; the on-device archive is pruned anyway.
            let trimmed = Array(newValue.suffix(200))
            UserDefaults.standard.set(trimmed, forKey: uploadedKey)
        }
    }

    /// Upload any completed session logs not yet sent. Call on background.
    /// The CURRENT session's file is skipped — it's still being written; it
    /// uploads on the next launch's pass once it's a previous session.
    static func uploadFinishedLogs(archive: LogArchive) {
        guard isEnabled, AppTelemetry.isConfigured else { return }
        // Directory enumeration can return a differently based/normalized
        // URL for the same iOS file. Session filenames are unique within this
        // one log directory and are already the upload/ledger identity.
        // Comparing URL representations leaked the growing current log into
        // uploads, which failed with HTTP 400 during the Napoleon capture.
        let currentName = archive.currentFileURL()?.lastPathComponent
        let files = archive.allFiles().filter { $0.lastPathComponent != currentName }
        let pending = files.filter {
            !uploaded.contains($0.lastPathComponent) && activeUploads[$0.lastPathComponent] == nil
        }
        guard !pending.isEmpty else { return }
        let storage = Storage.storage()
        for url in pending {
            let name = url.lastPathComponent
            let ref = storage.reference(withPath: "debug-logs/\(deviceTag)/\(name)")
            let meta = StorageMetadata()
            meta.contentType = "text/plain"
            let uploadID = UUID()
            let task = ref.putFile(from: url, metadata: meta) { _, error in
                Task { @MainActor in
                    guard activeUploads[name]?.id == uploadID else { return }
                    activeUploads[name] = nil
                    if let error {
                        NSLog("[Diagnostics] upload failed for \(name): \(error.localizedDescription)")
                    } else {
                        var u = uploaded; u.insert(name); uploaded = u
                        NSLog("[Diagnostics] uploaded \(name)")
                    }
                }
            }
            activeUploads[name] = (uploadID, task)
        }
    }
}
#else
@MainActor
enum DiagnosticsUploader {
    static let optInKey = "diagnostics.uploadDebugLogs"
    static var isEnabled: Bool { false }
    static func setEnabled(_ enabled: Bool) {}
    static func uploadFinishedLogs(archive: LogArchive) {}
}
#endif
