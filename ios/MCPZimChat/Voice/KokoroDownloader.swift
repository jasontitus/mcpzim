// SPDX-License-Identifier: MIT
//
// Foreground downloader for the Kokoro TTS assets listed in
// `KokoroAssets.downloads`. Uses URLSession download tasks so
// progress is reported per-file and the whole thing can be
// cancelled. Written to a temp location first and moved into
// place on success, so interrupted downloads don't leave a
// half-written model file that KokoroSwift would then try to
// parse.
//
// Background download (via URLSessionConfiguration.background) is
// a possible future upgrade. Keeping it simple for now — the
// first run takes a minute on Wi-Fi.

import Foundation
import Observation

@MainActor
@Observable
final class KokoroDownloader: NSObject {

    enum State: Equatable, Sendable {
        case idle
        case downloading(filename: String, bytesWritten: Int64, totalBytes: Int64, overall: Double)
        case finished
        case failed(String)
    }

    private(set) var state: State = .idle
    /// Sum of bytes written across all assets this run. Lets the UI
    /// show overall progress ("120 / 360 MB") without tracking
    /// per-file accounting.
    private(set) var cumulativeBytes: Int64 = 0

    /// Set from a Task when the caller wants to cancel — we check
    /// this between files. In-flight URLSession tasks can also be
    /// cancelled directly via `currentTask`.
    private var isCancelled = false
    private var currentTask: URLSessionDownloadTask?
    private var currentContinuation: CheckedContinuation<URL, Error>?

    /// Kick off the download for every asset not already present.
    /// Skips assets that already exist on disk — re-runs after a
    /// partial failure resume where they left off.
    func downloadIfNeeded() async {
        if KokoroAssets.isDownloaded {
            state = .finished
            return
        }
        isCancelled = false
        cumulativeBytes = KokoroAssets.currentBytesOnDisk
        for asset in KokoroAssets.downloads {
            if isCancelled {
                state = .failed("Cancelled")
                return
            }
            let dest = KokoroAssets.localURL(for: asset)
            if FileManager.default.fileExists(atPath: dest.path) {
                continue
            }
            do {
                let tmp = try await download(from: asset.url, advertising: asset.filename,
                                             expected: asset.expectedBytes)
                try FileManager.default.moveItem(at: tmp, to: dest)
                cumulativeBytes = KokoroAssets.currentBytesOnDisk
            } catch is CancellationError {
                state = .failed("Cancelled")
                return
            } catch {
                state = .failed(error.localizedDescription)
                return
            }
        }
        state = .finished
    }

    func cancel() {
        isCancelled = true
        currentTask?.cancel()
    }

    // MARK: - URLSession glue

    @ObservationIgnored
    private var urlSession: URLSession!

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60 * 30   // 30 min per file
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }

    private func download(from url: URL,
                          advertising filename: String,
                          expected: Int64) async throws -> URL {
        state = .downloading(filename: filename, bytesWritten: 0,
                             totalBytes: expected, overall: overallProgress(0))
        return try await withCheckedThrowingContinuation { cont in
            currentContinuation = cont
            let task = urlSession.downloadTask(with: url)
            currentTask = task
            task.resume()
        }
    }

    private func overallProgress(_ byteDelta: Int64) -> Double {
        let total = max(1, KokoroAssets.totalExpectedBytes)
        let done = cumulativeBytes + byteDelta
        return min(1.0, Double(done) / Double(total))
    }
}

extension KokoroDownloader: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession,
                                 downloadTask: URLSessionDownloadTask,
                                 didWriteData bytesWritten: Int64,
                                 totalBytesWritten: Int64,
                                 totalBytesExpectedToWrite: Int64) {
        Task { @MainActor [weak self] in
            guard let self, case let .downloading(name, _, total, _) = state else { return }
            let effectiveTotal = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : total
            state = .downloading(
                filename: name,
                bytesWritten: totalBytesWritten,
                totalBytes: effectiveTotal,
                overall: overallProgress(totalBytesWritten)
            )
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                 downloadTask: URLSessionDownloadTask,
                                 didFinishDownloadingTo location: URL) {
        // Move to a stable temp location inside NSTemporaryDirectory —
        // URLSession's default temp dir gets cleaned up immediately
        // after this delegate returns.
        let staging = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kokoro-dl-\(UUID().uuidString)")
        do {
            try FileManager.default.moveItem(at: location, to: staging)
            Task { @MainActor [weak self] in
                self?.currentContinuation?.resume(returning: staging)
                self?.currentContinuation = nil
                self?.currentTask = nil
            }
        } catch {
            Task { @MainActor [weak self] in
                self?.currentContinuation?.resume(throwing: error)
                self?.currentContinuation = nil
                self?.currentTask = nil
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                 didCompleteWithError error: Error?) {
        guard let error else { return }
        Task { @MainActor [weak self] in
            self?.currentContinuation?.resume(throwing: error)
            self?.currentContinuation = nil
            self?.currentTask = nil
        }
    }
}
