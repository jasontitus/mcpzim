// SPDX-License-Identifier: MIT
//
// Kokoro TTS asset management. The engine needs two files on disk:
//   * `kokoro-v1_0.safetensors` — the 82M-param bf16 weights (~312 MB)
//   * `voices.npz` — packed voice embeddings (~45 MB)
//
// We keep them in `Application Support/models/kokoro_mlx/` so they
// survive app upgrades and stay out of the user-visible Documents
// folder. URLs and layout mirror what CastCircle (a working iOS app
// we pulled from) uses, which means any future voice pack shipped
// by the Kokoro team or the KokoroTestApp upstream is one URL swap.

import Foundation

enum KokoroAssets {

    struct Download: Equatable, Sendable {
        let url: URL
        let filename: String
        /// Expected size in bytes when known; shown in the UI
        /// alongside real-time progress.
        let expectedBytes: Int64
    }

    /// bf16 model from the mlx-community mirror and the voices pack
    /// from the KokoroTestApp repo (same two URLs CastCircle uses).
    static let downloads: [Download] = [
        .init(
            url: URL(string:
                "https://huggingface.co/mlx-community/Kokoro-82M-bf16/resolve/main/kokoro-v1_0.safetensors"
            )!,
            filename: "kokoro-v1_0.safetensors",
            expectedBytes: 327_115_152
        ),
        .init(
            url: URL(string:
                "https://github.com/mlalma/KokoroTestApp/raw/main/Resources/voices.npz"
            )!,
            filename: "voices.npz",
            expectedBytes: 47_204_864 // ~45 MB; progress falls back to
                                        // HTTP content-length when available.
        ),
    ]

    /// `Application Support/models/kokoro_mlx/`. Created lazily on
    /// first access.
    static var modelDirectory: URL {
        let fm = FileManager.default
        let base = (try? fm.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        )) ?? fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
        let dir = base
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("kokoro_mlx", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func localURL(for download: Download) -> URL {
        modelDirectory.appendingPathComponent(download.filename)
    }

    /// True when every required file is present and non-empty.
    static var isDownloaded: Bool {
        for d in downloads {
            let url = localURL(for: d)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let size = attrs[.size] as? Int64, size > 0
            else { return false }
        }
        return true
    }

    /// Total bytes across all assets currently on disk — for the UI's
    /// storage line.
    static var currentBytesOnDisk: Int64 {
        downloads.reduce(0) { acc, d in
            let url = localURL(for: d)
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64 ?? 0
            return acc + size
        }
    }

    static var totalExpectedBytes: Int64 {
        downloads.reduce(0) { $0 + $1.expectedBytes }
    }

    /// Remove the downloaded files. Unloads anything in memory via
    /// the caller — this function just drops the bits on disk.
    static func deleteAll() throws {
        let dir = modelDirectory
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }
}
