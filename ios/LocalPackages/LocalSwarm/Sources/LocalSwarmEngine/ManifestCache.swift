import Foundation
import CryptoKit

/// Persists built manifests so re-hosting unchanged files skips the Chunker.
///
/// Hashing a 133 GB file takes minutes and runs on every app launch (the app
/// re-hosts remembered shares at startup), which also delays advertising and
/// churns the listener port. A cache entry is keyed by every source file's
/// (absolute path, size, mtime) plus the chunk size and display name — any
/// change produces a different key, so a stale hit is impossible without
/// mtime forgery. Entries are small JSON files (the manifest + the ordered
/// source paths) under Application Support.
enum ManifestCache {

    private struct Entry: Codable {
        let manifest: SwarmManifest
        let orderedPaths: [String]
    }

    /// Cache key for this exact hosting request, or nil if any file can't be
    /// stat'ed (caller falls through to a normal build, which will throw a
    /// proper error). Keyed per item so the same files shared under a
    /// different relative-path layout (flat vs. as a directory) can never
    /// alias. "v2": v1 predates relative paths.
    static func key(name: String, items: [ShareItem], chunkSize: Int = Chunker.defaultChunkSize) -> String? {
        var parts: [String] = ["v2", name, String(chunkSize)]
        for item in items.sorted(by: { $0.url.path < $1.url.path }) {
            // NB: not URL.resourceValues — NSURL caches those per instance, so a
            // re-stat of a changed file can return stale size/mtime and turn a
            // must-miss into a hit. FileManager stats fresh every call.
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: item.url.path),
                  let size = attrs[.size] as? Int64,
                  let mtime = attrs[.modificationDate] as? Date else { return nil }
            parts.append(item.url.path)
            parts.append(item.relativePath)
            parts.append(String(size))
            parts.append(String(mtime.timeIntervalSince1970))
        }
        return Hashing.sha256Hex(of: SHA256.hash(data: Data(parts.joined(separator: "|").utf8)))
    }

    /// Returns the cached manifest + ordered source URLs for an unchanged set
    /// of files, or nil on any miss/mismatch.
    static func lookup(name: String, items: [ShareItem]) -> (SwarmManifest, [URL])? {
        guard let key = key(name: name, items: items),
              let data = try? Data(contentsOf: fileURL(for: key)),
              let entry = try? JSONDecoder().decode(Entry.self, from: data) else { return nil }
        // The key already encodes path+size+mtime; re-check existence so a
        // deleted-and-recreated file with a forged mtime still has to exist.
        let ordered = entry.orderedPaths.map { URL(fileURLWithPath: $0) }
        guard ordered.count == entry.manifest.files.count,
              ordered.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else { return nil }
        return (entry.manifest, ordered)
    }

    /// Persists a freshly built manifest. Best-effort — a failed save only
    /// costs a future re-hash.
    static func store(manifest: SwarmManifest, ordered: [URL], name: String, items: [ShareItem]) {
        guard let key = key(name: name, items: items) else { return }
        let entry = Entry(manifest: manifest, orderedPaths: ordered.map(\.path))
        guard let data = try? JSONEncoder().encode(entry) else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: fileURL(for: key), options: .atomic)
    }

    /// Flat-share conveniences (bare filenames as relative paths).
    static func lookup(name: String, urls: [URL]) -> (SwarmManifest, [URL])? {
        lookup(name: name, items: urls.map { ShareItem(url: $0) })
    }

    static func store(manifest: SwarmManifest, ordered: [URL], name: String, urls: [URL]) {
        store(manifest: manifest, ordered: ordered, name: name,
              items: urls.map { ShareItem(url: $0) })
    }

    /// Wipe the cache (also handy for tests).
    static func clear() {
        try? FileManager.default.removeItem(at: directory)
    }

    private static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("localswarm-manifest-cache", isDirectory: true)
    }

    private static func fileURL(for key: String) -> URL {
        directory.appendingPathComponent(key + ".json")
    }
}
