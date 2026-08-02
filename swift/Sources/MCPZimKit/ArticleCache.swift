// SPDX-License-Identifier: MIT
//
// Small LRU over recently fetched articles, keyed (zim, path). One
// conversational turn re-reads the same article 2–4×: `sectionsByTitle`
// fetches once for the title probe and again for the full section list,
// `article_overview` re-reads the body for related links + hatnotes,
// `get_article_section` re-reads it for related links, and
// `narrate_article` re-resolves the whole article on every "continue"
// page. Holding the last few raw bodies plus their parsed section lists
// collapses each of those to one libzim read + one parse. Capacity stays
// tiny — a handful of Wikipedia bodies is a few MB — so this cannot grow
// into jetsam territory the way the unbounded place-chunk cache once did.

import Foundation

actor ArticleCache {
    struct Entry: Sendable {
        /// Canonical entry path as the reader reported it (may differ
        /// from the requested path when the ZIM resolved a redirect).
        let path: String
        let title: String
        let mimetype: String
        let html: String
        /// Raw content size — preserved separately because `html` is
        /// empty for non-UTF-8 entries while `bytes` must still report
        /// the real payload size.
        let bytes: Int
        /// Parsed lazily: `get_article` alone never needs sections, so
        /// storing the body must not force a parse.
        var sections: [ArticleSection]?
    }

    private struct Key: Hashable {
        let zim: String
        let path: String
    }

    private let capacity: Int
    private var entries: [Key: Entry] = [:]
    /// Most-recently-used last — same bookkeeping as the service's
    /// chunk LRU.
    private var order: [Key] = []

    init(capacity: Int = 4) {
        self.capacity = max(1, capacity)
    }

    func entry(zim: String, path: String) -> Entry? {
        let key = Key(zim: zim, path: path)
        guard let hit = entries[key] else { return nil }
        touch(key)
        return hit
    }

    func store(zim: String, path: String, entry: Entry) {
        let key = Key(zim: zim, path: path)
        if entries.updateValue(entry, forKey: key) == nil {
            order.append(key)
            while order.count > capacity {
                entries.removeValue(forKey: order.removeFirst())
            }
        } else {
            touch(key)
        }
    }

    /// Attach parsed sections to an already-cached entry. No-op when
    /// the entry was evicted in the meantime — the sections are still
    /// returned to the caller, they just aren't retained.
    func setSections(_ sections: [ArticleSection], zim: String, path: String) {
        let key = Key(zim: zim, path: path)
        guard var hit = entries[key] else { return }
        hit.sections = sections
        entries[key] = hit
    }

    private func touch(_ key: Key) {
        guard let idx = order.firstIndex(of: key), idx != order.count - 1
        else { return }
        order.append(order.remove(at: idx))
    }
}
