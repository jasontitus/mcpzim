// SPDX-License-Identifier: MIT
//
// Single-shot real-ZIM probes for diagnosing fast-path / adapter
// failures without round-tripping to the phone. Both modes go
// through the same `MCPToolAdapter` the iOS app uses, so any fix
// that makes the probe succeed will also work on device.
//
// Modes:
//   --probe-compare --zim <path> --titles "A,B"
//       Dispatches `compare_articles` with the given titles and
//       prints which articles resolved / errored.
//   --probe-article --zim <path> --title "X"
//       Dispatches `article_overview` for a single title — the
//       simpler case that exercises `articleByTitle`'s path
//       strategy without the two-title relations probe wrapper.
//
// Example (the exact on-device failure from 2026-04-22):
//   MCPZimEvalCLI --probe-compare \
//     --zim /Users/jasontitus/Downloads/wikipedia_en_all_maxi_2025-10.zim \
//     --titles "north korea,south korea"

import Foundation
import MCPZimKit

enum ProbeCompareCLI {

    static func run(args inputArgs: [String]) async {
        var zim: String? = nil
        var titlesArg: String? = nil
        var args = inputArgs[...]
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--zim":
                zim = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--titles":
                titlesArg = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            default:
                FileHandle.standardError.write(Data(
                    "probe-compare: unknown argument \(a)\n".utf8))
                exit(2)
            }
        }
        guard let zim, !zim.isEmpty else {
            FileHandle.standardError.write(Data(
                "probe-compare: --zim <path> required\n".utf8))
            exit(2)
        }
        guard let titlesArg, !titlesArg.isEmpty else {
            FileHandle.standardError.write(Data(
                "probe-compare: --titles \"A,B[,C…]\" required\n".utf8))
            exit(2)
        }
        let titles = titlesArg.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let adapter = await makeAdapter(zimPath: zim)
        print("== probe-compare ==")
        print("zim:    \(zim)")
        print("titles: \(titles)")
        do {
            let out = try await adapter.dispatch(
                tool: "compare_articles",
                args: ["titles": titles]
            )
            print("keys: \(out.keys.sorted())")
            if let strategy = out["strategy"] as? String {
                print("strategy: \(strategy)")
            }
            if let resolvedTitle = out["resolved_title"] as? String {
                print("resolved_title: \(resolvedTitle)")
            }
            if let resolvedPath = out["path"] as? String {
                print("path: \(resolvedPath)")
            }
            prettyPrintCompare(out)
            // Also render the caption the iOS fast-path would show,
            // so this probe catches synth drift and not just dispatch.
            let caption = IntentRouter.synthesizeCompareReply(
                args: ["titles": titles], fullResult: out
            )
            print("---")
            print("caption that ships to the bubble:")
            print(caption)
            exit(0)
        } catch {
            print("dispatch threw: \(error)")
            exit(1)
        }
    }

    static func runArticleProbe(args inputArgs: [String]) async {
        var zim: String? = nil
        var title: String? = nil
        var args = inputArgs[...]
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--zim":
                zim = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--title":
                title = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            default:
                FileHandle.standardError.write(Data(
                    "probe-article: unknown argument \(a)\n".utf8))
                exit(2)
            }
        }
        guard let zim, !zim.isEmpty, let title, !title.isEmpty else {
            FileHandle.standardError.write(Data(
                "probe-article: --zim <path> --title <text> required\n".utf8))
            exit(2)
        }
        let adapter = await makeAdapter(zimPath: zim)
        print("== probe-article ==")
        print("zim:   \(zim)")
        print("title: \(title)")
        do {
            let out = try await adapter.dispatch(
                tool: "article_overview",
                args: ["title": title]
            )
            prettyPrintArticle(out)
            exit(0)
        } catch {
            print("dispatch threw: \(error)")
            exit(1)
        }
    }

    // MARK: - Adapter plumbing

    private static func makeAdapter(zimPath: String) async -> MCPToolAdapter {
        let url = URL(fileURLWithPath: zimPath)
        let fileName = url.lastPathComponent
        let reader: ZimReader
        do {
            reader = try LibzimReader(url: url)
        } catch {
            FileHandle.standardError.write(Data(
                "Could not open ZIM \(zimPath): \(error)\n".utf8))
            exit(3)
        }
        let service = DefaultZimService(readers: [(fileName, reader)])
        return await MCPToolAdapter(service: service, hasStreetzim: false)
    }

    // MARK: - Output formatting

    private static func prettyPrintCompare(_ result: [String: Any]) {
        if let err = result["error"] as? String {
            print("error: \(err)")
            return
        }
        let articles = (result["articles"] as? [[String: Any]]) ?? []
        print("count: \(articles.count)")
        for (i, a) in articles.enumerated() {
            let t = (a["title"] as? String) ?? "(no title)"
            if let e = a["error"] as? String {
                print("  [\(i)] \(t) → ERROR: \(e)")
                continue
            }
            let sections = (a["sections"] as? [[String: Any]]) ?? []
            let first = (sections.first?["text"] as? String) ?? ""
            let preview = String(first.prefix(120))
                .replacingOccurrences(of: "\n", with: " ")
            let path = (a["path"] as? String) ?? ""
            print("  [\(i)] \(t) (\(path))")
            print("       \(preview)…")
        }
    }

    private static func prettyPrintArticle(_ result: [String: Any]) {
        if let err = result["error"] as? String {
            print("error: \(err)")
            return
        }
        let title = (result["title"] as? String) ?? "(no title)"
        let path = (result["path"] as? String) ?? ""
        let sections = (result["sections"] as? [[String: Any]]) ?? []
        print("resolved title: \(title)")
        print("resolved path:  \(path)")
        print("sections:       \(sections.count)")
        if let first = sections.first?["text"] as? String {
            let preview = String(first.prefix(240))
                .replacingOccurrences(of: "\n", with: " ")
            print("lead preview:   \(preview)…")
        }
    }
}
