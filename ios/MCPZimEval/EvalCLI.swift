// SPDX-License-Identifier: MIT
//
// Entry point for `MCPZimEvalCLI` — a headless command-line binary
// that runs the multi-model eval harness without injecting into
// `MCPZimChatMac.app`. Compared to the old `MCPZimChatMacTests` flow
// this gets us:
//
//   - no SwiftUI app standing up a second `ChatSession` in-process
//     (the old setup crashed MLX when both competed for the GPU)
//   - real stdout / exit codes for shell scripting and CI
//   - no Xcode test runner overhead — just xcodebuild + run the binary
//
// Build:  xcodebuild -scheme MCPZimEvalCLI -destination 'platform=macOS' build
// Run:    build/.../MCPZimEvalCLI [--variant qwen] [--scenario compare]

import Foundation

@main
struct EvalCLI {
    static func main() async {
        var opts = EvalHarness.RunOptions()
        var args = CommandLine.arguments.dropFirst()
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--variant":
                guard let v = args.first else { Self.usage() }
                args = args.dropFirst()
                opts.variantFilter.append(v)
            case "--scenario":
                guard let s = args.first else { Self.usage() }
                args = args.dropFirst()
                opts.scenarioFilter.append(s)
            case "-h", "--help":
                Self.usage()
            default:
                FileHandle.standardError.write(Data("unknown argument: \(a)\n".utf8))
                exit(2)
            }
        }
        let code = await runHarness(opts)
        exit(code)
    }

    private static func usage() -> Never {
        let msg = """
        Usage: MCPZimEvalCLI [--variant <substring>] [--scenario <substring>]

        Options:
          --variant   Only run variants whose id contains this substring.
                      May be repeated. Empty = all cached variants.
          --scenario  Only run scenarios whose name contains this substring.
                      May be repeated. Empty = all scenarios.
          -h, --help  Print this and exit 0.
        """
        print(msg)
        exit(0)
    }

    @MainActor
    private static func runHarness(_ opts: EvalHarness.RunOptions) async -> Int32 {
        let harness = EvalHarness()
        do {
            let result = try await harness.run(opts)
            print(result.scorecard.markdown())
            if result.scorecard.rows.isEmpty {
                FileHandle.standardError.write(Data(
                    "[eval] no cached variants matched — nothing to score.\n".utf8))
                return 2
            }
            if !result.scenariosWithNoWinner.isEmpty {
                let names = result.scenariosWithNoWinner.joined(separator: ", ")
                FileHandle.standardError.write(Data(
                    "[eval] \(result.scenariosWithNoWinner.count) scenario(s) had no passing variant: \(names)\n".utf8))
                return 1
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data("[eval] \(error)\n".utf8))
            return 3
        }
    }
}
