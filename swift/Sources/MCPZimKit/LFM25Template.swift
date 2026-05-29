// SPDX-License-Identifier: MIT
//
// `ModelTemplate` for LiquidAI LFM2.5-8B-A1B (fine-tuned, Q3_K_M GGUF
// via llama.cpp). LFM2.5 uses ChatML turn markers
// (`<|im_start|>` / `<|im_end|>`) like Qwen, with `<|startoftext|>` as
// its BOS (added automatically by llama.cpp, so `bos` stays empty here,
// same as `QwenChatMLTemplate`).
//
// Tool-call CONTENT is identical to the Gemma 3 path: our LFM2.5 LoRA
// (`tools/fine-tune` v7) was trained on the SAME `train_v4_combined +
// chains` corpus the Gemma 3 4B FT used — system prose + tool block
// folded into the first user turn, JSON tool calls. Only the chat
// template's turn markers differ (mlx-lm wraps the messages in LFM2.5's
// ChatML at train time vs Gemma's `<start_of_turn>`). So this template
// delegates every body/parse concern to a `Gemma3Template` instance and
// overrides only the turn markers, BOS, and stop token.
//
// Eval: v7 Q3_K_M = 12/13 on the llama-smoke 13-scenario grid (beats the
// Gemma 3 4B FT V7C's 10/13), 4.16 GB peak RSS, q8_0 KV. See
// tools/llama-smoke/LFM25_MEMORY_PERF_FRONTIER.md.

import Foundation

public struct LFM25Template: ModelTemplate {

    /// Shared body/parse logic. LFM2.5's fine-tune trained on the exact
    /// Gemma-3-FT corpus, so tool prose, JSON tool-call parsing, the
    /// repair passes, and tool-response rendering are byte-identical —
    /// only the turn markers below change.
    private let body = Gemma3Template()

    public init() {}

    /// llama.cpp auto-prepends LFM2.5's `<|startoftext|>` BOS, so the
    /// rendered transcript must NOT include it (else double-BOS).
    public var bos: String { "" }

    /// ChatML closes every turn with `<|im_end|>`.
    public var stopMarkers: [String] { ["<|im_end|>"] }

    public var logCategory: String { "LFM2.5" }

    /// llama.cpp KV cache is reusable across turns (no MLX stale-state
    /// bug); the host gets delta-prefill on follow-ups.
    public var hasStaleScratchStateBug: Bool { false }

    public func formatSystemTurn(
        systemMessage: String, tools: [ModelToolDeclaration]
    ) -> String {
        body.formatSystemTurn(systemMessage: systemMessage, tools: tools)
    }

    /// Same fold-into-first-user-turn structure as Gemma 3 (LFM2.5 has a
    /// system role, but the FT corpus folds system+tools into the first
    /// user turn, so we match that exactly), with ChatML turn markers.
    public func renderTranscript(
        systemPreamble: String,
        tools: [ModelToolDeclaration],
        turns: [ChatTurn]
    ) -> String {
        let sysBody = formatSystemTurn(systemMessage: systemPreamble, tools: tools)
        var out = bos                     // empty — llama.cpp adds <|startoftext|>
        var pendingSys = sysBody

        for t in turns {
            switch t.role {
            case .system:
                if pendingSys.isEmpty {
                    pendingSys = t.text
                } else {
                    pendingSys += "\n\n" + t.text
                }
            case .user:
                var bodyText = t.text
                if !pendingSys.isEmpty {
                    bodyText = pendingSys + "\n\n" + bodyText
                    pendingSys = ""
                }
                out += "<|im_start|>user\n\(bodyText)<|im_end|>\n"
            case .assistant:
                out += "<|im_start|>assistant\n\(t.text)<|im_end|>\n"
            case .tool:
                // Tool responses come back as user turns (matches the FT
                // corpus); the body is already rendered by
                // `formatToolResponse`.
                out += "<|im_start|>user\n\(t.text)<|im_end|>\n"
            }
        }

        if !pendingSys.isEmpty {
            out += "<|im_start|>user\n\(pendingSys)<|im_end|>\n"
        }

        // Open the assistant turn for generation.
        out += "<|im_start|>assistant\n"
        return out
    }

    public func firstToolCall(in buffer: String) -> ToolCallMatch? {
        body.firstToolCall(in: buffer)
    }

    public func firstToolCallAfterClip(in buffer: String) -> ToolCallMatch? {
        body.firstToolCallAfterClip(in: buffer)
    }

    public func formatToolCall(name: String, arguments: [String: Any]) -> String {
        body.formatToolCall(name: name, arguments: arguments)
    }

    public func formatToolResponse(name: String, payload: [String: Any]) -> String {
        body.formatToolResponse(name: name, payload: payload)
    }

    /// LFM2.5 (and its base) can emit `<think>…</think>` reasoning. The FT
    /// corpus has none, so the tuned model rarely does, but strip closed
    /// spans defensively — same contract as Qwen's reasoning strip.
    public func stripReasoning(_ text: String) -> String {
        guard let open = text.range(of: "<think>"),
              let close = text.range(of: "</think>",
                                     range: open.upperBound..<text.endIndex)
        else { return text }
        var out = text
        out.removeSubrange(open.lowerBound..<close.upperBound)
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
