# Android integration

[Google's AI Edge Gallery](https://github.com/google-ai-edge/gallery) is the
easiest path on Android: it's the official reference for on-device Gemma 4
with native tool-calling via LiteRT-LM. The iOS side of that project is closed
source, but the **Android source is open (Apache 2.0) and tool-callable** — so
the mobile integration is a small fork of the Android app.

## Minimal fork: teach Gemma on-device to call your MCP server

The built-in Agent Chat module registers three hard-coded tools on
`AgentTools` (`load_skill`, `run_js`, `run_intent`). Add a fourth that speaks
MCP/HTTP to your Python `mcpzim` server running on the LAN (or on-device via
Termux):

```kotlin
// app/src/main/java/com/google/ai/edge/gallery/customtasks/agentchat/AgentTools.kt
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import java.net.HttpURLConnection
import java.net.URL

class AgentTools(
    // ... existing fields
) {
    @Tool(description = "Call a tool on a remote MCP server that exposes ZIM-backed knowledge (Wikipedia, mdwiki, streetzim, ...). Use `list_libraries` first to see what's available; then e.g. `plan_driving_route` for routing questions.")
    fun callMcp(
        @ToolParam(description = "Base URL of the MCP server, e.g. http://192.168.1.20:8765/mcp")
        server: String,
        @ToolParam(description = "Tool name, e.g. plan_driving_route")
        toolName: String,
        @ToolParam(description = "JSON-encoded arguments object, e.g. {\"origin_lat\":42.36,...}")
        argsJson: String,
    ): Map<String, Any?> {
        val payload = """
            {"jsonrpc":"2.0","id":1,"method":"tools/call",
             "params":{"name":"$toolName","arguments":$argsJson}}
        """.trimIndent()
        val conn = URL(server).openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        conn.outputStream.use { it.write(payload.toByteArray()) }
        val body = conn.inputStream.bufferedReader().readText()
        return Json.parseToJsonElement(body).jsonObject.toMap()
    }
}
```

Then ship a `skills/call-mcp/SKILL.md` that tells Gemma how to use it, so the
model picks `call_mcp` for offline-knowledge / routing questions. Rebuild with
`./gradlew installDebug` and you have an on-device Gemma 4 that can ask your
Python server for driving routes, Wikipedia articles, medical answers, etc.

## Why an external MCP server?

Two practical reasons:

1. **Fits existing AI Edge Gallery idioms.** The app already has a Skills
   system that pulls in JS and intents from external assets — reaching out to
   a JSON-RPC server over HTTP is well within its comfort zone.
2. **ZIMs are large.** A full Wikipedia ZIM is 90 GB; even a streetzim is
   tens to hundreds of MB. Keeping them on a home server (or a Raspberry Pi)
   and having the phone make LAN calls keeps the app small and the phone cold.

## On-device option (Termux)

If you want everything on-device (no LAN, for true offline agents):

1. Install [Termux](https://termux.dev).
2. `pkg install python clang cmake && pip install mcpzim`.
3. `mcpzim --transport streamable-http --host 127.0.0.1 --port 8765 /sdcard/zims`.
4. In your forked AI Edge Gallery build, point the `callMcp` tool at
   `http://127.0.0.1:8765/mcp`.

This sidesteps iOS entirely (which has no Termux equivalent — hence the
native Swift path via `swift/MCPZimKit`).

## What you don't need

You do NOT need to modify LiteRT-LM, swap out the Gemma 4 weights, or write a
SKILL.md at all if you just want the Python server's behaviour. The native
Kotlin tool + a brief system-prompt instruction ("for offline knowledge or
driving routes, call `call_mcp`") is enough.
