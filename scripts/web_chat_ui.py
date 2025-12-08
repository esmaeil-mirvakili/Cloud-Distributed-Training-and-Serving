#!/usr/bin/env python3
"""
Simple web chat UI for the llama serving stack.

- Serves a small HTML/JS page on a local port (default: 8080).
- Proxies POST /api/chat to the serving stack's /v1/chat/completions endpoint.
- Uses only the Python standard library (no extra deps).
"""

import argparse
import html
import http.server
import json
import os
import socketserver
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CHAT_ROUTE = "/v1/chat/completions"
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LLM Chat UI</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #0b223e;
      --accent: #22d3ee;
      --muted: #8ba7c6;
      --text: #e6ecf5;
      --card: #102a44;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% 20%, rgba(34, 211, 238, 0.08), transparent 25%), radial-gradient(circle at 80% 0%, rgba(34, 211, 238, 0.06), transparent 35%), var(--bg);
      font-family: "Fira Sans", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .shell {
      width: min(960px, 100%);
      background: linear-gradient(145deg, var(--panel), #0c1a2f 65%, var(--panel));
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.45);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 18px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      flex-wrap: wrap;
    }
    .title {
      font-weight: 700;
      font-size: 20px;
      letter-spacing: 0.4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 12px rgba(34, 211, 238, 0.9);
    }
    .target {
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    input, textarea {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--text);
      font-size: 14px;
      transition: border-color 0.15s ease, background 0.15s ease;
    }
    input:focus, textarea:focus {
      outline: none;
      border-color: rgba(34, 211, 238, 0.65);
      background: rgba(255,255,255,0.06);
    }
    textarea {
      resize: vertical;
      min-height: 60px;
      max-height: 200px;
    }
    .transcript {
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 14px;
      height: 340px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      scroll-behavior: smooth;
    }
    .bubble {
      padding: 12px 14px;
      border-radius: 12px;
      max-width: 90%;
      line-height: 1.45;
      white-space: pre-wrap;
      box-shadow: 0 12px 20px rgba(0,0,0,0.25);
    }
    .bubble.user {
      align-self: flex-end;
      background: linear-gradient(120deg, #1b75ff, #22d3ee);
      color: #0b1426;
    }
    .bubble.assistant {
      align-self: flex-start;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.05);
    }
    .status {
      font-size: 13px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .send-row {
      display: flex;
      gap: 12px;
      align-items: stretch;
    }
    .send-row textarea {
      flex: 1;
      margin: 0;
    }
    button {
      border: none;
      border-radius: 12px;
      padding: 0 18px;
      background: linear-gradient(135deg, #1b75ff, #22d3ee);
      color: #0b1426;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
      min-width: 90px;
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }
    button:not(:disabled):hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(27, 117, 255, 0.35);
    }
    @media (max-width: 640px) {
      .shell { padding: 14px; }
      .transcript { height: 280px; }
      .send-row { flex-direction: column; }
      button { width: 100%; height: 44px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="title"><span class="dot"></span>LLM Chat UI</div>
      <div class="target" id="target">Model endpoint: __LLAMA_TARGET__</div>
    </header>
    <div class="controls">
      <label>System prompt
        <input id="system" type="text" placeholder="e.g. You are a helpful assistant." value="__DEFAULT_SYSTEM__">
      </label>
      <label>Temperature
        <input id="temperature" type="number" step="0.1" min="0" max="2" value="__DEFAULT_TEMP__">
      </label>
      <label>Max tokens
        <input id="maxTokens" type="number" min="1" max="4096" value="__DEFAULT_MAX__">
      </label>
    </div>
    <div id="transcript" class="transcript"></div>
    <div class="send-row">
      <textarea id="prompt" placeholder="Ask anything..." rows="3"></textarea>
      <button id="send">Send</button>
    </div>
    <div class="status">
      <span id="statusText">Ready</span>
      <span>UI port: __UI_PORT__</span>
    </div>
  </div>
  <script>
    const DEFAULT_TEMP = __DEFAULT_TEMP__;
    const DEFAULT_MAX_TOKENS = __DEFAULT_MAX__;

    const transcript = document.getElementById("transcript");
    const promptInput = document.getElementById("prompt");
    const sendBtn = document.getElementById("send");
    const statusText = document.getElementById("statusText");
    const systemInput = document.getElementById("system");
    const tempInput = document.getElementById("temperature");
    const maxTokensInput = document.getElementById("maxTokens");

    function addBubble(role, text) {
      const div = document.createElement("div");
      div.className = "bubble " + role;
      div.textContent = text;
      transcript.appendChild(div);
      transcript.scrollTop = transcript.scrollHeight;
    }

    function setStatus(text) {
      statusText.textContent = text;
    }

    function buildMessages(userText) {
      const msgs = [];
      const sys = systemInput.value.trim();
      if (sys) msgs.push({ role: "system", content: sys });
      msgs.push({ role: "user", content: userText });
      return msgs;
    }

    async function sendMessage() {
      const text = promptInput.value.trim();
      if (!text) return;

      addBubble("user", text);
      promptInput.value = "";
      sendBtn.disabled = true;
      setStatus("Waiting for model...");

      const body = {
        messages: buildMessages(text),
        temperature: parseFloat(tempInput.value) || DEFAULT_TEMP,
        max_tokens: parseInt(maxTokensInput.value, 10) || DEFAULT_MAX_TOKENS
      };

      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(errText || `Request failed with status ${resp.status}`);
        }
        const data = await resp.json();
        const reply = data?.choices?.[0]?.message?.content ?? "(empty reply)";
        addBubble("assistant", reply);
        setStatus("Ready");
      } catch (err) {
        const msg = err?.message || String(err);
        addBubble("assistant", `Error: ${msg}`);
        setStatus("Error");
      } finally {
        sendBtn.disabled = false;
        promptInput.focus();
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        sendMessage();
        e.preventDefault();
      }
    });

    setStatus("Ready");
    promptInput.focus();
  </script>
</body>
</html>
"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def forward_chat(llama_server: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Proxy the chat completion to the serving stack."""
    body = json.dumps(payload).encode("utf-8")
    url = f"{llama_server.rstrip('/')}{CHAT_ROUTE}"
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def build_handler(llama_server: str, default_temp: float, default_max: int, ui_port: int, default_system: str, timeout: float):
    escaped_system = html.escape(default_system, quote=True)
    index_html = (
        INDEX_HTML.replace("__LLAMA_TARGET__", llama_server)
        .replace("__DEFAULT_TEMP__", str(default_temp))
        .replace("__DEFAULT_MAX__", str(default_max))
        .replace("__UI_PORT__", str(ui_port))
        .replace("__DEFAULT_SYSTEM__", escaped_system)
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            if self.path != "/api/chat":
                self.send_error(404, "Not found")
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                content = index_html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            if self.path == "/healthz":
                self._send_json({"status": "ok"})
                return
            self.send_error(404, "Not found")

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
                messages: List[Dict[str, str]] = body.get("messages") or []
                temperature = float(body.get("temperature", default_temp))
                max_tokens = int(body.get("max_tokens", default_max))
            except Exception as exc:
                self._send_json({"error": f"Invalid request: {exc}"}, status=400)
                return
            try:
                print(f"Sending: \n {messages}")
                payload = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                data = forward_chat(llama_server, payload, timeout=timeout)
                print(f"res: \n{data}")
                self._send_json(data, status=200)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=502)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Quieter logs
            return

    return Handler


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Web chat UI for the llama serving stack")
    parser.add_argument(
        "--llama-server",
        default=os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8000"),
        help="Base URL for the serving stack (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("LLAMA_UI_PORT", 8080),
        help="Port for the web UI (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("LLAMA_UI_HOST", "127.0.0.1"),
        help="Host/interface to bind for the UI (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--system",
        default=os.getenv("LLAMA_SYSTEM_PROMPT", ""),
        help="Optional default system prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float("LLAMA_TEMPERATURE", 0.7),
        help="Default sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_env_int("LLAMA_MAX_TOKENS", 32),
        help="Default max tokens for each response.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout when calling the serving stack.",
    )
    args = parser.parse_args()

    handler = build_handler(
        llama_server=args.llama_server,
        default_temp=args.temperature,
        default_max=args.max_tokens,
        ui_port=args.port,
        default_system=args.system,
        timeout=args.timeout,
    )
    server = ThreadedHTTPServer((args.host, args.port), handler)
    print(f"Web UI running on http://{args.host}:{args.port}")
    print(f"Proxying chat to {args.llama_server}{CHAT_ROUTE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down UI server...")
        server.server_close()


if __name__ == "__main__":
    main()
