"""Provider clients against a tiny local HTTP server that mimics Ollama and OpenAI-style APIs."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from citebench.llm import LLMError, OllamaLLM, OpenAICompatibleLLM


class Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((self.path, body, dict(self.headers)))
        if self.path == "/api/chat":
            out = {"message": {"role": "assistant", "content": '{"ok": "ollama"}'}}
        elif self.path == "/v1/chat/completions":
            out = {"choices": [{"message": {"content": '{"ok": "openai"}'}}]}
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_native_api(server):
    Handler.seen.clear()
    llm = OllamaLLM(model="qwen2.5:7b", base_url=server + "/v1", num_ctx=8192)  # /v1 suffix is tolerated
    assert llm.complete("hi", system="sys") == '{"ok": "ollama"}'
    path, body, _ = Handler.seen[-1]
    assert path == "/api/chat"
    assert body["options"]["num_ctx"] == 8192 and body["format"] == "json" and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_compatible(server):
    Handler.seen.clear()
    llm = OpenAICompatibleLLM(model="m", base_url=server + "/v1", api_key="k")
    assert llm.complete("hi") == '{"ok": "openai"}'
    path, body, headers = Handler.seen[-1]
    assert path == "/v1/chat/completions"
    assert body["response_format"] == {"type": "json_object"}
    assert headers["Authorization"] == "Bearer k"


def test_http_error_is_readable(server):
    llm = OpenAICompatibleLLM(model="m", base_url=server + "/nope")
    with pytest.raises(LLMError, match="HTTP 404"):
        llm.complete("hi")


def test_unreachable_server():
    with pytest.raises(LLMError, match="Is the server running"):
        OllamaLLM(base_url="http://127.0.0.1:9").complete("hi")


def test_openai_compatible_needs_config():
    with pytest.raises(LLMError):
        OpenAICompatibleLLM(model="", base_url="")
