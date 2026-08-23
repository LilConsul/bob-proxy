"""
Local HMAC-signing proxy for Bob's LiteLLM backend (API Key version).

Uses a Bob API key directly — no SSO interception or `bob` CLI needed.
Much simpler and faster startup than the SSO-based proxy.py.

Claude Code (Anthropic mode):
  ANTHROPIC_BASE_URL=http://localhost:8989/anthropic ANTHROPIC_API_KEY=proxy claude --model premium

Claude Code (Bedrock mode — no Claude Pro login needed):
  CLAUDE_CODE_USE_BEDROCK=1 ANTHROPIC_BEDROCK_BASE_URL=http://localhost:8989/bedrock CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 claude --model premium

GitHub Copilot setup (VS Code settings.json):
    "github.copilot.chat.models": [
        {
            "name": "Bob",
            "vendor": "customoai",
            "apiKey": "${input:chat.lm.secret.<your-secret-id>}",
            "models": [
                {
                    "id": "premium",
                    "name": "Bob Premium",
                    "url": "http://localhost:8989/v1",
                    "toolCalling": true,
                    "vision": true,
                    "maxInputTokens": 128000,
                    "maxOutputTokens": 16000
                }
            ]
        }
    ]

Usage:
    1. Copy .env.sample to .env and set your API key
    2. uv run python proxy.py

Listens on http://localhost:8989 and forwards to Bob's backend with
HMAC signing + API key auth injected into every request.
"""

import contextlib
import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

# Load .env file if present
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

BASE_URL = "https://api.us-east.bob.ibm.com"
HMAC_SECRET = "c102b927-013e-42e4-bc65-16625f35807a"
PORT = 8989

# Path prefix used by Bob's new gateway for OpenAI-style endpoints.
INFERENCE_PREFIX = "/inference/v1"

# API key — loaded from .env or environment variable
API_KEY = os.environ.get("BOB_API_KEY", "")
INSTANCE_ID = os.environ.get("BOB_INSTANCE_ID", "")
TEAM_ID = os.environ.get("BOB_TEAM_ID", "")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bob-proxy")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _print_banner()
    yield


app = FastAPI(lifespan=lifespan)

# Map any model name to the backend alias.
# Well-known OpenAI aliases included so clients like JetBrains AI Assistant
# (which only accept recognised model IDs) can connect.
MODEL_MAP = {
    "premium": "premium",
    "premium-fallback": "premium-fallback",
    "premium-shell": "premium-shell",
    "premium-ide": "premium-ide",
    "mistral-medium-3.1": "mistral-medium-3.1",
    "granite-8b-code-instruct": "granite-8b-code-instruct",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "codestral-2508": "codestral-2508",
    "rnj-1-test": "rnj-1-test",
    "sonnet-4.5-global-ibm-only": "sonnet-4.5-global-ibm-only",
    "sonnet-4.5": "sonnet-4.5",
    "explorer": "explorer",
    "haiku-4.5": "haiku-4.5",
    "gpt-2026-5.4": "gpt-2026-5.4",
    "sonnet-4.6": "sonnet-4.6",
    # OpenAI aliases — map to premium
    "gpt-4o": "premium",
    "gpt-4o-mini": "premium",
    "gpt-4": "premium",
    "gpt-4-turbo": "premium",
    "gpt-3.5-turbo": "premium",
}
DEFAULT_MODEL = "premium"


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------
def sign_request(method: str, path: str, body: str = "") -> dict[str, str]:
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    message = f"{method.upper()}\n{path}\n{timestamp}\n{body_hash}"
    signature = hmac.new(HMAC_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return {"x-request-timestamp": timestamp, "x-request-signature": signature}


# ---------------------------------------------------------------------------
# Anthropic <-> OpenAI translation helpers
# ---------------------------------------------------------------------------
def _sse(event_type: str, data) -> bytes:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event_type}\ndata: {payload}\n\n".encode()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _translate_content_blocks(blocks) -> str | list:
    if isinstance(blocks, str):
        return blocks
    parts = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif btype == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                data_uri = f"data:{source['media_type']};base64,{source['data']}"
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        elif btype in ("thinking", "tool_use", "tool_result"):
            continue
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts if parts else ""


def _translate_messages(messages: list, system: str | list | None) -> list:
    out: list[dict] = []
    if system:
        if isinstance(system, list):
            text_parts = [b["text"] for b in system if b.get("type") == "text"]
            system = "\n\n".join(text_parts)
        out.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        if tool_results and role == "user":
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    text_parts = [b.get("text", "") for b in tr_content if b.get("type") == "text"]
                    tr_content = "\n".join(text_parts) if text_parts else ""
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr_content,
                    }
                )
            other_blocks = [b for b in content if b.get("type") not in ("tool_result", "thinking")]
            if other_blocks:
                out.append({"role": "user", "content": _translate_content_blocks(other_blocks)})
            continue
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if tool_uses and role == "assistant":
            text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            text_content = "\n".join(t for t in text_parts if t) or None
            tool_calls = [
                {
                    "id": tu.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for tu in tool_uses
            ]
            out.append({"role": "assistant", "content": text_content, "tool_calls": tool_calls})
            continue
        out.append({"role": role, "content": _translate_content_blocks(content)})
    return out


def _translate_tools(tools: list | None) -> list | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _translate_tool_choice(tc) -> str | dict | None:
    if tc is None:
        return None
    if isinstance(tc, str):
        return {"auto": "auto", "any": "required", "none": "none"}.get(tc, "auto")
    if isinstance(tc, dict):
        tc_type = tc.get("type", "auto")
        if tc_type == "any":
            return "required"
        if tc_type == "tool":
            name = tc.get("name") or tc.get("function", {}).get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        return "auto"
    return None


def _anthropic_request_to_openai(body: dict) -> dict:
    out: dict = {}
    original_model = body.get("model", "")
    out["model"] = MODEL_MAP.get(original_model, DEFAULT_MODEL)
    if out["model"] != original_model:
        logger.info("[anthropic->openai] Model rewrite: %s -> %s", original_model, out["model"])
    out["messages"] = _translate_messages(body.get("messages", []), body.get("system"))
    tools = _translate_tools(body.get("tools"))
    if tools:
        out["tools"] = tools
    tc = _translate_tool_choice(body.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc
    for key, okey in [
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ]:
        if key in body:
            out[okey] = body[key]
    if "temperature" in out and "top_p" in out:
        del out["top_p"]
    if body.get("stream", False):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _openai_response_to_anthropic(resp_json: dict, model: str) -> dict:
    choice = resp_json.get("choices", [{}])[0]
    message = choice.get("message", {})
    content_blocks = []
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": args,
            }
        )
    finish = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(finish, "end_turn")
    usage = resp_json.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _stream_openai_to_anthropic(resp: httpx.Response, model: str, input_tokens: int):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    block_index = 0
    current_text_open = False
    tool_calls_in_progress: dict[int, dict] = {}
    output_tokens = 0
    stop_reason = "end_turn"

    for raw_line in resp.iter_lines():
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data_str = raw_line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if chunk.get("usage"):
            output_tokens = chunk["usage"].get("completion_tokens", output_tokens)
            input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            stop_reason = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }.get(finish_reason, "end_turn")

        text_delta = delta.get("content")
        if text_delta:
            if not current_text_open:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                current_text_open = True
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text_delta},
                },
            )

        if delta.get("tool_calls"):
            if current_text_open:
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_index},
                )
                block_index += 1
                current_text_open = False
            for tc_delta in delta["tool_calls"]:
                tc_index = tc_delta.get("index", 0)
                if tc_index not in tool_calls_in_progress:
                    fn = tc_delta.get("function", {})
                    tc_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    tc_name = fn.get("name", "")
                    tool_calls_in_progress[tc_index] = {
                        "id": tc_id,
                        "name": tc_name,
                        "args": "",
                        "block_index": block_index,
                    }
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": {},
                            },
                        },
                    )
                    block_index += 1
                args_fragment = tc_delta.get("function", {}).get("arguments", "")
                if args_fragment:
                    tool_calls_in_progress[tc_index]["args"] += args_fragment
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_calls_in_progress[tc_index]["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_fragment,
                            },
                        },
                    )

    if current_text_open:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
    for tc_info in tool_calls_in_progress.values():
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": tc_info["block_index"]},
        )

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Shared HTTP client (connection pooling, keep-alive)
# ---------------------------------------------------------------------------
_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=300)
    return _client


def _backend_headers(method: str, path: str, body: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Apikey {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "bobshell/1.0.4",
    }
    if INSTANCE_ID:
        headers["x-instance-id"] = INSTANCE_ID
    if TEAM_ID:
        headers["x-team-id"] = TEAM_ID
    headers.update(sign_request(method, path, body))
    return headers


# ---------------------------------------------------------------------------
# /v1/models — JetBrains and other OpenAI-compatible clients probe this first
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    models = [{"id": name, "object": "model", "created": 0, "owned_by": "bob"} for name in MODEL_MAP]
    return JSONResponse({"object": "list", "data": models})


# ---------------------------------------------------------------------------
# Anthropic routes
# ---------------------------------------------------------------------------
@app.post("/anthropic/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    body = await request.body()
    tokens = _estimate_tokens(body.decode())
    return JSONResponse({"input_tokens": tokens})


@app.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request):
    raw_body = await request.body()
    try:
        anthropic_body = json.loads(raw_body)
    except json.JSONDecodeError as e:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e)},
            },
            status_code=400,
        )

    original_model = anthropic_body.get("model", "unknown")
    is_stream = anthropic_body.get("stream", False)
    openai_body = _anthropic_request_to_openai(anthropic_body)
    openai_json = json.dumps(openai_body)

    logger.info(
        "[anthropic->openai] model=%s, stream=%s, messages=%s, tools=%s, tool_choice=%r",
        original_model,
        is_stream,
        len(anthropic_body.get("messages", [])),
        len(anthropic_body.get("tools", [])),
        anthropic_body.get("tool_choice"),
    )

    target_path = f"{INFERENCE_PREFIX}/chat/completions"
    headers = _backend_headers("POST", target_path, openai_json)

    if is_stream:
        resp = get_client().build_request(
            "POST",
            f"{BASE_URL}{target_path}",
            headers=headers,
            content=openai_json,
        )
        backend_resp = get_client().send(resp, stream=True)
        logger.info("[anthropic->openai] POST %s -> %s", target_path, backend_resp.status_code)
        if backend_resp.status_code >= 400:
            error_text = backend_resp.text[:1000]
            logger.error("[anthropic->openai] Error: %s", error_text)
            backend_resp.close()
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": error_text}},
                status_code=backend_resp.status_code,
            )
        return StreamingResponse(
            _stream_openai_to_anthropic(backend_resp, original_model, _estimate_tokens(openai_json)),
            media_type="text/event-stream",
        )

    resp = get_client().post(
        f"{BASE_URL}{target_path}",
        headers=headers,
        content=openai_json,
    )
    logger.info("[anthropic->openai] POST %s -> %s", target_path, resp.status_code)

    if resp.status_code >= 400:
        error_text = resp.text[:1000]
        logger.error("[anthropic->openai] Error: %s", error_text)
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": error_text}},
            status_code=resp.status_code,
        )
    try:
        anthropic_resp = _openai_response_to_anthropic(resp.json(), original_model)
    except Exception:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Invalid JSON from backend"},
            },
            status_code=502,
        )
    return JSONResponse(anthropic_resp)


# ---------------------------------------------------------------------------
# Bedrock routes
# ---------------------------------------------------------------------------
@app.post("/bedrock/model/{model_id:path}/invoke")
async def bedrock_invoke(model_id: str, request: Request):
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as e:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e)},
            },
            status_code=400,
        )

    original_model = body.pop("model", model_id)
    body.pop("anthropic_version", None)
    openai_json = json.dumps(_anthropic_request_to_openai(body))

    logger.info("[bedrock] invoke: model=%s, messages=%s", original_model, len(body.get("messages", [])))
    target_path = f"{INFERENCE_PREFIX}/chat/completions"
    headers = _backend_headers("POST", target_path, openai_json)

    try:
        resp = get_client().post(
            f"{BASE_URL}{target_path}",
            headers=headers,
            content=openai_json,
            timeout=120,
        )
    except httpx.TimeoutException:
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "timeout_error",
                    "message": "Request timed out after 120s",
                },
            },
            status_code=504,
        )

    logger.info("[bedrock] POST %s -> %s", target_path, resp.status_code)
    if resp.status_code >= 400:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": resp.text[:1000]},
            },
            status_code=resp.status_code,
        )
    try:
        return JSONResponse(_openai_response_to_anthropic(resp.json(), original_model))
    except Exception:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Invalid JSON from backend"},
            },
            status_code=502,
        )


@app.post("/bedrock/model/{model_id:path}/invoke-with-response-stream")
async def bedrock_invoke_stream(model_id: str, request: Request):
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as e:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e)},
            },
            status_code=400,
        )

    original_model = body.pop("model", model_id)
    body.pop("anthropic_version", None)
    body["stream"] = True
    openai_json = json.dumps(_anthropic_request_to_openai(body))

    logger.info("[bedrock] invoke-stream: model=%s, messages=%s", original_model, len(body.get("messages", [])))
    target_path = f"{INFERENCE_PREFIX}/chat/completions"
    headers = _backend_headers("POST", target_path, openai_json)

    req = get_client().build_request(
        "POST",
        f"{BASE_URL}{target_path}",
        headers=headers,
        content=openai_json,
    )
    resp = get_client().send(req, stream=True)
    logger.info("[bedrock] stream POST %s -> %s", target_path, resp.status_code)
    if resp.status_code >= 400:
        error_text = resp.text[:1000]
        resp.close()
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": error_text},
            },
            status_code=resp.status_code,
        )

    return StreamingResponse(
        _stream_openai_to_anthropic(resp, original_model, _estimate_tokens(openai_json)),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# WebSocket catch-all — reject upgrade so clients fall back to HTTP
# ---------------------------------------------------------------------------
@app.websocket("/{path:path}")
async def websocket_reject(websocket: WebSocket, path: str):
    await websocket.close(code=1001)


# ---------------------------------------------------------------------------
# Catch-all proxy
# ---------------------------------------------------------------------------
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request):
    target_path = f"/{path}" if path else "/"

    # Rewrite /v1/* -> /inference/v1/*
    if target_path.startswith("/v1/"):
        target_path = INFERENCE_PREFIX + target_path[len("/v1") :]
    elif target_path == "/chat/completions":
        target_path = f"{INFERENCE_PREFIX}/chat/completions"

    query_string = request.url.query
    signed_path = f"{target_path}?{query_string}" if query_string else target_path
    target_url = f"{BASE_URL}{signed_path}"
    body_bytes = await request.body()
    body = body_bytes.decode()

    if body:
        try:
            payload = json.loads(body)
            if "model" in payload:
                original_model = payload["model"]
                payload["model"] = MODEL_MAP.get(original_model, DEFAULT_MODEL)
                if payload["model"] != original_model:
                    logger.info("[proxy] Model rewrite: %s -> %s", original_model, payload["model"])
                if "temperature" in payload and "top_p" in payload:
                    del payload["top_p"]
                body = json.dumps(payload)
        except (json.JSONDecodeError, KeyError):
            pass

    is_stream = False
    if body:
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            is_stream = json.loads(body).get("stream", False)

    headers = _backend_headers(request.method, signed_path, body)
    if request.headers.get("content-type"):
        headers["Content-Type"] = request.headers["content-type"]
    if target_path.endswith("/chat/completions") and not body_bytes:
        if request.method == "POST":
            accept = request.headers.get("accept", "")
            if "text/event-stream" in accept:
                logger.info("[proxy] Empty SSE probe detected")
                return StreamingResponse(iter([b"event: done\ndata: [DONE]\n\n"]), media_type="text/event-stream")
        logger.warning("[proxy] Empty chat request body received")
        return JSONResponse(
            {
                "error": {
                    "message": "Empty request body received from client",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    if is_stream:
        req = get_client().build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
        resp = get_client().send(req, stream=True)
        logger.info("[proxy] %s %s -> %s", request.method, signed_path, resp.status_code)
        return StreamingResponse(
            resp.iter_bytes(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/event-stream"),
        )

    resp = get_client().request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
    )
    logger.info("[proxy] %s %s -> %s", request.method, signed_path, resp.status_code)
    if resp.status_code >= 400:
        logger.error("[proxy] Error body: %s", resp.text[:1000])

    return StreamingResponse(
        iter([resp.content]),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------------------
# Startup: discover instance/team IDs
# ---------------------------------------------------------------------------
def _discover_instance_and_team() -> None:
    global INSTANCE_ID, TEAM_ID
    if INSTANCE_ID and TEAM_ID:
        return
    try:
        path = "/admin/v1/profile"
        r = httpx.get(
            f"{BASE_URL}{path}",
            headers={
                "Authorization": f"Apikey {API_KEY}",
                "User-Agent": "bobshell/1.0.4",
                **sign_request("GET", path, ""),
            },
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning("[startup] /admin/v1/profile -> %s: %s", r.status_code, r.text[:200])
            return
        data = r.json()
        instances = data.get("instances", [])
        if not instances:
            logger.warning("[startup] No instances on profile; cannot auto-discover team.")
            return
        inst = instances[0]
        if not INSTANCE_ID:
            INSTANCE_ID = inst.get("instance_id", "")
        teams = inst.get("teams", [])
        if not TEAM_ID and teams:
            TEAM_ID = teams[0].get("id", "")
    except Exception as e:
        logger.warning("[startup] Failed to auto-discover instance/team: %s", e)


if not API_KEY:
    logger.error("No API key configured. Set BOB_API_KEY in .env or environment.")
    raise SystemExit(1)

_discover_instance_and_team()


def _print_banner():
    logger.info("=== bob-proxy ===\n")
    logger.info("  API key:  %s...%s", API_KEY[:10], API_KEY[-4:])
    logger.info("  Instance: %s", INSTANCE_ID or "(not set)")
    logger.info("  Team:     %s", TEAM_ID or "(not set)")
    logger.info("  Backend:  %s", BASE_URL)
    logger.info("  Proxy:    http://localhost:%s\n", PORT)
    logger.info("OpenCode / OpenAI-compatible:")
    logger.info("  baseURL: http://localhost:%s/v1", PORT)
    logger.info("  apiKey:  any-value\n")
    logger.info("JetBrains AI Assistant:")
    logger.info("  URL: http://localhost:%s/v1  |  Model: sonnet-4.5\n", PORT)
    logger.info("Claude Code (Anthropic mode):")
    logger.info("  ANTHROPIC_BASE_URL=http://localhost:%s/anthropic ANTHROPIC_API_KEY=proxy claude\n", PORT)
    logger.info("Claude Code (Bedrock mode — Windows, set once):")
    logger.info('  [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_USE_BEDROCK","1","User")')
    logger.info(
        '  [System.Environment]::SetEnvironmentVariable("ANTHROPIC_BEDROCK_BASE_URL","http://localhost:%s/bedrock","User")',
        PORT,
    )
    logger.info('  [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_SKIP_BEDROCK_AUTH","1","User")')
    logger.info(
        '  [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD","1","User")'
    )
    logger.info('  [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC","1","User")')
    logger.info('  [System.Environment]::SetEnvironmentVariable("AWS_ACCESS_KEY_ID","dummy","User")')
    logger.info('  [System.Environment]::SetEnvironmentVariable("AWS_SECRET_ACCESS_KEY","dummy","User")')
    logger.info('  [System.Environment]::SetEnvironmentVariable("AWS_REGION","us-east-1","User")')
    logger.info('  [System.Environment]::SetEnvironmentVariable("ANTHROPIC_DEFAULT_MODEL","premium","User")\n')


if __name__ == "__main__":
    uvicorn.run("proxy:app", host="0.0.0.0", port=PORT, reload=True, http="h11", log_level="error")
