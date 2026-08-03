"""Thin Python adapter for the Rust-owned LMX engine."""

from __future__ import annotations

import json
from base64 import b64decode
from typing import Any

from ._native import (
    ResponseSession,
    build_message_item_json,
    build_wire_request_json,
    generate_image_json,
    generate_video_json,
    load_request_context_json,
    output_text_from_items_json,
    provider_registry_json,
    normalize_tool_output_json,
    tool_failure_output_json,
    version,
)

__all__ = [
    "build_wire_request",
    "build_message_item",
    "generate_image",
    "generate_video",
    "load_request_context",
    "output_text_from_items",
    "provider_registry",
    "respond",
    "structured_response",
    "stream_response",
    "version",
]


def provider_registry() -> list[dict[str, Any]]:
    """Return provider defaults from the shared Rust core."""
    return json.loads(provider_registry_json())


def load_request_context(provider: str) -> dict[str, Any]:
    """Load environment or Codex OAuth credentials in the Rust core."""
    return json.loads(load_request_context_json(json.dumps(provider)))


def build_message_item(*, role: str, text: str) -> dict[str, Any]:
    return json.loads(build_message_item_json(role, text))


def output_text_from_items(items: list[dict[str, Any]]) -> str:
    return output_text_from_items_json(json.dumps(items))


def build_wire_request(request: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible request without Python-side provider logic."""
    return json.loads(build_wire_request_json(json.dumps(request)))


def generate_image(request: dict[str, Any]) -> dict[str, Any]:
    """Generate an image through the Rust engine and return bytes as ``content``."""
    payload = dict(request)
    if "context" not in payload and "provider" in payload:
        payload["context"] = load_request_context(str(payload.pop("provider")))
    result = json.loads(generate_image_json(json.dumps(payload)))
    result["content"] = b64decode(result.pop("contentBase64"))
    return result


def generate_video(request: dict[str, Any]) -> dict[str, Any]:
    """Generate a video through the Rust engine and return bytes as ``content``."""
    payload = dict(request)
    if "context" not in payload and "provider" in payload:
        payload["context"] = load_request_context(str(payload.pop("provider")))
    result = json.loads(generate_video_json(json.dumps(payload)))
    result["content"] = b64decode(result.pop("contentBase64"))
    return result


def _tool_output(call: dict[str, Any], handler: Any) -> tuple[dict[str, Any], Any]:
    try:
        result = handler(call.get("arguments", {}))
        output = json.loads(normalize_tool_output_json(call["callId"], json.dumps(result)))
        return output, output["result"]
    except Exception as error:  # Tool failures are model-visible outputs, not engine failures.
        output = json.loads(tool_failure_output_json(call["callId"], str(error)))
        return output, output["result"]


def stream_response(request: dict[str, Any]):
    """Yield normalized events while Rust owns transport and response state."""
    handlers = request.get("toolHandlers", request.get("tool_handlers", {}))
    payload = {key: value for key, value in request.items() if key not in {"toolHandlers", "tool_handlers", "observer"}}
    if "context" not in payload and "provider" in payload:
        payload["context"] = load_request_context(str(payload.pop("provider")))
    session = ResponseSession(json.dumps(payload))
    try:
        while True:
            session.start_round()
            while True:
                frame = json.loads(session.next_frame_json())
                if frame["type"] == "event":
                    yield frame["event"]
                    continue
                if frame["type"] == "failed":
                    raise RuntimeError(frame["error"])
                action = frame["next"]
                break
            if action["type"] == "completed":
                result = action["result"]
                yield {"type": "completed", **result}
                return result
            outputs = []
            for call in action["calls"]:
                yield {"type": "tool_call_started", **call}
                handler = handlers.get(call["name"])
                if handler is None:
                    output = json.loads(tool_failure_output_json(call["callId"], f"unknown function tool: {call['name']}"))
                    observed = output["result"]
                else:
                    output, observed = _tool_output(call, handler)
                outputs.append(output)
                yield {"type": "tool_call_completed", "name": call["name"], "callId": call["callId"], "result": observed}
            session.submit_tool_outputs_json(json.dumps(outputs))
    finally:
        session.cancel()


def respond(request: dict[str, Any]) -> dict[str, Any]:
    """Run a complete response, including host-language tool callbacks."""
    completed: dict[str, Any] | None = None
    for event in stream_response(request):
        if event["type"] == "completed":
            completed = event
    if completed is None:
        raise RuntimeError("response did not complete")
    return {key: value for key, value in completed.items() if key != "type"}


def structured_response(request: dict[str, Any], text_format: type[Any]) -> Any:
    """Request native JSON-schema output, then validate it with the caller's Pydantic model."""
    payload = dict(request)
    payload["textFormat"] = {
        "type": "json_schema",
        "name": getattr(text_format, "__name__", "structured_response"),
        "schema": text_format.model_json_schema(),
        "strict": True,
    }
    result = respond(payload)
    candidates = [result.get("outputText", "")]
    for item in result.get("outputItems", []):
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            candidates.append(content)
        elif isinstance(content, list):
            candidates.extend(part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    for candidate in reversed(candidates):
        try:
            return text_format.model_validate_json(candidate)
        except Exception:
            continue
    raise RuntimeError("structured response did not contain parsed output")
