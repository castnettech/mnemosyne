# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ollama tool-calling agent loop and public run() API."""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mnemosyne_ollama.bridge import McpBridge

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

TOOL_CAPABLE = [
    "llama3.1", "llama3.2", "llama3.3", "llama4",
    "qwen2.5", "qwen3",
    "gemma3", "gemma4",
    "phi4",
    "mistral-nemo", "command-r",
]

SYSTEM_PROMPT = """\
You are a code search assistant powered by Mnemosyne. Answer questions about \
codebases by searching the indexed code.

Rules:
- ALWAYS use the search tool before answering. Never guess.
- If search returns "No files indexed yet", call index first, then search again.
- Cite file paths and line numbers from results.
- Be concise. Quote relevant code, don't paraphrase.
- Project root: {project_root}
- Default search budget: {budget}"""


@dataclass
class AgentResult:
    """Result from a single run() invocation."""
    response: str
    error: str | None = None


def _resolve_url(ollama_url: str | None) -> str:
    if ollama_url:
        return ollama_url.rstrip("/")
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON to a URL, return parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get(url: str, timeout: int = 10) -> dict:
    """GET JSON from a URL, return parsed response."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _pick_model(base_url: str) -> str:
    """Auto-detect a tool-capable model from Ollama."""
    try:
        result = _get(f"{base_url}/api/tags")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url} -- is it running?"
        ) from exc
    models = result.get("models", [])
    names = [m.get("name", "") for m in models]
    for prefix in TOOL_CAPABLE:
        for name in names:
            if name.startswith(prefix):
                return name
    installed = ", ".join(names) if names else "(none)"
    raise RuntimeError(
        f"No tool-capable model found. Installed: {installed}. "
        f"Try: ollama pull qwen2.5"
    )


def _ollama_chat(
    base_url: str, model: str, messages: list[dict], tools: list[dict]
) -> dict:
    """Send a chat request to Ollama with tool definitions."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools if tools else None,
        "stream": False,
        "options": {"num_ctx": 32768},
    }
    try:
        return _post(f"{base_url}/api/chat", payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}: {body}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url} -- is it running?"
        ) from exc


async def run(
    query: str,
    *,
    model: str | None = None,
    project_root: str | None = None,
    budget: int = 8000,
    ollama_url: str | None = None,
    verbose: bool = False,
) -> AgentResult:
    """Run a single query through Ollama with mnemosyne-mcp tools.

    Args:
        query: Natural language question about a codebase.
        model: Ollama model name. Auto-detected if omitted.
        project_root: Absolute path to the project. Defaults to cwd.
        budget: Token budget for mnemosyne search results.
        ollama_url: Ollama base URL. Defaults to OLLAMA_HOST env or localhost:11434.
        verbose: Print tool calls to stderr.

    Returns:
        AgentResult with the model's response.
    """
    import sys

    base_url = _resolve_url(ollama_url)
    root = str(Path(project_root).resolve()) if project_root else str(Path.cwd())

    if not model:
        model = _pick_model(base_url)
        if verbose:
            print(f"[mnemosyne-ollama] auto-detected model: {model}", file=sys.stderr)

    bridge = McpBridge()
    try:
        await bridge.start()
    except FileNotFoundError:
        return AgentResult(
            response="",
            error="mnemosyne-mcp not found on PATH. Install: pip install mnemosyne-mcp",
        )

    try:
        tools = bridge.get_tools_for_ollama()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                project_root=root, budget=budget
            )},
            {"role": "user", "content": query},
        ]

        loop = asyncio.get_running_loop()
        max_steps = 10

        for step in range(max_steps):
            resp = await loop.run_in_executor(
                None, _ollama_chat, base_url, model, messages, tools
            )
            msg = resp.get("message", {})
            raw_content = msg.get("content", "")
            content = _THINK_RE.sub("", raw_content).strip()
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                return AgentResult(response=content)

            # Append assistant message with tool calls
            messages.append(msg)

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if verbose:
                    print(
                        f"[mnemosyne-ollama] tool: {name}({json.dumps(arguments, indent=None)})",
                        file=sys.stderr,
                    )

                result_text = await bridge.call_tool(name, arguments)
                messages.append({
                    "role": "tool",
                    "content": result_text,
                })

        return AgentResult(
            response="", error="Reached max tool-call iterations without a final answer."
        )
    finally:
        await bridge.stop()
