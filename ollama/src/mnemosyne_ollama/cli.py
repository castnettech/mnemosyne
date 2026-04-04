# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI entry point for mnemosyne-ollama."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mnemosyne-ollama",
        description="Ask your codebase questions using Ollama and Mnemosyne.",
    )
    parser.add_argument("query", nargs="?", default=None, help="Question (omit for interactive mode)")
    parser.add_argument("-m", "--model", default=None, help="Ollama model (auto-detected if omitted)")
    parser.add_argument("-b", "--budget", type=int, default=8000, help="Token budget for search (default: 8000)")
    parser.add_argument("-r", "--project-root", default=None, help="Project root (default: cwd)")
    parser.add_argument("--ollama-url", default=None, help="Ollama URL (default: OLLAMA_HOST env or localhost:11434)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print tool calls to stderr")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    args = parser.parse_args()

    if args.query:
        _run_single(args)
    else:
        _run_interactive(args)


def _run_single(args: argparse.Namespace) -> None:
    from mnemosyne_ollama.agent import run

    try:
        result = asyncio.run(run(
            args.query,
            model=args.model,
            project_root=args.project_root,
            budget=args.budget,
            ollama_url=args.ollama_url,
            verbose=args.verbose,
        ))
    except ConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)
    print(result.response)


def _run_interactive(args: argparse.Namespace) -> None:
    from mnemosyne_ollama.bridge import McpBridge
    from mnemosyne_ollama.agent import (
        SYSTEM_PROMPT, _ollama_chat, _pick_model,
        _resolve_url, _THINK_RE,
    )
    import json
    from pathlib import Path

    base_url = _resolve_url(args.ollama_url)
    root = str(Path(args.project_root).resolve()) if args.project_root else str(Path.cwd())

    try:
        model = args.model or _pick_model(base_url)
    except (ConnectionError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"mnemosyne-ollama interactive ({model})")
    print(f"Project: {root}")
    print("Type your question. Ctrl+C to exit.\n")

    async def _session():
        bridge = McpBridge()
        try:
            await bridge.start()
        except FileNotFoundError:
            print("Error: mnemosyne-mcp not found. Install: pip install mnemosyne-mcp", file=sys.stderr)
            return

        try:
            tools = bridge.get_tools_for_ollama()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    project_root=root, budget=args.budget
                )},
            ]
            loop = asyncio.get_running_loop()

            while True:
                try:
                    query = input("> ")
                except EOFError:
                    break
                if not query.strip():
                    continue

                messages.append({"role": "user", "content": query})

                for _ in range(10):
                    resp = await loop.run_in_executor(
                        None, _ollama_chat, base_url, model, messages, tools
                    )
                    msg = resp.get("message", {})
                    raw_content = msg.get("content", "")
                    content = _THINK_RE.sub("", raw_content).strip()
                    tool_calls = msg.get("tool_calls") or []

                    if not tool_calls:
                        if content:
                            print(f"\n{content}\n")
                        messages.append({"role": "assistant", "content": content})
                        break

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
                        if args.verbose:
                            print(f"  [tool] {name}({json.dumps(arguments, indent=None)})", file=sys.stderr)
                        result_text = await bridge.call_tool(name, arguments)
                        messages.append({"role": "tool", "content": result_text})
        finally:
            await bridge.stop()

    try:
        asyncio.run(_session())
    except KeyboardInterrupt:
        print()
