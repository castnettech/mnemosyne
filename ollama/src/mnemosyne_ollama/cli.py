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
    from mnemosyne_ollama import __version__
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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


def _read_multiline() -> str:
    """
    Read one logical message from stdin.

    Single-line mode (default):
        Type your question and press Enter -- submits immediately.

    Multi-line mode:
        Type \"\"\" and press Enter to open a multi-line block.
        Type or paste freely (blank lines preserved).
        Type \"\"\" and press Enter to close and submit.

    Bracketed paste (ESC[?2004h) is enabled on startup so terminals that
    support it deliver pasted text as one atomic block through input().

    Raises EOFError when stdin closes (Ctrl+D).
    """
    try:
        line = input("> ")
    except EOFError:
        raise

    # Strip bracketed-paste escape wrappers that some terminals leave in the
    # string when bracketed paste mode is active but readline strips them late.
    line = line.replace("\x1b[200~", "").replace("\x1b[201~", "")

    if line.strip() == '"""':
        return _read_multiline_block()

    return line


def _read_multiline_block() -> str:
    """
    Read lines until a closing \"\"\" sentinel, then return the joined block.
    Called after the opening \"\"\" has already been consumed.
    """
    lines: list[str] = []
    print('  (multi-line -- type """ to send)')
    while True:
        try:
            line = input("... ")
        except EOFError:
            break
        line = line.replace("\x1b[200~", "").replace("\x1b[201~", "")
        if line.strip() == '"""':
            break
        lines.append(line)
    return "\n".join(lines)


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

    # Enable readline history/editing where available (stdlib, no-op on Windows
    # without pyreadline3).
    try:
        import readline as _rl  # noqa: F401
    except ImportError:
        pass

    # Enable bracketed paste mode -- terminals that support it will deliver
    # pasted content as one atomic block, preventing per-line submissions.
    # Disabled on close via the deferred reset sequence.
    sys.stdout.write("\x1b[?2004h")
    sys.stdout.flush()

    print(f"mnemosyne-ollama interactive ({model})")
    print(f"Project: {root}")
    print("Type your question and press Enter. Use triple-quote for multi-line input.")
    print("Ctrl+C or Ctrl+D to exit.\n")

    async def _session():
        # Optional runtime integration point.
        _capture = None
        _conv_id = None
        try:
            from mnemosyne_capture.capture import Capture
            _capture = Capture(Path(root))
            await _capture.start()
            _conv_id = _capture.new_conversation(source="ollama_interactive", model_id=model)
        except ImportError:
            pass
        except Exception:
            _capture = None

        bridge = McpBridge()
        try:
            await bridge.start()
        except FileNotFoundError:
            print("Error: mnemosyne-mcp not found. Install: pip install mnemosyne-mcp", file=sys.stderr)
            # Session state hook.
            if _capture is not None:
                try:
                    await _capture.stop()
                except Exception:
                    pass
            return

        try:
            tools = bridge.get_tools_for_ollama()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    project_root=root, budget=args.budget, model=model
                )},
            ]
            loop = asyncio.get_running_loop()

            while True:
                try:
                    query = _read_multiline()
                except EOFError:
                    break
                if not query.strip():
                    continue

                # Turn record hook.
                if _capture is not None and _conv_id:
                    try:
                        _capture.record(
                            _conv_id, author="user", content=query,
                            capture_source="ollama_interactive",
                        )
                    except Exception:
                        pass

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
                        # Turn record hook.
                        if _capture is not None and _conv_id and content:
                            try:
                                _capture.record(
                                    _conv_id, author="assistant", content=content,
                                    capture_source="ollama_interactive",
                                )
                            except Exception:
                                pass
                        messages.append({"role": "assistant", "content": content})
                        break

                    # Turn record hook.
                    if _capture is not None and _conv_id and content:
                        try:
                            _capture.record(
                                _conv_id, author="assistant", content=content,
                                capture_source="ollama_interactive",
                            )
                        except Exception:
                            pass

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
                        # Turn record hook.
                        if _capture is not None and _conv_id:
                            try:
                                _capture.record(
                                    _conv_id, author="tool", content=result_text,
                                    capture_source="ollama_interactive",
                                )
                            except Exception:
                                pass
                        messages.append({"role": "tool", "content": result_text})
        finally:
            await bridge.stop()
            # Optional session finalization hook.
            if _capture is not None and _conv_id:
                try:
                    _capture.store.set_conversation_state(_conv_id, "closed")
                    from mnemosyne_capture.summarizer_l3 import summarize_conversation
                    print("[mnemosyne-ollama] finalizing session...", file=sys.stderr)
                    summarize_conversation(_capture.store, _conv_id, model=model)
                except Exception:
                    pass
            # Session state hook.
            if _capture is not None:
                try:
                    await _capture.stop()
                except Exception:
                    pass

    try:
        asyncio.run(_session())
    except KeyboardInterrupt:
        print()
    finally:
        # Disable bracketed paste mode on exit.
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()