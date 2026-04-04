# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MCP server lifecycle -- spawn mnemosyne-mcp, discover tools, route calls."""

from __future__ import annotations

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class McpBridge:
    """Manages a single MCP server subprocess over stdio."""

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._cm = None
        self._tools: dict[str, dict] = {}

    async def start(self, command: str = "mnemosyne-mcp") -> None:
        """Spawn the MCP server and discover its tools."""
        params = StdioServerParameters(command=command, args=[])
        self._cm = stdio_client(params)
        read_stream, write_stream = await self._cm.__aenter__()
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools = {t.name: t for t in result.tools}

    def get_tools_for_ollama(self) -> list[dict]:
        """Convert MCP tools to Ollama's function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return the text result."""
        result = await self._session.call_tool(name, arguments)
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
        return "\n".join(parts)

    async def stop(self) -> None:
        """Shut down the MCP session and server subprocess."""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
        except BaseException:
            pass
        try:
            if self._cm:
                await self._cm.__aexit__(None, None, None)
        except BaseException:
            pass
        self._session = None
        self._cm = None
