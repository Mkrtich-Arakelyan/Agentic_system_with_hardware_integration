"""Give the printer MCP server's tools to the executor as LangChain tools.

load_printer_tools(env) starts printer_mcp_server.py over stdio, loads its tools
with langchain-mcp-adapters and wraps each one so a plain synchronous .invoke(args)
works and returns the server's JSON result as one string. Each call starts the
server, runs the tool and stops it, which is fine for a printer.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER = Path(__file__).resolve().parent / "printer_mcp_server.py"


def mcp_client(env=None):
    """A client for one stdio server; os.environ keeps the venv, PATH and FAKE_PRINTER_PROBLEM."""
    return MultiServerMCPClient({
        "printer": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVER)],
            "env": dict(os.environ) | (env or {}),
        }
    })


def _run_sync(coro):
    """Run a coroutine to the end, also when an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _text(content):
    """Turn MCP content into one string: a str stays, content blocks are joined."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        for block in content
    )


def _wrap(mcp_tool: BaseTool) -> BaseTool:
    """Same name, description and args schema, but callable with a plain .invoke()."""

    async def call(**kwargs):
        content, _artifact = await mcp_tool.coroutine(**kwargs)
        return _text(content)

    def call_sync(**kwargs):
        return _run_sync(call(**kwargs))

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
        func=call_sync,
        coroutine=call,
    )


def load_printer_tools(env=None) -> list[BaseTool]:
    """Start the printer MCP server with env added and return its tools for the executor."""
    return [_wrap(mcp_tool) for mcp_tool in _run_sync(mcp_client(env).get_tools())]


if __name__ == "__main__":
    for printer_tool in load_printer_tools():
        print(f"{printer_tool.name}: {printer_tool.description.splitlines()[0]}")
