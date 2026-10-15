"""Public printer connector and MCP adapter entry points.

Resolve exports on demand so importing this package does not pre-import modules
that also serve as command-line entry points (``python -m``).
"""

from importlib import import_module

__all__ = [
    "PrinterError",
    "list_printers",
    "printer_status",
    "submit_print",
    "job_status",
    "load_printer_tools",
    "mcp_client",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "tools" if name in ("load_printer_tools", "mcp_client") else "connector"
    value = getattr(import_module(f"app.printing.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
