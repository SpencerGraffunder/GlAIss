"""No unreferenced top-level names in the console package.

GlAIss accumulated dead surface twice: a whole proxy module nobody routed
through, and orphaned helpers/constants left behind when it went. Endpoint and UI
contracts are covered by the other tests; this one is the general tripwire — a
top-level function, class or constant that nothing (in-repo or the page) mentions
must be deleted or wired up, not carried along.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCANNED = sorted((REPO / "glaiss_app").glob("*.py")) + [REPO / "glaiss.py"]

# Names that are entrypoints or framework/export contracts rather than call sites.
ALLOWLIST = {"main", "app", "CONFIG", "hub", "install", "__version__"}


def _sources():
    srcs = {p: p.read_text() for p in SCANNED}
    for extra in list((REPO / "llpc").glob("*.py")) + list(REPO.glob("test_*.py")):
        srcs[extra] = extra.read_text()
    srcs[REPO / "glaiss_app/static/index.html"] = (REPO / "glaiss_app/static/index.html").read_text()
    srcs[REPO / "scripts/install.sh"] = (REPO / "scripts/install.sh").read_text()
    return srcs


def _top_level_names(text):
    tree = ast.parse(text)
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append((node.name, node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and (t.id.isupper() or t.id[0].isupper()):
                    names.append((t.id, node))
    return names


def test_console_package_has_no_unreferenced_names():
    srcs = _sources()
    dead = []
    for path, text in srcs.items():
        if path.suffix != ".py" or path not in SCANNED:
            continue
        for name, node in _top_level_names(text):
            if name in ALLOWLIST or name.startswith("__") or name == "app":
                continue
            if getattr(node, "decorator_list", []):
                continue  # framework-wired (route handlers, startup hooks)
            pattern = rf"\b{re.escape(name)}\b"
            own_uses = len(re.findall(pattern, text)) - 1  # subtract the definition
            external = sum(len(re.findall(pattern, t)) for p, t in srcs.items() if p != path)
            if own_uses <= 0 and external == 0:
                dead.append(f"{path.relative_to(REPO)}:{node.lineno} {name}")
    assert not dead, "unreferenced top-level names (delete or wire them up): " + "; ".join(dead)
