#!/usr/bin/env python3
"""Print RAPL energy counters as JSON. Intended to be invoked via sudo."""

from __future__ import annotations

import glob
import json
from pathlib import Path


def domain(path: Path) -> dict | None:
    try:
        energy = int((path / "energy_uj").read_text().strip())
    except Exception:
        return None
    try:
        name = (path / "name").read_text().strip()
    except Exception:
        name = None
    try:
        max_range = int((path / "max_energy_range_uj").read_text().strip())
    except Exception:
        max_range = 0
    return {
        "path": str(path),
        "name": name,
        "energy_uj": energy,
        "max_energy_range_uj": max_range,
    }


def main() -> None:
    out = []
    patterns = [
        "/sys/class/powercap/intel-rapl:*",
        "/sys/class/powercap/intel-rapl:*/*",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if not (p / "energy_uj").exists():
                continue
            item = domain(p)
            if item:
                out.append(item)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
