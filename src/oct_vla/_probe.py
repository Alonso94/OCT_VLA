"""Read-only environment metadata probe, executed by each selected interpreter.

This file must remain standard-library-only and Python 3.10 compatible. Finding
top-level module specs does not import simulator, CUDA, or learning libraries.
"""

import importlib.util
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def inspect_environment(packages: dict[str, str]) -> dict:
    results = {}
    for module, package in packages.items():
        try:
            spec = importlib.util.find_spec(module)
            origin = spec.origin if spec else None
            dist = distribution(package)
            version = dist.version
            direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
            revision = direct_url.get("vcs_info", {}).get("commit_id")
        except PackageNotFoundError:
            version, revision = None, None
        results[module] = {"origin": origin, "version": version, "revision": revision}
    stale = []
    for directory in sys.path:
        if not directory or Path(directory).name not in {"site-packages", "dist-packages"}:
            continue
        for pth in sorted(Path(directory).glob("*.pth")):
            for line in pth.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "import ", "import\t")):
                    continue
                target = pth.parent / line
                if not target.exists():
                    stale.append({"file": str(pth), "target": str(target)})
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": results,
        "stale_paths": stale,
    }


if __name__ == "__main__":
    print(json.dumps(inspect_environment(json.loads(sys.argv[1]))))
