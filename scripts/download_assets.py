"""Fetch third-party asset repos into third_party/.

Run once on a fresh checkout. Re-run is idempotent — skips already-cloned repos.

Adds:
  third_party/dexmate_urdf/   — Dexmate Vega-1 URDF + meshes + collision spheres
  third_party/autobio_assets/ — Bio-lab consumable meshes (M2)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"

REPOS: dict[str, str] = {
    "dexmate_urdf": "https://github.com/dexmate-ai/dexmate-urdf",
    # M2: enable when we start converting consumable meshes.
    # "autobio_assets": "https://github.com/autobio-bench/AutoBio",
}


def clone(name: str, url: str) -> None:
    dest = THIRD_PARTY / name
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    THIRD_PARTY.mkdir(exist_ok=True)
    print(f"[clone] {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def main() -> None:
    for name, url in REPOS.items():
        clone(name, url)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
