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
    # Code reference (cuRobo wrapper, controller patterns) + source of Agibot
    # G1 URDF and pre-built cuRobo config. Code MPL-2.0, assets CC BY-NC-SA 4.0
    # (research/personal use only).
    "genie_sim": "https://github.com/AgibotTech/genie_sim",
    # M2: enable when we start converting consumable meshes.
    # "autobio_assets": "https://github.com/autobio-bench/AutoBio",
}

# IsaacLab and cuRobo install editable via `uv pip install -e` after cloning,
# so we pin to specific refs here. M0 install order: isaacsim (pip) -> IsaacLab
# (editable from this clone) -> curobo (editable from this clone, source build).
PINNED_REPOS: dict[str, tuple[str, str]] = {
    "IsaacLab": ("https://github.com/isaac-sim/IsaacLab.git", "v2.3.0"),
    "curobo": ("https://github.com/NVlabs/curobo.git", "main"),
}


def clone(name: str, url: str, ref: str | None = None, shallow: bool = True) -> None:
    dest = THIRD_PARTY / name
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    THIRD_PARTY.mkdir(exist_ok=True)
    args = ["git", "clone"]
    if shallow:
        args += ["--depth", "1"]
    if ref:
        args += ["--branch", ref]
    args += [url, str(dest)]
    print(f"[clone] {url}@{ref or 'HEAD'} -> {dest} (shallow={shallow})")
    subprocess.run(args, check=True)


def main() -> None:
    for name, url in REPOS.items():
        clone(name, url)
    for name, (url, ref) in PINNED_REPOS.items():
        # curobo's setuptools_scm reads the version from `git describe`, so it
        # needs the full history with tags. The other repos can stay shallow.
        clone(name, url, ref, shallow=(name != "curobo"))
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
