#
# Fetch the asset pack (USDs, meshes, HDRs) from a HuggingFace dataset repo
# into ./assets/. Run once after cloning; re-run to update / pin a different
# revision.
#
# Why HF: assets are ~300 MB+ binary files, past GitHub LFS's free quota
# (1 GB storage / 1 GB monthly bandwidth) and one file (r1pro.usda, 126 MB)
# is past GitHub's hard 100 MB per-file cap. HF Datasets gives versioned
# git+LFS hosting with much more generous free-tier limits.
#
# Repo layout assumed on HF:
#   <repo-id>/
#       robot/G2/...
#       robot/r1pro/...
#       objects/...
#       lighting/...
#   (mirrors what should land under ./assets/ here)
#

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO_ID = "jihai518/bio_demo"
DEFAULT_REVISION = "main"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the bio_sim asset pack from a HuggingFace dataset "
            "repo into ./assets/."
        ),
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help=(
            "Branch / tag / commit SHA on the HF repo. Pin a tag (e.g. "
            f"'v0.1') for reproducibility. (default: {DEFAULT_REVISION})"
        ),
    )
    parser.add_argument(
        "--dest", default=None,
        help=(
            "Where to write the assets. Defaults to <repo-root>/assets/. "
            "<repo-root> is computed from this script's location."
        ),
    )
    parser.add_argument(
        "--token", default=None,
        help=(
            "HF auth token. Only needed for private repos; public repos "
            "(the default) don't need this. You can also run "
            "`huggingface-cli login` once instead of passing this each time."
        ),
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[fetch_assets] huggingface_hub not installed. Run `uv sync` "
            "first (it's declared in pyproject.toml).",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    dest = Path(args.dest) if args.dest else repo_root / "assets"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_assets] repo:     {args.repo_id}")
    print(f"[fetch_assets] revision: {args.revision}")
    print(f"[fetch_assets] dest:     {dest}")
    print("[fetch_assets] downloading (this can take a few minutes on first run)...")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(dest),
        token=args.token,
    )

    print(f"[fetch_assets] done -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
