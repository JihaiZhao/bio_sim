#
# Asset library: the single place that knows WHERE assets live and HOW to
# read their metadata. Modelled on genie_sim's convention so the two stay
# interchangeable:
#
#   * one asset ROOT, resolved via the SIM_ASSETS env var with a sensible
#     in-repo default (genie: geniesim/utils/system_utils.py).
#   * every asset is a self-contained DIRECTORY under a typed subtree
#     (objects/ , robot/ , background/ ...), addressed by a path RELATIVE
#     to the root -- genie calls this the `data_info_dir`.
#   * metadata (mass, size, scale, unit, upAxis, ...) lives WITH the asset
#     in `object_parameters.json`, never inlined in a scene/task file
#     (genie: plugins/ader/action/common_actions.py:get_object_size).
#
# Scene/task recipes therefore only ever carry a relative `asset` string +
# a pose; resolution and metadata are this module's job.
#

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Sequence

# <repo>/  (this file is <repo>/bio_sim/asset_lib.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ROOT = os.path.join(_REPO_ROOT, "assets")

# genie uses this exact env var name; honour it so a shared asset checkout
# (e.g. the GenieSimAssets dataset) can back both simulators unchanged.
ENV_VAR = "SIM_ASSETS"

_PARAMS_FILE = "object_parameters.json"


def asset_root() -> str:
    """Absolute path to the asset root.

    `SIM_ASSETS` wins if set and existing; otherwise the in-repo `assets/`.
    """
    env = os.environ.get(ENV_VAR)
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    return _DEFAULT_ROOT


def asset_path(rel: str) -> str:
    """Resolve a root-relative asset path to an absolute one."""
    return os.path.join(asset_root(), rel)


@dataclass
class ObjectAsset:
    """A resolved asset directory + its `object_parameters.json`.

    `data_info_dir` is the genie-style root-relative directory, e.g.
    ``objects/bio_optica_aus240plus``.
    """

    data_info_dir: str
    params: dict = field(default_factory=dict)

    @property
    def dir(self) -> str:
        return asset_path(self.data_info_dir)

    @property
    def usd_path(self) -> str:
        # Explicit `usd_file` in the sidecar wins; else the first *.usd[a]
        # in the directory (genie objects ship a single `Aligned.usd`).
        named = self.params.get("usd_file")
        if named:
            return os.path.join(self.dir, named)
        for f in sorted(os.listdir(self.dir)):
            if f.endswith((".usd", ".usda", ".usdc")):
                return os.path.join(self.dir, f)
        raise FileNotFoundError(f"no USD in asset dir {self.dir!r}")

    @property
    def object_id(self) -> str:
        return self.params.get("object_id", os.path.basename(self.data_info_dir))

    @property
    def scale(self) -> float:
        return float(self.params.get("scale", 1.0))

    @property
    def size(self) -> List[float]:
        """Scaled AABB extents [x, y, z] (genie: scale * size)."""
        s = self.params.get("size", [0.05, 0.05, 0.05])
        return [self.scale * float(v) for v in s]

    @property
    def mass(self) -> float:
        return float(self.params.get("mass", 0.05))

    @property
    def is_fixed(self) -> bool:
        """True => a static scene fixture, not a graspable rigid body."""
        return bool(self.params.get("fixed", False))

    @property
    def up_axis(self) -> Sequence[str]:
        return self.params.get("upAxis", ["z"])


def load_object(data_info_dir: str) -> ObjectAsset:
    """Load an asset by its root-relative directory (genie data_info_dir)."""
    data_info_dir = data_info_dir.strip("/")
    ppath = os.path.join(asset_path(data_info_dir), _PARAMS_FILE)
    params: dict = {}
    if os.path.isfile(ppath):
        with open(ppath, "r") as f:
            params = json.load(f)
    return ObjectAsset(data_info_dir=data_info_dir, params=params)


__all__ = [
    "ENV_VAR",
    "ObjectAsset",
    "asset_path",
    "asset_root",
    "load_object",
]
