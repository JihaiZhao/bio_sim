"""Generate src/bio_sim/robot/dexmate_planning.urdf from the upstream Vega URDF.

The sim USD (Vega 0.8.3) uses real wheeled-base joints; cuRobo expects a
holonomic kinematic chain. This script:

  1. Reads third_party/dexmate_urdf/robots/humanoid/vega_1/vega_1_gripper.urdf
  2. Removes wheel joints/links ([BLR]_wheel_j[12], [BLR]_wheel_l[12]).
  3. Prepends a planar base chain: world -> base_x -> base_y -> base
     via [x_prismatic, y_prismatic, yaw_revolute] dummy joints.
  4. Writes src/bio_sim/robot/dexmate_planning.urdf.

Rerun whenever the upstream URDF changes.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_URDF = REPO_ROOT / "third_party/dexmate_urdf/robots/humanoid/vega_1/vega_1_gripper.urdf"
DST_URDF = REPO_ROOT / "src/bio_sim/robot/dexmate_planning.urdf"

WHEEL_LINK_RE = re.compile(r"^[BLR]_wheel_l[12]$")
WHEEL_JOINT_RE = re.compile(r"^[BLR]_wheel_j[12]$")

# Planar-base limits chosen for a single-room bio-lab workspace.
BASE_X_LIMIT = 20.0     # meters
BASE_Y_LIMIT = 20.0     # meters
BASE_X_VEL = 1.0        # m/s
BASE_Y_VEL = 1.0        # m/s
BASE_YAW_VEL = 1.5      # rad/s


def _make_link(name: str) -> ET.Element:
    return ET.fromstring(f'<link name="{name}"/>')


def _make_planar_joints() -> list[ET.Element]:
    return [
        ET.fromstring(
            f'''<joint name="base_x_slide" type="prismatic">
  <parent link="world"/>
  <child link="base_x_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="{-BASE_X_LIMIT}" upper="{BASE_X_LIMIT}" effort="1000" velocity="{BASE_X_VEL}"/>
</joint>'''
        ),
        ET.fromstring(
            f'''<joint name="base_y_slide" type="prismatic">
  <parent link="base_x_link"/>
  <child link="base_y_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="{-BASE_Y_LIMIT}" upper="{BASE_Y_LIMIT}" effort="1000" velocity="{BASE_Y_VEL}"/>
</joint>'''
        ),
        ET.fromstring(
            f'''<joint name="base_yaw_rotate" type="revolute">
  <parent link="base_y_link"/>
  <child link="base"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14159265" upper="3.14159265" effort="1000" velocity="{BASE_YAW_VEL}"/>
</joint>'''
        ),
    ]


def main() -> int:
    if not SRC_URDF.exists():
        print(f"missing upstream URDF: {SRC_URDF}", file=sys.stderr)
        print("run `python scripts/download_assets.py` first.", file=sys.stderr)
        return 1

    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    root.set("name", "vega_1_planning")

    removed_links = []
    removed_joints = []
    for child in list(root):
        if child.tag == "link" and WHEEL_LINK_RE.match(child.get("name", "")):
            root.remove(child)
            removed_links.append(child.get("name"))
        elif child.tag == "joint" and WHEEL_JOINT_RE.match(child.get("name", "")):
            root.remove(child)
            removed_joints.append(child.get("name"))

    world_link = _make_link("world")
    base_x_link = _make_link("base_x_link")
    base_y_link = _make_link("base_y_link")
    planar_joints = _make_planar_joints()

    insert_at = 0
    root.insert(insert_at, world_link)
    root.insert(insert_at + 1, base_x_link)
    root.insert(insert_at + 2, base_y_link)
    for i, joint in enumerate(planar_joints, start=3):
        root.insert(insert_at + i, joint)

    DST_URDF.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(DST_URDF, encoding="utf-8", xml_declaration=True)

    print(f"wrote {DST_URDF.relative_to(REPO_ROOT)}")
    print(f"  removed {len(removed_links)} wheel links: {removed_links}")
    print(f"  removed {len(removed_joints)} wheel joints: {removed_joints}")
    print(f"  prepended planar base chain: world -> base_x_link -> base_y_link -> base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
