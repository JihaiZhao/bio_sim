#!/usr/bin/env python
#
# grasp_probe.py -- SOLVE a reachable R1 Pro world grasp quaternion.
#
# WHY this exists: grasp_quat_r1 in task_pick_place.yaml was a guessed
# WORLD orientation. Probe run 1 proved why it is wrong -- the current
# [0.7071,0.7071,0,0] IS reachable, but its finger-close axis points
# along WORLD Z (vertical): the gripper tries to pinch a table-top cube
# top-to-bottom, impossible. Run 1 also empirically established (exact on
# every in-reach candidate):
#
#   * R1 right_eef_link LOCAL-Y axis == the finger-close direction
#     (finger_link1 - finger_link2).
#
# So the fix = keep the reachable approach but roll the wrist so finger-
# close becomes HORIZONTAL and lined up with the cube's thin 3 cm axis.
#
# HOW: the runtime grasp (MoveArmTo -> ArmPlanner.plan_single, idle arm
# pinned) targets a WORLD pose; cuRobo converts it to the base frame
# using the base pose at marker A, which is the FIXED known transform
# (0, 0, z0, face_yaw). cuRobo / MotionGen runs on the GPU independently
# of the sim loop, so this probe does NOT step physics or drive the base
# at all (that nav + the env's ~40 s "Simulation App Shutting Down"
# instability is exactly what kept eating the budget before the sweep
# ran). It boots Isaac only because R1ProRobot.load_into needs a live
# stage to build the cuRobo kinematics, then immediately sweeps candidate
# world orientations through plan_single with the SAME fixed-A world<->
# base transform the runtime uses, FK's the planned final config, and
# ranks them. Output is produced within ~25 s -- before the crash window.
#
#   .venv/bin/python grasp_probe.py --headless_mode native
#   .venv/bin/python grasp_probe.py --headless_mode native --check w x y z
#
# Ordering mirrors play.py: SimApp() boots SimulationApp before any
# curobo / isaacsim.core import.
#

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# base_link world z at rest on the R1 wheels (probe run 1 measured the
# settled base at z=+0.005; mm error is negligible vs the 4 cm IK tol).
BASE_Z0 = 0.005


# ---- tiny quaternion helpers (w, x, y, z) ----------------------------- #
def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def q_axis_angle(axis, ang):
    import numpy as np

    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    s = math.sin(ang / 2.0)
    return (math.cos(ang / 2.0), ax[0] * s, ax[1] * s, ax[2] * s)


def q_norm(q):
    n = math.sqrt(sum(c * c for c in q)) + 1e-12
    return tuple(c / n for c in q)


def q_to_basis(q):
    """Columns = the quat's local X/Y/Z axes in the parent frame."""
    import numpy as np

    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ---- fixed-A world<->base transform (EXACT copy of HolonomicNav math,
#      with the base pinned at marker A = (0,0,BASE_Z0,face_yaw)) ------- #
def world_to_base(p_world, q_world, yaw):
    import numpy as np

    dx = float(p_world[0]) - 0.0
    dy = float(p_world[1]) - 0.0
    dz = float(p_world[2]) - BASE_Z0
    c, s = math.cos(-yaw), math.sin(-yaw)
    p_b = np.array([c * dx - s * dy, s * dx + c * dy, dz], dtype=np.float64)
    h = yaw / 2.0
    bw, bzq = math.cos(h), math.sin(h)
    qw, qx, qy, qz = (float(q_world[0]), float(q_world[1]),
                      float(q_world[2]), float(q_world[3]))
    rw = bw * qw - (-bzq) * qz
    rx = bw * qx - (-bzq) * qy
    ry = bw * qy + (-bzq) * qx
    rz = bw * qz + (-bzq) * qw
    q_b = np.array([rw, rx, ry, rz], dtype=np.float64)
    q_b /= np.linalg.norm(q_b) + 1e-12
    return p_b, q_b


def base_to_world(p_base, q_base, yaw):
    import numpy as np

    c, s = math.cos(yaw), math.sin(yaw)
    px, py, pz = float(p_base[0]), float(p_base[1]), float(p_base[2])
    p_w = np.array([c * px - s * py + 0.0,
                    s * px + c * py + 0.0,
                    pz + BASE_Z0], dtype=np.float64)
    h = yaw / 2.0
    bw, bzq = math.cos(h), math.sin(h)
    qw, qx, qy, qz = (float(q_base[0]), float(q_base[1]),
                      float(q_base[2]), float(q_base[3]))
    rw = bw * qw - bzq * qz
    rx = bw * qx - bzq * qy
    ry = bw * qy + bzq * qx
    rz = bw * qz + bzq * qw
    q_w = np.array([rw, rx, ry, rz], dtype=np.float64)
    q_w /= np.linalg.norm(q_w) + 1e-12
    return p_w, q_w


def candidate_world_quats():
    """Run 1: anchor q~[0.7071,0.7071,0,0] reaches the cube (|EE-cube|~0)
    with EE-localZ ~ world -Y (approach toward the cube) and EE-localY ~
    world +Z (finger-close, VERTICAL -> the bug). Roll the wrist about
    the EE's OWN approach axis (local Z) to swing finger-close to
    horizontal; sweep that fully (cheap -- pure cuRobo)."""
    anchor = q_norm((0.7071, 0.7071, 0.0, 0.0))
    out = []
    for roll in range(-180, 180, 10):
        qr = q_axis_angle((0, 0, 1), math.radians(roll))     # local Z
        base = q_norm(q_mul(anchor, qr))
        out.append((f"roll{roll:+d}", base))
        for tilt in (-20, 20):                               # small tilts
            out.append((f"roll{roll:+d}_tx{tilt:+d}",
                        q_norm(q_mul(base, q_axis_angle((1, 0, 0),
                                                        math.radians(tilt))))))
    for bn, b in (("Rx+90", q_axis_angle((1, 0, 0), math.pi / 2)),
                  ("Rx-90", q_axis_angle((1, 0, 0), -math.pi / 2))):
        for yd in (-90, 0, 90, 180):
            out.append((f"{bn}@z{yd:+d}", q_norm(
                q_mul(q_axis_angle((0, 0, 1), math.radians(yd)), b))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless_mode", type=str, default=None)
    ap.add_argument("--check", type=float, nargs=4, default=None,
                    metavar=("W", "X", "Y", "Z"))
    args, _ = ap.parse_known_args()

    from bio_sim.sim import SimApp
    sim = SimApp(headless=args.headless_mode)

    import numpy as np

    from curobo.types.math import Pose
    from curobo.types.state import JointState

    from bio_sim.scene import BioScene
    from bio_sim.specs import ROBOTS, load_ref
    from bio_sim.tasks import load_full_cfg

    # Probe is R1-only -> resolve through the registry so the yml and
    # cfg overlay both come from one source of truth.
    spec = ROBOTS["r1pro"]
    RobotCls = load_ref(spec.cls_ref)
    cfg = load_full_cfg(spec.cfg_overlay)

    # fast no-graph cuRobo warmup (probe only uses enable_graph=False
    # plan_single -> the default graph warmup is ~30 s of wasted work
    # that pushes output past the env crash window).
    from bio_sim.robot.arm import ArmPlanner

    def _fast_warmup(self):
        print("[probe] cuRobo FAST warmup (no graph)...")
        self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)
        print("[arm] cuRobo ready")

    ArmPlanner.warmup = _fast_warmup

    scene = BioScene.from_cfg(cfg)
    scene.build(sim)
    robot = RobotCls(robot_yml=spec.default_curobo_yml)
    robot.apply_init_pose(cfg)
    robot.load_into(sim, scene)            # builds cuRobo kinematics
    scene.place_for_validation(robot, cfg)  # sets the cube world pose
    robot.arm.compute_idle_retract_pin(robot.retract_config, robot.j_names)

    face_yaw = float(np.radians(cfg.get("robot_face_yaw_deg", -90.0)))
    obj_name = cfg.get("object", "object_a")
    # cube world pose straight from the declarative layout (no sim step).
    obj = scene.objects[0]
    cube_w = np.asarray(obj.position, dtype=np.float64)
    print(f"[probe] cube world={cube_w}  base@A=(0,0,{BASE_Z0}) "
          f"yaw={math.degrees(face_yaw):.0f}deg")

    cands = ([("CHECK", q_norm(tuple(args.check)))] if args.check
             else candidate_world_quats())

    def _plan(qw):
        p_b, q_b = world_to_base(cube_w, np.asarray(qw), face_yaw)
        goal = Pose(position=robot.arm.tensor_args.to_device(p_b),
                    quaternion=robot.arm.tensor_args.to_device(q_b))
        res = robot.arm.motion_gen.plan_single(
            robot.arm._retract_js.clone(), goal, robot.arm.plan_config,
            link_poses={robot.arm.idle_link: robot.arm._idle_pin})
        if not bool(res.success.item()):
            return None
        return robot.arm.motion_gen.get_full_js(res.get_interpolated_plan())

    def _fk_world(plan):
        names = list(plan.joint_names)
        last = plan.position[-1].cpu().numpy()
        js = JointState.from_position(
            robot.arm.tensor_args.to_device(
                np.asarray(last, dtype=np.float64)).view(1, -1),
            joint_names=names,
        ).get_ordered_joint_state(robot.arm.joint_names)
        pose = robot.arm.fk_link_pose(js, robot.ee_link)
        p_b = pose.position.cpu().numpy().ravel()
        q_b = pose.quaternion.cpu().numpy().ravel()
        p_w, q_w = base_to_world(p_b, q_b, face_yaw)
        q7 = [float(last[names.index(f"right_arm_joint{i}")])
              for i in range(1, 8)]
        return np.asarray(p_w), np.asarray(q_w), q7

    print(f"[probe] sweeping {len(cands)} orientations (pure cuRobo, "
          f"no sim step)...")
    results = []
    for i, (nm, qw) in enumerate(cands):
        plan = _plan(qw)
        if plan is None:
            continue
        p_w, q_w, q7 = _fk_world(plan)
        B = q_to_basis(tuple(q_w))
        fclose = B[:, 1]                    # local-Y == finger-close
        approach = B[:, 2]                  # local-Z == approach
        d = float(np.linalg.norm(p_w - cube_w))
        horiz = abs(float(fclose[2]))
        score = d + 2.0 * horiz + (0.5 if approach[2] > 0.2 else 0.0)
        results.append(dict(nm=nm, q=tuple(float(c) for c in q_w),
                            p=p_w, d=d, fclose=fclose, approach=approach,
                            horiz=horiz, q7=q7, score=score))
        print(f"[probe] {i+1}/{len(cands)} {nm:14s} PLANNED d={d:.3f} "
              f"fclose=[{fclose[0]:+.2f},{fclose[1]:+.2f},{fclose[2]:+.2f}]"
              f" appr=[{approach[0]:+.2f},{approach[1]:+.2f},"
              f"{approach[2]:+.2f}] q=[{q_w[0]:+.4f},{q_w[1]:+.4f},"
              f"{q_w[2]:+.4f},{q_w[3]:+.4f}]")

    print("\n" + "=" * 66)
    if not results:
        print("[probe] NOTHING plannable -- widen family / move cube.")
    else:
        results.sort(key=lambda r: r["score"])
        print(f"[probe] {len(results)} plannable. Top 12 by score "
              f"(|EE-cube| + 2*|fclose_z| + up-approach penalty):")
        for r in results[:12]:
            print(f"  {r['nm']:14s} score={r['score']:.3f} d={r['d']:.3f} "
                  f"horiz={r['horiz']:.2f} fclose=[{r['fclose'][0]:+.2f},"
                  f"{r['fclose'][1]:+.2f},{r['fclose'][2]:+.2f}] "
                  f"q=[{r['q'][0]:+.5f},{r['q'][1]:+.5f},{r['q'][2]:+.5f},"
                  f"{r['q'][3]:+.5f}]")
        b = results[0]
        fc = b["fclose"]
        print(f"\n[probe] BEST = {b['nm']}")
        print(f"[probe]   grasp_quat_r1: [{b['q'][0]:+.6f}, "
              f"{b['q'][1]:+.6f}, {b['q'][2]:+.6f}, {b['q'][3]:+.6f}]")
        print(f"[probe]   EE world pos = ({b['p'][0]:+.3f},"
              f"{b['p'][1]:+.3f},{b['p'][2]:+.3f})  |EE-cube|={b['d']:.3f}")
        print(f"[probe]   finger-close world axis = [{fc[0]:+.3f},"
              f"{fc[1]:+.3f},{fc[2]:+.3f}]")
        print(f"[probe]   -> cube's thin 3cm axis is its LOCAL X. With "
              f"cube_quat_r1=[1,0,0,0] that axis = world X; |fclose.x|="
              f"{abs(fc[0]):.2f} (want ~1 so fingers close on the 3cm "
              f"dim). Rotate cube_quat_r1 about Z if fclose ~ world Y.")
    print("=" * 66 + "\n")
    sim.close()


if __name__ == "__main__":
    main()
