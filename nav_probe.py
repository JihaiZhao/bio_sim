#!/usr/bin/env python
#
# nav_probe.py -- DYNAMIC base navigation-only probe (no arm / cuRobo / grasp).
#
# Question it answers: if the base is driven for real (NOT a kinematic
# teleport), can a turn-drive-turn controller take it A->B->A stably, and
# how accurately/planar does it settle? Those numbers decide whether a
# dynamic base is viable before we touch the grasp pipeline.
#
#   G2 (swerve, real wheel contact):   python nav_probe.py --headless_mode native
#   R1 Pro (BEHAVIOR-1K holonomic):    python nav_probe.py --r1 --headless_mode native
#
# The R1 path is a faithful BEHAVIOR-1K port: r1pro_holonomic.usda
# world-anchors the virtual base chain but leaves z/rx/ry PASSIVE (the
# 3-wheel base is statically stable and rests on its boundingSphere
# wheels -- no balance controller, exactly like OmniGibson). x/y/rz are
# POSITION-driven by HolonomicBaseDriver via the rate-limited q_to_action
# interface (holonomic strafe; no in-place turn -> welded wheels unscrubbed).
#

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _yaw_rp_from_quat(q):
    """quat (w,x,y,z) -> (yaw, roll, pitch) in rad (Z-up)."""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    # ZYX-ish; we only need yaw + how far roll/pitch drift from 0 (tip).
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return yaw, roll, pitch


def _diag_articulation(sim, robot, driver):
    """One-shot falsifiable check: are the 6 wheel/steer joints ACTUALLY
    0-DOF at sim time (not just typed FixedJoint in the USD), is the
    WorldAnchor live, and what type/state are the passive base joints?
    A wheel/steer joint showing up in dof_names => the weld did NOT take
    at parse time => rx/ry unconstrained => the topple is explained."""
    names = list(robot.dof_names)
    print(f"[navprobe][diag] n_dof={len(names)} dof_names={names}")
    bad = [n for n in names
           if ("wheel_motor" in n or "steer_motor" in n)]
    print(f"[navprobe][diag] wheel/steer joints STILL DOFs (should be "
          f"EMPTY -- welded): {bad if bad else 'none (OK, welded)'}")
    want = {"steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3",
            "wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3",
            "WorldAnchor", "base_footprint_z_joint",
            "base_footprint_rx_joint", "base_footprint_ry_joint",
            "base_footprint_x_joint", "base_footprint_y_joint",
            "base_footprint_rz_joint"}
    for pr in sim.stage.Traverse():
        nm = pr.GetName()
        if nm not in want:
            continue
        enb = pr.GetAttribute("physics:jointEnabled")
        enb = enb.Get() if enb and enb.IsValid() else "n/a"
        act = pr.IsActive()
        print(f"[navprobe][diag]   {pr.GetPath()}  type={pr.GetTypeName()} "
              f"active={act} jointEnabled={enb}")


def run_r1(sim, args, np, Robot, add_reference_to_stage,
           join_path, asset_root, HolonomicBaseDriver):
    """R1 Pro probe, FULL BEHAVIOR-1K holonomic base.

    Loads r1pro_holonomic.usda (WorldAnchor only; z/rx/ry PASSIVE). The
    base is moved by PhysX POSITION drives on x/y/rz via the
    HolonomicBaseDriver q_to_action interface (world goal -> rate-limited
    incremental joint targets; holonomic strafe, no in-place 90 deg turn,
    no per-step set_joint_positions teleport).
    """
    zprobe = bool(getattr(args, "zprobe", False))
    usd_path = join_path(asset_root(), "robot/r1pro/usd/r1pro_holonomic.usda")
    robot_prim = "/World/r1pro"
    print(f"[navprobe] R1 Pro (BEHAVIOR-1K holonomic)  USD = {usd_path}")

    sim.world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim)
    robot = sim.world.scene.add(
        Robot(prim_path=robot_prim, name="r1pro",
              position=np.array([0.0, 0.0, 0.0])))
    sim.world.initialize_physics()
    robot.initialize()
    av = robot._articulation_view
    av.initialize()

    driver = HolonomicBaseDriver(robot, av, sim.stage, robot_prim)

    FY = -math.pi / 2.0
    GOALS = [("A", 0.0, 0.0, FY), ("B", 1.0, 0.0, FY), ("A'", 0.0, 0.0, FY)]

    sim.world.play()

    # arrival tolerances; the driver's rate-limit IS the trajectory.
    POS_TOL, YAW_TOL = 0.05, math.radians(3.0)
    SETTLE_PREROLL = 60          # let z/rx/ry settle on the wheels first
    SETTLE_NEED = 30             # consecutive in-tol steps = settled
    MAXSTEPS = 3000 if zprobe else 12000

    started = False              # goal-feed engaged (post settle pre-roll)
    gi = 0
    t0 = None
    settle_ctr = 0
    step = 0
    settle_start = None          # step at which the driver came up
    z0 = None
    prev_bl = None
    dstep_hist = []
    stats = {"max_tip": 0.0, "max_dz": 0.0}

    while sim.is_running() and step < MAXSTEPS:
        sim.step(render=True)
        if not sim.world.is_playing():
            continue
        step += 1
        if robot.get_joints_state() is None:
            continue

        # one-time driver bring-up (needs live joints state)
        if not driver._ready:
            driver.setup()
            # TEST SCAFFOLD ONLY -- stands in for the future arm interface.
            # The driver's contract is mobile-base + torso; arms/grippers
            # are a separate interface (co-tuned at final integration).
            # For a clean base-only probe we pin them at the loaded pose
            # so arm-droop doesn't confound the base result. One-time
            # position drive (NOT per-step set_joint_positions).
            if driver.arm_idx:
                aidx = np.asarray(driver.arm_idx, dtype=np.int32)
                qa = np.asarray(robot.get_joints_state().positions,
                                dtype=np.float32)[aidx]
                driver.av.set_gains(
                    kps=np.full((1, len(aidx)), 1.0e6, dtype=np.float32),
                    kds=np.full((1, len(aidx)), 1.0e4, dtype=np.float32),
                    joint_indices=aidx)
                for _di in driver.arm_idx:
                    try:
                        driver.av.switch_dof_control_mode("position", _di)
                    except Exception:  # noqa: BLE001
                        pass
                driver.av.set_joint_position_targets(
                    qa.reshape(1, -1), joint_indices=aidx)
                print(f"[navprobe] arm scaffold: pinned {len(aidx)} "
                      f"arm/grip DOFs at load pose (placeholder for the "
                      f"real arm interface)")
            settle_start = step
            z0 = driver.base_pose()[2]
            _diag_articulation(sim, robot, driver)
            print(f"[navprobe] driver up; settling {SETTLE_PREROLL} steps "
                  f"on wheels (z/rx/ry PASSIVE)  z0={z0:+.4f}")
            continue

        bx, by, bz, byaw, tip = driver.base_pose_full()
        stats["max_tip"] = max(stats["max_tip"], tip)
        stats["max_dz"] = max(stats["max_dz"], abs(bz - z0))

        # DENSE early-frame trace: see the topple MODE (clean rx/ry tip vs
        # z-sink vs anchor free-float) + base joint scalars vs true pose.
        if step - settle_start <= 150 and (step - settle_start) % 10 == 0:
            js = robot.get_joints_state()
            p = js.positions
            ji = driver.idx
            print(f"[navprobe][trace] t={step:4d} "
                  f"BL=({bx:+.3f},{by:+.3f},{bz:+.4f}) tip={tip:6.2f}deg | "
                  f"joints x={float(p[ji['x']]):+.3f} "
                  f"y={float(p[ji['y']]):+.3f} z={float(p[ji['z']]):+.4f} "
                  f"rx={math.degrees(float(p[ji['rx']])):+6.1f} "
                  f"ry={math.degrees(float(p[ji['ry']])):+6.1f} "
                  f"rz={math.degrees(float(p[ji['rz']])):+6.1f}")

        # let the chassis find its level on the wheels before driving
        if step - settle_start <= SETTLE_PREROLL:
            driver.stop()
            continue

        if zprobe:
            driver.stop()
            if step % 60 == 0:
                print(f"[navprobe][zprobe] t={step:5d} "
                      f"base_link_z={bz:+.5f} tip={tip:5.2f}deg "
                      f"dz={bz - z0:+.5f}")
            continue

        if not started:
            started = True
            t0 = step
            print(f"[navprobe] settled  base_link=({bx:+.3f},{by:+.3f},"
                  f"{bz:+.4f}) yaw={math.degrees(byaw):+.1f} tip={tip:.2f}deg"
                  f"  -> holonomic A(0,0,-90)->B(1,0,-90)->A(0,0,-90)")

        name, gx, gy, gyaw = GOALS[gi]
        dt = float(sim.physics_dt)
        # Feed the WORLD-absolute goal straight to the q_to_action
        # interface; the driver's per-step rate-limit IS the trajectory
        # (holonomic strafe -- no in-place 90 deg turn, so the welded
        # wheels are never scrubbed; position drive holds pose against
        # any residual skid).
        driver.drive_to(gx, gy, gyaw, dt)

        # judder diagnostic: real chassis per-step world motion
        if prev_bl is not None:
            dstep = math.hypot(bx - prev_bl[0], by - prev_bl[1])
        else:
            dstep = 0.0
        prev_bl = (bx, by)
        dstep_hist.append(dstep)
        if len(dstep_hist) > 30:
            dstep_hist.pop(0)
        d_mean = sum(dstep_hist) / max(1, len(dstep_hist))
        d_max = max(dstep_hist) if dstep_hist else 0.0

        perr = math.hypot(gx - bx, gy - by)
        yerr = abs(math.atan2(math.sin(gyaw - byaw),
                              math.cos(gyaw - byaw)))

        if step % 30 == 0:
            ratio = (d_max / d_mean) if d_mean > 1e-9 else 0.0
            print(f"[navprobe] t={step:5d} goal={name} "
                  f"base_link=({bx:+.3f},{by:+.3f},{bz:+.4f}) "
                  f"yaw={math.degrees(byaw):+6.1f} "
                  f"|perr|={perr:.3f} yerr={math.degrees(yerr):4.1f} "
                  f"dstep(mm) mean={d_mean*1e3:6.2f} max={d_max*1e3:6.2f} "
                  f"jud={ratio:4.1f} tip={tip:4.2f}deg")

        if perr < POS_TOL and yerr < YAW_TOL:
            settle_ctr += 1
        else:
            settle_ctr = 0

        if settle_ctr >= SETTLE_NEED:
            print(f"[navprobe] >>> {name} REACHED steps={step - t0} "
                  f"|perr|={perr*100:.1f}cm yerr={math.degrees(yerr):.2f}deg "
                  f"max_tip={stats['max_tip']:.2f}deg "
                  f"max_dz={stats['max_dz']*100:.2f}cm "
                  f"({'STABLE' if stats['max_tip'] < 5.0 else 'TIPPING'})")
            gi += 1
            if gi >= len(GOALS):
                print("[navprobe] ALL GOALS DONE -- R1 BEHAVIOR-1K holonomic "
                      "nav VIABLE (see per-leg errors / tip above)")
                break
            t0 = step
            settle_ctr = 0
            stats = {"max_tip": 0.0, "max_dz": 0.0}

    if step >= MAXSTEPS:
        print(f"[navprobe] HIT STEP CAP at goal idx {gi} -- did NOT "
              f"converge (drifting / tipping / stuck)")
    sim.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--headless_mode", type=str, default=None)
    p.add_argument("--robot", type=str, default="G2_omnipicker_fixed_dual.yml")
    p.add_argument("--free_base", action="store_true",
                   help="load robot_freebase.usda (root FixedJoint "
                        "deactivated -> floating base, real wheel drive)")
    p.add_argument("--r1", action="store_true",
                   help="probe the R1 Pro instead of G2: BEHAVIOR-1K "
                        "holonomic virtual base (x/y/rz position-driven, "
                        "z/rx/ry passive -> rests on wheels).")
    p.add_argument("--zprobe", action="store_true",
                   help="R1: don't navigate -- just let it settle and "
                        "report base_link world z / tip (z/rx/ry passive).")
    args, _ = p.parse_known_args()

    # 1. boot sim FIRST (RTX/Kit), then heavy imports (same ordering as play.py)
    from bio_sim.sim import SimApp
    sim = SimApp(headless=args.headless_mode)

    import numpy as np
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.stage import add_reference_to_stage
    from curobo.util_file import join_path, load_yaml

    from bio_sim.asset_lib import asset_root
    from bio_sim.robot.base import (
        MAX_ANG_ACCEL, MAX_LIN_ACCEL, NavController, SwerveBaseController,
    )
    from bio_sim.robot.holonomic import HolonomicBaseDriver

    if args.r1:
        return run_r1(sim, args, np, Robot, add_reference_to_stage,
                      join_path, asset_root, HolonomicBaseDriver)

    cfg_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config", "curobo", "configs", "robot")
    robot_cfg = load_yaml(os.path.join(cfg_dir, args.robot))["robot_cfg"]
    kin = robot_cfg["kinematics"]
    rel_usd = kin["usd_path"]
    if args.free_base:
        rel_usd = os.path.join(os.path.dirname(rel_usd), "robot_freebase.usda")
    usd_path = join_path(asset_root(), rel_usd)
    robot_prim = "/World/" + kin["usd_robot_root"].strip("/")
    print(f"[navprobe] USD = {usd_path}  free_base={args.free_base}")

    # 2. minimal world: ground + robot only (no scene, no cuRobo)
    sim.world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim)
    robot = sim.world.scene.add(
        Robot(prim_path=robot_prim, name="robot",
              position=np.array([0.0, 0.0, 0.0])))
    sim.world.initialize_physics()
    robot.initialize()
    robot._articulation_view.initialize()

    swerve = SwerveBaseController(robot, robot._articulation_view)
    nav = NavController(swerve)

    # turn-drive-turn over ~1 m WITH the ~90 deg in-place turns -- the exact
    # motion that shears the carried cube in the kinematic base.
    FY = -math.pi / 2.0
    GOALS = [("A", 0.0, 0.0, FY), ("B", 1.0, 0.0, FY), ("A'", 0.0, 0.0, FY)]

    sim.world.play()  # probe always auto-plays (headless or windowed)

    def true_pose():
        pos, quat = robot.get_world_pose()
        yaw, roll, pitch = _yaw_rp_from_quat(quat)
        return (float(pos[0]), float(pos[1]), float(pos[2]),
                yaw, roll, pitch)

    configured = False
    gi = 0
    nav.set_goal(*GOALS[0][1:])
    t0 = None
    z0 = None
    stats = {"max_dz": 0.0, "max_roll": 0.0, "max_pitch": 0.0}
    settle_ctr = 0
    step = 0
    SETTLE_NEED = 30          # consecutive low-motion steps = "settled"
    MAXSTEPS = 12000          # hard cap

    print("[navprobe] DYNAMIC base (real wheel drive, no teleport). "
          "Goals: A(0,0,-90) -> B(1,0,-90) -> A(0,0,-90)")

    while sim.is_running() and step < MAXSTEPS:
        sim.step(render=True)
        if not sim.world.is_playing():
            continue
        step += 1

        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        if not configured:
            swerve.configure_drive_modes()
            configured = True
            x, y, z, yaw, _, _ = true_pose()
            z0 = z
            t0 = step
            print(f"[navprobe] start pose x={x:.3f} y={y:.3f} z={z:.3f} "
                  f"yaw={math.degrees(yaw):.1f}")

        # --- feed the controller the TRUE root pose (not kinematic _pose) ---
        x, y, z, yaw, roll, pitch = true_pose()
        swerve._pose = [x, y, yaw]      # so nav._twist_to_goal() uses truth
        swerve._z = z
        stats["max_dz"] = max(stats["max_dz"], abs(z - z0))
        stats["max_roll"] = max(stats["max_roll"], abs(roll))
        stats["max_pitch"] = max(stats["max_pitch"], abs(pitch))

        # --- twist from the SAME turn-drive-turn phase machine -------------
        tgt = np.array(nav._twist_to_goal(), dtype=float)
        dt = float(sim.physics_dt)
        dv = np.array([MAX_LIN_ACCEL, MAX_LIN_ACCEL, MAX_ANG_ACCEL]) * dt
        nav._v += np.clip(tgt - nav._v, -dv, dv)
        vx, vy, wz = float(nav._v[0]), float(nav._v[1]), float(nav._v[2])

        # --- drive the REAL wheels; PhysX moves the chassis (no teleport) --
        cur_steer = swerve.read_cur_steer(sim_js)
        swerve.update(vx, vy, wz, cur_steer)

        name, gx, gy, gyaw = GOALS[gi]
        pos_err = math.hypot(gx - x, gy - y)
        yaw_err = abs(math.atan2(math.sin(gyaw - yaw), math.cos(gyaw - yaw)))
        speed = math.hypot(vx, vy) + abs(wz)

        if step % 60 == 0:
            print(f"[navprobe] t={step:5d} goal={name} "
                  f"pos=({x:+.3f},{y:+.3f}) yaw={math.degrees(yaw):+6.1f} "
                  f"|perr|={pos_err:.3f} yawerr={math.degrees(yaw_err):4.1f} "
                  f"dz={z - z0:+.3f} roll={math.degrees(roll):+.1f} "
                  f"pitch={math.degrees(pitch):+.1f}")

        arrived = nav.arrived()
        if arrived and speed < 0.02:
            settle_ctr += 1
        else:
            settle_ctr = 0

        if arrived and settle_ctr >= SETTLE_NEED:
            print(f"[navprobe] >>> {name} REACHED  steps={step - t0}  "
                  f"final |perr|={pos_err*100:.1f}cm  "
                  f"yawerr={math.degrees(yaw_err):.2f}deg  "
                  f"max_dz={stats['max_dz']*100:.1f}cm  "
                  f"max_roll={math.degrees(stats['max_roll']):.1f}deg  "
                  f"max_pitch={math.degrees(stats['max_pitch']):.1f}deg")
            gi += 1
            if gi >= len(GOALS):
                print("[navprobe] ALL GOALS DONE -- dynamic nav viable "
                      "(see per-leg errors / tip indicators above)")
                break
            nav.clear_goal()
            nav.set_goal(*GOALS[gi][1:])
            t0 = step
            settle_ctr = 0
            stats = {"max_dz": 0.0, "max_roll": 0.0, "max_pitch": 0.0}

    if step >= MAXSTEPS:
        print(f"[navprobe] HIT STEP CAP at goal index {gi} "
              f"({GOALS[gi][0] if gi < len(GOALS) else 'done'}) -- "
              f"did NOT converge (see trace: drifting / tipping / stuck)")
    sim.close()


if __name__ == "__main__":
    main()
