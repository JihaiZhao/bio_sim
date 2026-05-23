#
# Gripper = force-controlled friction grasp (genie_sim ParallelGripper).
# The fingers close under a capped force; PhysX contact friction holds the
# cube. cuRobo attach is planning-only (so the carried legs avoid
# self-collision with the payload) and does NOT touch physics.
#
# `mode` selects the physics contract (cfg `grasp_mode`):
#   "physics" -> friction only (above). Realistic, but the KINEMATIC swerve
#                base teleport flings the cube mid-carry (PhysX can't push a
#                teleport through a friction contact).
#   "assist"  -> ASSISTED grasp: ALSO weld the object to the grasp link with
#                a UsdPhysics.FixedJoint at the live contacted relative pose,
#                so it rigidly follows the gripper through the kinematic base
#                motion; not friction-dependent. Friction close + cuRobo
#                attach still run (visuals/planning unchanged); the weld is
#                the load path.
#
# The actual drive control (zero stiffness + capped max force + closing
# velocity, reasserted every step) lives in G2Robot.set_gripper /
# _apply_gripper. This class just sequences attach/detach + bookkeeping.
#

from __future__ import annotations

import numpy as np

GRASP_GAP_TOL = 0.10  # m; loose sanity (force-closure tolerates cm-level
#                       arm error; only fail a wildly-missed reach)


# ---- tiny quaternion helpers (wxyz, Isaac convention) --------------------
def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _qrot(q, v):
    return _qmul(_qmul(q, np.array([0.0, v[0], v[1], v[2]])), _qconj(q))[1:]


def _compose(A, B):
    """Pose A ∘ B (apply B then A). pose = (p[3], q[4] wxyz)."""
    pa, qa = A
    pb, qb = B
    return (pa + _qrot(qa, pb), _qmul(qa, qb))


def _inv(T):
    p, q = T
    qi = _qconj(q)
    return (-_qrot(qi, p), qi)


def _world_scale(stage, prim_path: str) -> np.ndarray:
    """Accumulated world-space scale of a prim by composing every ancestor's
    xformOp:scale. Returns (sx, sy, sz). Used by _weld to convert the
    WORLD-frame relative position into the body's LOCAL frame for the
    FixedJoint localPosN attr (which is scale-affected)."""
    from pxr import Gf, Usd, UsdGeom

    m = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)) \
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Pull length of each basis vector -- robust to any rotation in the
    # matrix (we don't care about per-row direction, just magnitude).
    rows = [Gf.Vec3d(m[0][0], m[0][1], m[0][2]),
            Gf.Vec3d(m[1][0], m[1][1], m[1][2]),
            Gf.Vec3d(m[2][0], m[2][1], m[2][2])]
    return np.array([rows[0].GetLength(),
                     rows[1].GetLength(),
                     rows[2].GetLength()], dtype=np.float64)


class Gripper:
    def __init__(self, robot, mode: str = "physics"):
        self._robot = robot
        self._held = None
        # Assist mode: one FixedJoint per env, all created in _weld()
        # under /World/env_i/{obj}/grasp_weld. Cloner replicates the
        # robot + plate with IDENTICAL relative geometry, so the
        # localPos / localRot constants (computed once on env_0) reuse
        # bit-for-bit across envs.
        self._weld_joint_paths: list[str] = []
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        mode = (mode or "physics").lower()
        if mode not in ("physics", "assist"):
            print(f"[gripper] unknown grasp_mode {mode!r} -> 'physics'")
            mode = "physics"
        self.mode = mode
        print(f"[gripper] grasp_mode = {mode}")

    @property
    def holding(self) -> str | None:
        return self._held

    def grasp(self, ctx, obj_name: str) -> bool:
        """Called AFTER the fingers have closed (force mode). Sanity-checks
        the reach, then cuRobo-attaches the payload for planning."""
        ee_p, _ = self._robot.ee_world_pose(ctx)
        obj_p, _ = ctx.scene.object_pose(obj_name)
        gap = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_p)))
        print(f"[gripper] grasp gap |ee - object| = {gap:.4f} m")
        # Did the fingers actually move? open target = GRIP_OPEN_Q (0.8);
        # if idx81 is still ~0.8 the drive isn't actuating; if ~0 it closed.
        gi, gpos = self._robot.gripper_joint_state()
        ee_w, _ = self._robot.ee_world_pose(ctx)
        gpos_s = ("n/a" if gpos is None
                  else "[" + ", ".join(f"{p:.4f}" for p in gpos) + "]")
        print(f"[gripper] DIAG cmd-joint(idx81) dof={gi} pos={gpos_s} "
              f"(open~0.8, closed~0) "
              f"ee_world={np.round(np.asarray(ee_w),3).tolist()} "
              f"obj_world={np.round(np.asarray(obj_p),3).tolist()}")
        if gap > GRASP_GAP_TOL:
            print(f"[gripper] gap exceeds {GRASP_GAP_TOL} m -> reach missed")
            return False

        # cuRobo payload-aware planning for the carried legs (planning only;
        # no physics constraint -- the grip is held by finger friction).
        cu_js = self._robot.read_cu_js()
        p_w, q_w = ctx.scene.object_pose(obj_name)
        p_b, q_b = self._robot.base.world_to_base(p_w, q_w)
        dims = ctx.scene.object_dims(obj_name)
        self._robot.arm.attach_payload(
            cu_js, obj_name, dims,
            [float(p_b[0]), float(p_b[1]), float(p_b[2]),
             float(q_b[0]), float(q_b[1]), float(q_b[2]), float(q_b[3])],
        )
        # Lock the fingers as a stiff vice at the contacted aperture so the
        # carry (base accel, arm motion) can't ratchet them shut and eject
        # the thin box. Still pure friction -- no joint to the object.
        self._robot.clamp_hold()
        # assist mode: rigidly weld the object to the grasp link at its
        # current (just-contacted) LIVE relative pose, so the kinematic base
        # teleport can't eject it. Friction close + cuRobo attach above
        # still ran, so visuals/planning are unchanged; this is the load
        # path. physics mode leaves it pure friction (no joint).
        if self.mode == "assist":
            self._weld(ctx, obj_name)
        self._held = obj_name
        held_by = ("rigid FixedJoint weld" if self.mode == "assist"
                   else "friction vice-hold")
        print(f"[gripper] grasped {obj_name} ({held_by}; cuRobo attached)")
        return True

    def release(self, ctx) -> None:
        """Synchronous full release: open fingers + unweld + detach all at
        once. Used by the R-key env-reset path (sim/app.py) where there's
        no time budget for a phased open. The Release SKILL splits this
        into release_open() + release_unweld() with a wait window between
        so the fingers physically clear the plate before the FixedJoint is
        deleted (otherwise plate is still in finger contact when it goes
        free, and the fast PD open swing can flick it sideways)."""
        if self._held is None:
            return
        ee_p, _ = self._robot.ee_world_pose(ctx)
        obj_p, _ = ctx.scene.object_pose(self._held)
        d = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_p)))
        print(f"[gripper] at release: |ee-object|={d:.4f} m  "
              f"object_z={float(obj_p[2]):.3f} (held OK if small & z>0.5)")
        if self.mode == "assist":
            self._unweld(ctx)
        self._robot.arm.detach_payload()
        self._robot.set_gripper(close=False)
        print(f"[gripper] released {self._held}")
        self._held = None

    # ---- phased release (used by the Release skill in normal task flow) ---
    def release_open(self, ctx) -> None:
        """Phase 1 of phased release: open the fingers ONLY. Weld stays in
        place so the plate is rigidly held (no friction needed) while the
        PD-driven fingers slowly swing open and clear the plate."""
        if self._held is None:
            return
        ee_p, _ = self._robot.ee_world_pose(ctx)
        obj_p, _ = ctx.scene.object_pose(self._held)
        d = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_p)))
        print(f"[gripper] release_open: |ee-object|={d:.4f} m  "
              f"object_z={float(obj_p[2]):.3f}")
        # Switch to position-open target; PD drives the fingers wide over
        # the next ~0.5-2 s (slow because the carry-time effort cap is
        # ~300 N + KD=1e3 -> terminal vel ~0.3 rad/s).
        self._robot.set_gripper(close=False)
        print(f"[gripper] fingers PD-opening; weld still holds {self._held}")

    def release_unweld(self, ctx) -> None:
        """Phase 2 of phased release: now that the fingers have moved off
        the plate, delete the FixedJoint and detach the cuRobo payload.
        Plate becomes a free dynamic body and falls onto the place
        surface."""
        if self._held is None:
            return
        if self.mode == "assist":
            self._unweld(ctx)
        self._robot.arm.detach_payload()
        print(f"[gripper] release_unweld: weld removed; {self._held} is free")
        self._held = None

    # ---- assist-mode rigid weld (grasp_mode: assist) ----------------------
    # Robot-agnostic: both G2Robot and R1ProRobot expose grasp_link_path() +
    # grasp_link + ee_world_pose(); the object prim follows scene's
    # /World/<name> convention.
    #
    # WHY NOT UsdGeom.ComputeLocalToWorldTransform for the live poses:
    # under PhysX/fabric the USD xform attrs of a simulated prim stay at the
    # SPAWN pose (physics writes to fabric, not back to USD), so a relative
    # transform read that way welds the cube at a stale offset -> it hangs
    # OUTSIDE the fingers (the observed bug). So:
    #   * object & ee_link world poses come from the proven-LIVE accessors
    #     (scene.object_pose / robot.ee_world_pose);
    #   * the ee_link->grasp_link transform K is INVARIANT (both are rigidly
    #     fixed in the same gripper body; ee_center is a frame on the base
    #     link), so reading K as a *ratio* of two USD transforms is correct
    #     even though each is individually stale.
    def _weld(self, ctx, obj_name: str) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

        stage = ctx.world.stage
        env_root = self._robot.env_root
        link_path = self._robot.grasp_link_path(stage)   # env_0 grasp_link
        obj_path = f"{env_root}/{obj_name}"
        if not stage.GetPrimAtPath(obj_path).IsValid():
            raise RuntimeError(f"_weld: no prim at {obj_path}")

        # --- live world poses (p[3], q[4] wxyz) ---------------------------
        # Both position AND orientation come from the LIVE physics state,
        # matching OmniGibson's assisted-grasp convention: the relative
        # offset captures the MECHANICALLY-CORRECT contact geometry. An
        # earlier version snapped q to scene.cube_quat to "force horizontal"
        # at grasp time, but that introduces a finger<->plate penetration
        # the instant the weld is created (PhysX has to resolve plate from
        # live quat to the snapped quat), which can pop/jitter. The plate
        # tilting during carry is a separate issue caused by wrist-roll
        # drift in the trajectory, not by the live-quat weld -- a snap here
        # does not fix it, the right fix is in the controller / IK target.
        p_obj, q_obj = ctx.scene.object_pose(obj_name)
        p_ee, q_ee = self._robot.ee_world_pose(ctx)
        T_obj = (np.asarray(p_obj, float), np.asarray(q_obj, float))
        T_ee = (np.asarray(p_ee, float), np.asarray(q_ee, float))

        # --- invariant ee_link -> grasp_link transform K (USD ratio) ------
        ee_path = self._link_path(stage, self._robot.ee_link)
        tc = Usd.TimeCode.Default()

        def _usd_pose(path):
            m = UsdGeom.Xformable(
                stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(tc)
            t = m.ExtractTranslation()
            qq = m.ExtractRotationQuat()
            iq = qq.GetImaginary()
            return (np.array([t[0], t[1], t[2]], float),
                    np.array([qq.GetReal(), iq[0], iq[1], iq[2]], float))

        ee_usd = _usd_pose(ee_path)
        gl_usd = _usd_pose(link_path)
        K = _compose(_inv(ee_usd), gl_usd)            # ee_local -> gl_local

        # grasp_link live world pose, then object expressed in it:
        T_gl = _compose(T_ee, K)
        rel_p, rel_q = _compose(_inv(T_gl), T_obj)    # object in grasp_link

        # FixedJoint's LocalPosN is expressed in each body's LOCAL frame,
        # which is affected by the body chain's xformOp:scale. If either
        # body sits under a non-unit scale, the rel positions we computed
        # in WORLD units must be divided by that scale to land in the right
        # spot (cf. OmniGibson manipulation_robot.py:1562-1565). Today
        # everything we weld is at scale=1 -- but loudly catch the day that
        # changes instead of producing a silently-offset weld.
        s0 = _world_scale(stage, link_path)
        s1 = _world_scale(stage, obj_path)
        rel_p_local0 = rel_p / s0
        if not (np.allclose(s0, 1.0) and np.allclose(s1, 1.0)):
            print(f"[gripper] WARN assist weld under non-unit scale: "
                  f"grasp_link {s0.tolist()} obj {s1.tolist()} -- "
                  f"localPos0 divided by grasp_link scale; verify weld.")

        # Phase 4: one FixedJoint per env. Clones share IDENTICAL relative
        # geometry (Cloner is a rigid translation), so the same (rel_p,
        # rel_q) authoring works for every env_i once the body0/body1
        # paths are re-prefixed.
        num_envs = max(1, int(getattr(self._robot, "_num_envs", 1)))
        link_rel = link_path[len(env_root):]   # leading "/" preserved
        obj_rel = obj_path[len(env_root):]
        self._weld_joint_paths = []
        for i in range(num_envs):
            env_i_root = f"/World/env_{i}"
            link_path_i = f"{env_i_root}{link_rel}"
            obj_path_i = f"{env_i_root}{obj_rel}"
            joint_path_i = f"{obj_path_i}/grasp_weld"
            j = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path_i))
            j.CreateBody0Rel().SetTargets([Sdf.Path(link_path_i)])
            j.CreateBody1Rel().SetTargets([Sdf.Path(obj_path_i)])
            j.CreateLocalPos0Attr().Set(
                Gf.Vec3f(*[float(x) for x in rel_p_local0]))
            j.CreateLocalRot0Attr().Set(
                Gf.Quatf(float(rel_q[0]),
                         Gf.Vec3f(float(rel_q[1]), float(rel_q[2]),
                                  float(rel_q[3]))))
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            self._weld_joint_paths.append(joint_path_i)
        suffix = (f" (+{num_envs - 1} env clones)" if num_envs > 1 else "")
        print(f"[gripper] assist-mode weld: FixedJoint "
              f"{self._weld_joint_paths[0]}{suffix} "
              f"({self._robot.grasp_link} <-> {obj_name}) "
              f"rel_p={np.round(rel_p,4).tolist()}")

    def _link_path(self, stage, link_name: str) -> str:
        """USD prim path of a named robot link (first match under the robot
        root). grasp_link has a dedicated cached accessor; this is the
        generic form used for ee_link."""
        root = self._robot.robot_prim_path
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if prim.GetName() == link_name and p.startswith(root):
                return p
        raise RuntimeError(f"_link_path: '{link_name}' not found under {root}")

    def _unweld(self, ctx) -> None:
        if not self._weld_joint_paths:
            return
        stage = ctx.world.stage
        for jp in self._weld_joint_paths:
            if stage.GetPrimAtPath(jp).IsValid():
                stage.RemovePrim(jp)
        suffix = (f" (+{len(self._weld_joint_paths) - 1} env clones)"
                  if len(self._weld_joint_paths) > 1 else "")
        print(f"[gripper] assist-mode unweld: removed "
              f"{self._weld_joint_paths[0]}{suffix}")
        self._weld_joint_paths = []
