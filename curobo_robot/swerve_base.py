#
# G2 4-wheel swerve-drive base: inverse kinematics, Isaac articulation
# controller, and keyboard teleop.
#
# Milestone 1 of mobile manipulation: drive the chassis with a body-frame
# twist (vx, vy, omega) and validate the swerve kinematics. The arm-planning
# path in g2_motion_gen_reacher.py is untouched; this only commands the 8
# wheel DOFs (4 steering + 4 drive), which are disjoint from the arm DOFs.
#
# Geometry/limits are from assets/robot/G2/G2_omnipicker_fixed_dual.urdf;
# the actually-loaded asset is robot.usda, which ships the drive joints as a
# stiff position drive holding 0 -> configure_drive_modes() MUST switch them
# to velocity control or the wheels stay locked.
#

import math

import numpy as np

# ---- module geometry (chassis_link frame; base->chassis is a fixed identity
# joint so this is effectively the base frame for planar motion) ------------
MODULE_ORDER = ("lwheel_front", "rwheel_front", "rwheel_rear", "lwheel_rear")

# (Lx, Ly): joint1 origin x,y in chassis_link, from the URDF.
MODULE_XY = {
    "lwheel_front": (0.23, 0.218),
    "rwheel_front": (0.23, -0.218),
    "rwheel_rear": (-0.23, -0.218),
    "lwheel_rear": (-0.23, 0.218),
}

STEER_JOINTS = {
    "lwheel_front": "idx111_chassis_lwheel_front_joint1",
    "rwheel_front": "idx131_chassis_rwheel_front_joint1",
    "rwheel_rear": "idx141_chassis_rwheel_rear_joint1",
    "lwheel_rear": "idx121_chassis_lwheel_rear_joint1",
}
DRIVE_JOINTS = {
    "lwheel_front": "idx112_chassis_lwheel_front_joint2",
    "rwheel_front": "idx132_chassis_rwheel_front_joint2",
    "rwheel_rear": "idx142_chassis_rwheel_rear_joint2",
    "lwheel_rear": "idx122_chassis_lwheel_rear_joint2",
}

# ---- tunables (TODO: confirm against hardware / robot.usda) ----------------
# Rolling radius is only from a commented-out collision cylinder in the URDF;
# it scales wheel speed magnitude (not direction). Tune against observed vs
# commanded translation.
WHEEL_RADIUS = 0.07
# Steering joint1 URDF limit is +-2.97 rad (NOT continuous).
STEER_LIMIT = 2.97
STEER_MARGIN = 0.02
STEER_LIMIT_EFF = STEER_LIMIT - STEER_MARGIN
# Teleop command caps.
MAX_LIN_SPEED = 0.4  # m/s
MAX_ANG_SPEED = 0.6  # rad/s
# Per-update slew toward the commanded twist (units/s * dt-free; applied once
# per outer loop iteration).
LIN_RAMP = 0.04
ANG_RAMP = 0.06
# PD gains. Drive (velocity mode) MUST have stiffness 0.
STEER_KP = 800.0
STEER_KD = 40.0
DRIVE_KP = 0.0
DRIVE_KD = 30.0
WHEEL_MAX_EFFORT = 200.0
ZERO_TWIST_EPS = 1e-4
# Natural standing height: base_link sits ~0.04 m above wheel ground contact
# (URDF FK). The kinematic base holds the root at this z so the wheels sit
# visually on the floor.
BASE_STAND_Z = 0.04
# Cheap insurance if a sign convention is mirrored on screen.
WHEEL_DIR_SIGN = 1.0

# Set True to print per-step twist / base pose for debugging.
DEBUG = False


def _wrap_pi(a):
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def swerve_ik(vx, vy, w, cur_steer):
    """Body-frame twist -> per-module (steer angle target, wheel speed).

    Order follows MODULE_ORDER. Returns (steer_targets[4], wheel_speeds[4]).

    Steering is range-limited (no continuous wrap), so for each module we pick
    between heading theta (wheel +) and theta+pi (wheel -), choosing the
    in-limit candidate with the smallest change from the *measured* steer
    angle. This keeps targets near the current angle (no +-pi jitter at the
    limit) and reverses the wheel instead of spinning the module 180 deg.
    At ~zero twist we hold the last steer angle (no snap to 0).
    """
    steer_targets = np.zeros(4, dtype=np.float64)
    wheel_speeds = np.zeros(4, dtype=np.float64)
    near_zero = max(abs(vx), abs(vy), abs(w)) < ZERO_TWIST_EPS

    for i, name in enumerate(MODULE_ORDER):
        if near_zero:
            steer_targets[i] = cur_steer[i]
            wheel_speeds[i] = 0.0
            continue

        lx, ly = MODULE_XY[name]
        vix = vx - w * ly
        viy = vy + w * lx
        speed = math.hypot(vix, viy)
        if speed < ZERO_TWIST_EPS:
            steer_targets[i] = cur_steer[i]
            wheel_speeds[i] = 0.0
            continue

        theta = math.atan2(viy, vix)
        wheel = speed / WHEEL_RADIUS

        best = None  # (delta, abs_target, signed_speed, in_limit)
        for cand_theta, cand_speed in (
            (theta, wheel),
            (_wrap_pi(theta + math.pi), -wheel),
        ):
            d = _wrap_pi(cand_theta - cur_steer[i])
            abs_target = cur_steer[i] + d
            in_limit = abs(abs_target) <= STEER_LIMIT_EFF
            key = (not in_limit, abs(d))  # prefer in-limit, then min travel
            if best is None or key < best[0]:
                best = (key, abs_target, cand_speed)

        abs_target = float(np.clip(best[1], -STEER_LIMIT_EFF, STEER_LIMIT_EFF))
        steer_targets[i] = abs_target
        wheel_speeds[i] = WHEEL_DIR_SIGN * best[2]

    return steer_targets, wheel_speeds


class SwerveBaseController:
    """Commands the 8 G2 wheel DOFs from a body twist.

    The drive joints ship (in robot.usda) as a stiff position drive holding
    0; configure_drive_modes() switches steering->position and drive->velocity
    and must be called once after the articulation view is initialized.
    """

    def __init__(self, robot, articulation_view,
                 av=None, num_envs: int = 1, env_spacing: float = 0.0,
                 robot_facade=None):
        self._robot = robot
        self._view = articulation_view
        # Phase 3: when multi-env is active, _av is the broadcast view
        # across /World/env_*/<robot> and writes go through it. env_offsets
        # places env_i's base at robot_start + (i*env_spacing, 0, 0).
        # robot_facade (RobotBase) exposes broadcast_view() so we can
        # refresh the AV's _physics_view each tick when prim-deletion
        # events tear it down.
        self._av = av
        self._robot_facade = robot_facade
        self._num_envs = int(num_envs)
        if num_envs > 1:
            self._env_offsets = np.array(
                [[i * env_spacing, 0.0, 0.0] for i in range(num_envs)],
                dtype=np.float32,
            )
        else:
            self._env_offsets = None
        self.steer_idx = [robot.get_dof_index(STEER_JOINTS[n]) for n in MODULE_ORDER]
        self.drive_idx = [robot.get_dof_index(DRIVE_JOINTS[n]) for n in MODULE_ORDER]
        self._configured = False
        self._last_twist = np.zeros(3, dtype=np.float64)
        # Kinematic base pose state (world): x, y, yaw; z held constant.
        self._pose = None  # (x, y, yaw), lazy-init from current root pose
        self._z = None

    def _live_view(self):
        """Return the broadcast Articulation if it's healthy; falls back
        to the env_0 single-articulation view. The facade re-attaches
        _physics_view on the AV when prim-deletion events tear it down."""
        if self._robot_facade is not None:
            av = self._robot_facade.broadcast_view()
            if av is not None:
                return av
        return self._view

    def _set_root_pose(self, pos, quat) -> None:
        """Teleport the kinematic base root. Multi-env: broadcast
        env_i to (pos + env_offsets[i], quat). Single-env: vanilla
        Robot.set_world_pose."""
        if (self._robot_facade is not None and self._env_offsets is not None
                and self._num_envs > 1):
            av = self._robot_facade.broadcast_view()
            if av is not None:
                n = self._num_envs
                positions = (np.asarray(pos, dtype=np.float32)
                             + self._env_offsets).astype(np.float32)
                orientations = np.tile(
                    np.asarray(quat, dtype=np.float32), (n, 1))
                av.set_world_poses(positions, orientations)
                return
        self._robot.set_world_pose(position=pos, orientation=quat)

    def configure_drive_modes(self):
        if self._configured:
            return
        view = self._live_view()
        n = self._num_envs if view is not self._view else 1
        steer_idx = np.asarray(self.steer_idx, dtype=np.int32)
        drive_idx = np.asarray(self.drive_idx, dtype=np.int32)

        # set_gains expects (M, K) with M = #articulations in the view.
        view.set_gains(
            kps=np.full((n, 4), STEER_KP, dtype=np.float32),
            kds=np.full((n, 4), STEER_KD, dtype=np.float32),
            joint_indices=steer_idx,
        )
        view.set_gains(
            kps=np.full((n, 4), DRIVE_KP, dtype=np.float32),
            kds=np.full((n, 4), DRIVE_KD, dtype=np.float32),
            joint_indices=drive_idx,
        )
        # switch_dof_control_mode broadcasts to all envs by default.
        for di in self.steer_idx:
            view.switch_dof_control_mode("position", di)
        for di in self.drive_idx:
            view.switch_dof_control_mode("velocity", di)

        try:
            view.set_max_efforts(
                values=np.tile(
                    np.full(8, WHEEL_MAX_EFFORT, dtype=np.float32), (n, 1)
                ) if n > 1 else
                np.full(8, WHEEL_MAX_EFFORT, dtype=np.float32),
                joint_indices=np.asarray(
                    self.steer_idx + self.drive_idx, dtype=np.int32
                ),
            )
        except Exception:
            pass  # non-fatal; gains already set
        self._configured = True

    def read_cur_steer(self, sim_js):
        return np.array(
            [float(sim_js.positions[i]) for i in self.steer_idx], dtype=np.float64
        )

    def _ramp(self, target):
        t = np.asarray(target, dtype=np.float64)
        cur = self._last_twist
        step = np.array([LIN_RAMP, LIN_RAMP, ANG_RAMP], dtype=np.float64)
        delta = np.clip(t - cur, -step, step)
        self._last_twist = cur + delta
        return self._last_twist

    def step_kinematic(self, vx, vy, w, dt, cur_steer):
        """Kinematic base: integrate the body twist and teleport the
        articulation root each step. Steering/drive joints are still driven
        so the wheels visibly steer/spin, but they bear no load (the body is
        positioned, not pushed by wheel-ground contact). This validates the
        swerve IK without the tip-over of physically driving a high-CoM
        humanoid on a small wheel base, and matches the virtual planar base
        that whole-body cuRobo planning will use later.
        """
        raw = (vx, vy, w)
        vx, vy, w = self._ramp(raw)

        if self._pose is None:
            # Start at the robot's authored standing pose (origin, z=0.04 ==
            # the USD's genie translate), so the first set_world_pose matches
            # where the fixed-base robot already stands (no jolt).
            self._pose = [0.0, 0.0, 0.0]
            self._z = BASE_STAND_Z

        x, y, yaw = self._pose
        cy, sy = math.cos(yaw), math.sin(yaw)
        # body-frame twist -> world-frame integration
        x += (vx * cy - vy * sy) * dt
        y += (vx * sy + vy * cy) * dt
        yaw = _wrap_pi(yaw + w * dt)
        self._pose = [x, y, yaw]

        quat = np.array(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            dtype=np.float32,
        )
        pos = np.array([x, y, self._z], dtype=np.float32)
        try:
            self._set_root_pose(pos, quat)
        except Exception as e:
            if DEBUG:
                print(f"[kbase] set root pose failed: {e}")

        # Visual-only wheel steering/spin.
        steer_t, wheel_v = swerve_ik(vx, vy, w, cur_steer)
        self._apply(self.steer_idx, positions=steer_t)
        self._apply(self.drive_idx, velocities=wheel_v)

        if DEBUG and max(abs(raw[0]), abs(raw[1]), abs(raw[2])) > ZERO_TWIST_EPS:
            print(
                f"[kbase] cmd={tuple(round(v,3) for v in raw)} "
                f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                f"steer_deg={np.round(np.degrees(steer_t),1).tolist()}"
            )

    def base_pose(self):
        """Current kinematic base world pose (x, y, z, yaw)."""
        if self._pose is None:
            return 0.0, 0.0, BASE_STAND_Z, 0.0
        x, y, yaw = self._pose
        return x, y, (self._z if self._z is not None else BASE_STAND_Z), yaw

    def world_to_base(self, p_world, q_world):
        """Transform a world pose into the robot base_link frame.

        The kinematic base is a planar (x, y, yaw) transform at height z, so
        base_link world pose = T(x,y,z) . Rz(yaw). Returns (p_base, q_base)
        with quaternions in (w, x, y, z) order (cuRobo / Isaac convention).
        Needed because cuRobo plans in base_link frame but the target cubes
        are read in world frame; once the base drives away from the origin
        the untransformed world target becomes unreachable (IK_FAIL).
        """
        bx, by, bz, yaw = self.base_pose()
        dx = float(p_world[0]) - bx
        dy = float(p_world[1]) - by
        dz = float(p_world[2]) - bz
        c, s = math.cos(-yaw), math.sin(-yaw)  # rotate by -yaw about +Z
        p_base = np.array(
            [c * dx - s * dy, s * dx + c * dy, dz], dtype=np.float64
        )
        # q_base = conj(Rz(yaw)) (x) q_world ; Rz(yaw) = (cos h, 0, 0, sin h)
        h = yaw / 2.0
        bw, bz_q = math.cos(h), math.sin(h)  # base quat (w,0,0,z)
        qw, qx, qy, qz = (
            float(q_world[0]),
            float(q_world[1]),
            float(q_world[2]),
            float(q_world[3]),
        )
        # conj(base) = (bw, 0, 0, -bz_q); Hamilton product conj(base) * q_world
        rw = bw * qw - (-bz_q) * qz
        rx = bw * qx - (-bz_q) * qy
        ry = bw * qy + (-bz_q) * qx
        rz = bw * qz + (-bz_q) * qw
        q_base = np.array([rw, rx, ry, rz], dtype=np.float64)
        q_base /= np.linalg.norm(q_base) + 1e-12
        return p_base, q_base

    def update(self, vx, vy, w, cur_steer):
        raw = (vx, vy, w)
        vx, vy, w = self._ramp(raw)
        steer_t, wheel_v = swerve_ik(vx, vy, w, cur_steer)
        self._apply(self.steer_idx, positions=steer_t)
        self._apply(self.drive_idx, velocities=wheel_v)
        if DEBUG and max(abs(raw[0]), abs(raw[1]), abs(raw[2])) > ZERO_TWIST_EPS:
            print(
                f"[swerve] cmd={tuple(round(x,3) for x in raw)} "
                f"ramped=({vx:.3f},{vy:.3f},{w:.3f}) "
                f"steer_deg={np.round(np.degrees(steer_t),1).tolist()} "
                f"wheel={np.round(wheel_v,2).tolist()} "
                f"steer_idx={self.steer_idx} drive_idx={self.drive_idx}"
            )

    def _apply(self, idx, positions=None, velocities=None):
        view = self._live_view()
        n = self._num_envs if view is not self._view else 1
        idx_arr = np.asarray(idx, dtype=np.int32)
        # Prefer the typed-targets API; fall back to ArticulationActions.
        if positions is not None:
            vals = np.asarray(positions, dtype=np.float32).reshape(-1)
            tiled = np.tile(vals, (n, 1)) if n > 1 else vals.reshape(1, -1)
            setter = getattr(view, "set_joint_position_targets", None)
            if setter is not None:
                setter(tiled, joint_indices=idx_arr)
                return
        if velocities is not None:
            vals = np.asarray(velocities, dtype=np.float32).reshape(-1)
            tiled = np.tile(vals, (n, 1)) if n > 1 else vals.reshape(1, -1)
            setter = getattr(view, "set_joint_velocity_targets", None)
            if setter is not None:
                setter(tiled, joint_indices=idx_arr)
                return
        from omni.isaac.core.utils.types import ArticulationActions

        view.apply_action(
            ArticulationActions(
                joint_positions=(
                    None
                    if positions is None
                    else np.asarray(positions, dtype=np.float32).reshape(1, -1)
                ),
                joint_velocities=(
                    None
                    if velocities is None
                    else np.asarray(velocities, dtype=np.float32).reshape(1, -1)
                ),
                joint_indices=idx_arr,
            )
        )


class KeyboardTeleop:
    """Base teleop via carb keyboard events (needs viewport focus).

    Deliberately avoids W/A/S/D/Q/E because the Isaac viewport uses those for
    fly-camera navigation. Arrow keys drive, ,/. yaw, B toggles teleop so the
    camera keys never fight base control.
    """

    HELP = (
        "[base teleop] Up/Down = +/-x  Left/Right = +/-y (strafe)  "
        ", / . = +/-yaw  B = toggle on/off  H = help"
    )

    def __init__(self):
        import carb.input
        import omni.appwindow

        self._pressed = set()
        self._kbd_input = carb.input
        self._enabled = True
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_kbd
        )
        print(self.HELP)

    def _on_kbd(self, e):
        et = self._kbd_input.KeyboardEventType
        K = self._kbd_input.KeyboardInput
        if e.type == et.KEY_PRESS:
            self._pressed.add(e.input)
            if e.input == K.H:
                print(self.HELP)
            elif e.input == K.B:
                self._enabled = not self._enabled
                print(f"[base teleop] {'ENABLED' if self._enabled else 'DISABLED'}")
        elif e.type == et.KEY_RELEASE:
            self._pressed.discard(e.input)
        return True

    def get_twist(self):
        if not self._enabled:
            return (0.0, 0.0, 0.0)
        K = self._kbd_input.KeyboardInput
        p = self._pressed
        vx = (K.UP in p) - (K.DOWN in p)
        vy = (K.LEFT in p) - (K.RIGHT in p)
        w = (K.COMMA in p) - (K.PERIOD in p)
        return (
            vx * MAX_LIN_SPEED,
            vy * MAX_LIN_SPEED,
            w * MAX_ANG_SPEED,
        )

    def close(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass
