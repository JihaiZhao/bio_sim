#
# Skill contract.
#
# cuRobo + Isaac share ONE my_world.step() pump: you plan once, then execute
# an interpolated trajectory over many steps. So a skill cannot be a blocking
# call -- it is a state machine ticked once per sim step:
#
#     start(ctx)            called once when the skill becomes active
#     update(ctx) -> Status called every sim step until it returns SUCCESS/FAILURE
#
# A task is just an ordered list of skills advanced by SkillRunner.
#

from __future__ import annotations

from enum import Enum


class Status(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class SkillContext:
    """Handles every skill gets each tick.

    world : bio_sim.sim.app.SimApp   (owns the Isaac World; .step(), dt, etc.)
    robot : bio_sim.robot.g2.G2Robot (facade: .arm .base .gripper, world frame)
    scene : bio_sim.scene.bio_scene.BioScene (object world poses)
    blackboard : free-form dict for cross-skill state (e.g. grasped object)
    """

    def __init__(self, world, robot, scene):
        self.world = world
        self.robot = robot
        self.scene = scene
        self.blackboard: dict = {}


class Skill:
    """Base class. Subclasses override start()/update().

    Keep update() cheap and non-blocking: it runs inside the sim loop. Long
    work (motion planning) is kicked off in start() or on the first update()
    and then polled across subsequent ticks.
    """

    name: str = "skill"

    def start(self, ctx: SkillContext) -> None:  # noqa: D401
        """Called once before the first update()."""

    def update(self, ctx: SkillContext) -> Status:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Skill {self.name}>"
