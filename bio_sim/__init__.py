"""bio_sim — G2 mobile-manipulation pick-and-place on Isaac Sim 5.1 + cuRobo.

Layering (high -> low):
    play.py            entrypoint, wires everything
    tasks/             ordered skill lists (PickAndPlace)
    skills/            tick state machines (NavigateTo, MoveArmTo, Grasp...)
    robot/             G2Robot facade (arm planner / base / gripper / bridge)
    scene/             declarative environment (ground, table, objects)
    sim/               SimulationApp + World runtime

The low-level swerve IK + cuRobo motion-gen wiring is ported from the
validated curobo_robot/g2_motion_gen_reacher.py; that script stays as the
reference "engine" and is not deleted.
"""
