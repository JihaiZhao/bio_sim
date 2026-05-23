"""bio_sim — mobile-manipulation pick-and-place on Isaac Sim 5.1 + cuRobo.

Layering (high -> low):
    cli.py / __main__  entrypoint: `python -m bio_sim {list,run}`
    specs.py           robot / scene / task registries (metadata only)
    tasks/             ordered skill lists (PickAndPlace)
    skills/            tick state machines (NavigateTo, MoveArmTo, Grasp...)
    robot/             G2Robot / R1ProRobot facades (arm / base / gripper)
    scene/             declarative environment (ground, table, objects)
    sim/               SimulationApp + World runtime

The low-level swerve IK + cuRobo motion-gen wiring is ported from the
validated curobo_robot/g2_motion_gen_reacher.py; that script stays as the
reference "engine" and is not deleted.
"""
