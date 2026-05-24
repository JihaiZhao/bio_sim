#
# SkillRunner: advance an ordered list of skills, one at a time, one tick per
# sim step. This replaces the reactive "which cube moved -> track it" logic of
# the old reacher with an explicit, inspectable sequence.
#

from __future__ import annotations

from typing import List

from .skills.skill import Skill, SkillContext, Status


class SkillRunner:
    def __init__(self, skills: List[Skill]):
        self._skills = list(skills)
        self._i = 0
        self._started = False
        self.done = False
        self.failed = False

    def restart(self) -> None:
        """Re-run the whole task from the first skill (the keyboard env-reset
        calls this so the user can replay without relaunching)."""
        self._i = 0
        self._started = False
        self.done = False
        self.failed = False
        print("[runner] restart -> task will re-run from the start")

    @property
    def current(self) -> Skill | None:
        if 0 <= self._i < len(self._skills):
            return self._skills[self._i]
        return None

    @property
    def index(self) -> int:
        return self._i

    @property
    def total(self) -> int:
        return len(self._skills)

    def tick(self, ctx: SkillContext) -> None:
        if self.done:
            return
        skill = self.current
        if skill is None:
            self.done = True
            print("[runner] all skills complete")
            return

        if not self._started:
            print(f"[runner] -> {skill.name} ({self._i + 1}/{len(self._skills)})")
            skill.start(ctx)
            self._started = True

        status = skill.update(ctx)

        if status is Status.SUCCESS:
            print(f"[runner]    {skill.name}: SUCCESS")
            self._i += 1
            self._started = False
        elif status is Status.FAILURE:
            print(f"[runner]    {skill.name}: FAILURE -> aborting task")
            self.failed = True
            self.done = True
