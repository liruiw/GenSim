import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils


class SweepingPiles(Task):
    """Push piles of small objects into a target goal zone marked on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "push the pile of blocks into the green square"
        self.task_completed_desc = "done sweeping."
        self.primitive = primitives.push
        self.ee = Spatula
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add goal zone.
        zone_size = (0.12, 0.12, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        env.add_object('zone/zone.urdf', zone_pose, 'fixed')

        # Add pile of small blocks with `make_piles` function
        obj_ids = self.make_piles(env)

        # Add goal
        self.add_goal(objs=obj_ids, matches=np.ones((50, 1)), targ_poses=[zone_pose], replace=True,
                rotations=False, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1, language_goal=self.lang_template)