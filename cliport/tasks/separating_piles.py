import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class SeparatingPiles(Task):
    """Sweep the pile of blocks into the specified zone. Each scene contains two square zones: one
relevant to the task, another as a distractor."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "push the pile of {block_color} blocks into the {square_color} square"
        self.task_completed_desc = "done separating pile."
        self.primitive = primitives.push
        self.ee = Spatula
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # sample colors
        (zone1_color, zone2_color, block_color), color_names = utils.get_colors(mode=self.mode, n_colors=3)

        # Add goal zone.
        zone_size = (0.15, 0.15, 0)
        zone1_pose = self.get_random_pose(env, zone_size)
        zone2_pose = self.get_random_pose(env, zone_size)
        while np.linalg.norm(np.array(zone2_pose[0]) - np.array(zone1_pose[0])) < 0.2:
            zone2_pose = self.get_random_pose(env, zone_size)

        zone1_obj_id = env.add_object('zone/zone.urdf', zone1_pose, 'fixed')
        p.changeVisualShape(zone1_obj_id, -1, rgbaColor=zone1_color + [1])
        zone2_obj_id = env.add_object('zone/zone.urdf', zone2_pose, 'fixed')
        p.changeVisualShape(zone2_obj_id, -1, rgbaColor=zone2_color + [1])

        # Choose zone
        zone_target_idx = random.randint(0, 1)
        zone_target = [zone1_pose, zone2_pose][zone_target_idx]
        zone_target_color = [color_names[0], color_names[1]][zone_target_idx]

        # Add pile of small blocks with `make_piles` function
        obj_ids = self.make_piles(env, block_color=block_color)

        # Goal: all small blocks must be in the correct zone.
        language_goal = self.lang_template.format(block_color=color_names[2], square_color=zone_target_color)
        self.add_goal(objs=obj_ids, matches=np.ones((50, 1)), targ_poses=[zone_target], replace=True,
                rotations=False, metric='zone', params=[(zone_target, zone_size)], step_max_reward=1, language_goal=language_goal)