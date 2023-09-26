import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class ColorCoordinatedArchConstruction(Task):
    """Construct an arch using six blocks: three red and three blue."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "construct an arch using six blocks: three red and three blue"
        self.task_completed_desc = "done constructing arch."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.005)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, category='fixed')

        # Block colors.
        colors = [utils.COLORS['red'], utils.COLORS['red'], utils.COLORS['blue'],
                  utils.COLORS['red'], utils.COLORS['red'], utils.COLORS['blue']]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0.05, 0.02),
                     (0, 0, 0.06), (0, -0.05, 0.08),
                     (0, 0.05, 0.08), (0, 0, 0.12)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in an arch (bottom row: red, red, blue).
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                language_goal=self.lang_template)

        # Goal: blocks are stacked in an arch (top row: red, red, blue).
        self.add_goal(objs=blocks[3:], matches=np.ones((3, 3)), targ_poses=targs[3:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                language_goal=self.lang_template)