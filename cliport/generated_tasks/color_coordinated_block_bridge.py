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

class ColorCoordinatedBlockBridge(Task):
    """Construct a bridge by interleaving three differently colored blocks (red, blue, and green) on a pallet in a specific sequence - red block at the edges, blue block in the middle, and a green block on top of the red and blue blocks. Repeat this sequence until a bridge is formed across the length of the pallet."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "construct a bridge by interleaving three differently colored blocks (red, blue, and green) on a pallet in a specific sequence"
        self.task_completed_desc = "done constructing the bridge."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.30, 0.15, 0.02)
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object('pallet/pallet.urdf', pallet_pose, 'fixed')

        # Block colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'

        objs = []
        for i in range(9):  # 3 sets of 3 colored blocks
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i % 3])
            objs.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0, 0.02), (0, 0.05, 0.02),  # bottom layer
                     (0, -0.05, 0.06), (0, 0, 0.06), (0, 0.05, 0.06),  # middle layer
                     (0, -0.05, 0.10), (0, 0, 0.10), (0, 0.05, 0.10)]  # top layer
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a bridge (bottom layer: red, blue, red).
        self.add_goal(objs=objs[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*3,
                      language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (middle layer: green, green, green).
        self.add_goal(objs=objs[3:6], matches=np.ones((3, 3)), targ_poses=targs[3:6], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*3,
                      language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (top layer: red, blue, red).
        self.add_goal(objs=objs[6:], matches=np.ones((3, 3)), targ_poses=targs[6:], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*3,
                      language_goal=self.lang_template) 
