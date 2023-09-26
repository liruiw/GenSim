import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class CodeRainbowPyramid(Task):
    """Arrange a cylinder in a zone marked by a green box on the tabletop."""
    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "build a pyramid on a pallet using six blocks of different colors (red, blue, green, yellow, orange, and purple)"
        self.task_completed_desc = "done building pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, category='fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['purple']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03), (0, 0.05, 0.03),
                     (0, -0.05, 0.08), (0, 0, 0.08), (0, 0.05, 0.08)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (bottom row: green, blue, red).
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                language_goal="place the blocks in a pyramid on the pallet "
                              "(bottom row: green, blue, red)")

        # Goal: blocks are stacked in a pyramid (middle row: yellow, orange).
        self.add_goal(objs=blocks[3:5], matches=np.ones((2, 2)), targ_poses=targs[3:5], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*2,
                language_goal="place the blocks in a pyramid on the pallet "
                              "(middle row: yellow, orange)")

        # Goal: blocks are stacked in a pyramid (top row: purple).
        self.add_goal(objs=blocks[5:], matches=np.ones((1, 1)), targ_poses=targs[5:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2],
                language_goal="place the blocks in a pyramid on the pallet "
                              "(top row: purple)")