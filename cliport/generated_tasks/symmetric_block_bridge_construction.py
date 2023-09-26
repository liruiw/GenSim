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

class SymmetricBlockBridgeConstruction(Task):
    """Create a symmetrical bridge-shaped structure on a stand using eight blocks of two different colors (four red and four blue)."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "create a symmetrical bridge-shaped structure on a stand using eight blocks of two different colors (four red and four blue)"
        self.task_completed_desc = "done building the bridge."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, category='fixed')

        # Block colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue']]

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'

        objs = []
        for i in range(8):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i%2])
            objs.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13),
                     (0, -0.025, 0.18), (0, 0.025, 0.18)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a bridge (bottom row: red, red).
        self.add_goal(objs=objs[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2]*2,
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (second row: blue).
        self.add_goal(objs=objs[2:3], matches=np.ones((1, 1)), targ_poses=targs[2:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (third row: red).
        self.add_goal(objs=objs[3:4], matches=np.ones((1, 1)), targ_poses=targs[3:4], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (fourth row: blue).
        self.add_goal(objs=objs[4:5], matches=np.ones((1, 1)), targ_poses=targs[4:5], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (fifth row: red).
        self.add_goal(objs=objs[5:6], matches=np.ones((1, 1)), targ_poses=targs[5:6], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (sixth row: blue).
        self.add_goal(objs=objs[6:7], matches=np.ones((1, 1)), targ_poses=targs[6:7], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a bridge (top row: red, red).
        self.add_goal(objs=objs[7:], matches=np.ones((1, 1)), targ_poses=targs[7:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)