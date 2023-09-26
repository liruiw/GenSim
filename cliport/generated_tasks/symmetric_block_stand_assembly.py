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

class SymmetricBlockStandAssembly(Task):
    """Build two symmetrical structures on two stands using eight blocks of two different colors (four red and four blue)."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "build two symmetrical structures on two stands using eight blocks of two different colors"
        self.task_completed_desc = "done building symmetrical structures."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add stands.
        stand_size = (0.05, 0.15, 0.005)
        stand_urdf = 'stacking/stand.urdf'
        stand1_pose = self.get_random_pose(env, stand_size)
        stand2_pose = self.get_random_pose(env, stand_size)
        env.add_object(stand_urdf, stand1_pose, category='fixed')
        env.add_object(stand_urdf, stand2_pose, category='fixed')

        # Block colors.
        colors = [utils.COLORS['blue'], utils.COLORS['red']]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'

        objs = []
        for i in range(8):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i%2])
            objs.append(block_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs1 = [(utils.apply(stand1_pose, i), stand1_pose[1]) for i in place_pos]
        targs2 = [(utils.apply(stand2_pose, i), stand2_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a symmetrical pattern on both stands.
        language_goal = self.lang_template
        self.add_goal(objs=objs[:4], matches=np.ones((4, 4)), targ_poses=targs1[:4], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*4, language_goal=language_goal)
        self.add_goal(objs=objs[4:], matches=np.ones((4, 4)), targ_poses=targs2[:4], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*4, language_goal=language_goal)