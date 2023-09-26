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

class InsertBlocksIntoPyramid(Task):
    """Insert blocks into a pyramid structure according to their color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "insert the {color} block into the {color} level of the pyramid"
        self.task_completed_desc = "done inserting blocks into pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pyramid.
        pyramid_size = (0.15, 0.15, 0.15)
        pyramid_urdf = 'corner/corner-template.urdf'
        pyramid_pose = self.get_random_pose(env, pyramid_size)
        env.add_object(pyramid_urdf, pyramid_pose, category='fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'

        objs = []
        for i in range(3):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            objs.append(block_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03), (0, 0.05, 0.03)]
        targs = [(utils.apply(pyramid_pose, i), pyramid_pose[1]) for i in place_pos]

        # Goal: blocks are inserted into the pyramid (bottom: red, middle: blue, top: green).
        for i, color in enumerate(['red', 'blue', 'green']):
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[objs[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*1, language_goal=language_goal)