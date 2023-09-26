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

class GuidedBlockPath(Task):
    """Pick up each block and move it along the line of the same color from start to end."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "move the {color} block along the {color} line from start to end"
        self.task_completed_desc = "done moving blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add lines and blocks.
        # x, y, z dimensions for the asset size
        line_size = (0.3, 0.01, 0.01)
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        line_urdf = 'line/line-template.urdf'

        blocks = []
        lines = []
        for i in range(4):
            # Add line
            line_pose = self.get_random_pose(env, line_size)
            env.add_object(line_urdf, line_pose, color=colors[i], category='fixed')
            lines.append(line_pose)

            # Add block
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

            # Add goal
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[line_pose], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=color_names[i]))