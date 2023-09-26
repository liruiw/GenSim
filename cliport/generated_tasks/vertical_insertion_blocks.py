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

class VerticalInsertionBlocks(Task):
    """Pick up four color specific blocks and insert each block into four differently colored stands set upright on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} block into the {color} stand"
        self.task_completed_desc = "done inserting blocks into stands."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for blocks and stands
        colors = ['red', 'blue', 'green', 'yellow']

        # Add stands.
        # x, y, z dimensions for the asset size
        stand_size = (0.04, 0.04, 0.1)
        stand_urdf = 'stacking/stand.urdf'
        stands = []
        for color in colors:
            stand_pose = self.get_random_pose(env, stand_size)
            stand_id = env.add_object(stand_urdf, stand_pose, color=utils.COLORS[color], category='fixed')
            stands.append(stand_id)

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        blocks = []
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is inserted into the stand of the same color.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(stands[i])], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/len(blocks),
                language_goal=self.lang_template.format(color=colors[i]))