import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p

class ConstructCornerBuilding(Task):
    """Construct a building-like structure by placing four blocks of different colors 
    at each corner of a square and one block at the center. 
    Starting from the center, each block should be placed in a clockwise direction 
    in the following order: red, green, blue, orange, and yellow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "construct a building-like structure by placing five blocks of different colors at each corner of a square and one block at the center. Starting from the center, each block should be placed in a clockwise direction in the following order: red, green, blue, orange, and yellow."
        self.task_completed_desc = "done constructing corner building."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block colors
        colors = [utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['blue'], utils.COLORS['orange'], utils.COLORS['yellow']]

        # Define block size
        block_size = (0.04, 0.04, 0.04)

        # Define block urdf
        block_urdf = 'block/block.urdf'

        # Add blocks
        body_block_urdf = "box/box-template.urdf"
        body_block_urdf = self.fill_template(body_block_urdf,  {'DIM': (0.10, 0.10, 0.04)})
        block_pose = self.get_random_pose(env, block_size)
        block_id = env.add_object(body_block_urdf, block_pose, color=colors[0])
        blocks = [block_id]

        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i+1])
            blocks.append(block_id)

        # Define target positions for blocks
        center_pos = (0.5, 0, 0.02)
        corner_positions = [(0.55, 0.05, 0.02), (0.45, 0.05, 0.02), (0.45, -0.05, 0.02), (0.55, -0.05, 0.02)]
        target_positions = [center_pos] + corner_positions

        # Define target poses for blocks
        target_poses = [(pos, p.getQuaternionFromEuler((0, 0, 0))) for pos in target_positions]

        # Add goals
        for i in range(1,5):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[target_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/6, language_goal=self.lang_template)
        self.add_goal(objs=[blocks[0]], matches=np.ones((1, 1)), targ_poses=[target_poses[0]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/6, language_goal=self.lang_template)