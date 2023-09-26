import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class RainbowStack(Task):
    """Pick up blocks of seven different colors and stack them on the stand in the order of the rainbow (red, orange, yellow, green, blue, indigo, violet) from bottom to top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack the blocks on the stand in the order of the rainbow from bottom to top"
        self.task_completed_desc = "done stacking."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add stand.
        # x, y, z dimensions for the asset size
        stand_size = (0.12, 0.12, 0.02)
        stand_pose = self.get_random_pose(env, stand_size)
        stand_urdf = 'stacking/stand.urdf'
        env.add_object(stand_urdf, stand_pose, 'fixed')

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        blocks = []
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Goal: stack the blocks on the stand in the order of the rainbow from bottom to top.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[stand_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks), language_goal=self.lang_template)