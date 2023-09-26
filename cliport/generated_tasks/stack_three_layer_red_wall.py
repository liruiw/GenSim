import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class StackThreeLayerRedWall(Task):
    """Build a wall by stacking blocks. The wall should consist of three layers with each layer having three red blocks aligned in a straight line."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the red blocks to form a three-layer wall"
        self.task_completed_desc = "done stacking blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks.
        block_size = (0.05, 0.05, 0.03)  # x, y, z dimensions for the block size
        block_urdf = 'block/block_for_anchors.urdf'  # URDF for the block
        block_color = utils.COLORS['red']  # Color for the block

        # We need 9 blocks for a three-layer wall with each layer having three blocks.
        blocks = []
        for _ in range(9):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_color)
            blocks.append(block_id)

        # Define target poses for the blocks to form a three-layer wall.
        # The target poses are defined relative to a base pose.
        base_pose = ((0.5, 0.0, 0.0), (0, 0, 0, 1))
        target_poses = []
        for i in range(3):  # three layers
            for j in range(3):  # three blocks per layer
                target_pos = (j * block_size[0], 0, i * block_size[2])
                target_pose = (utils.apply(base_pose, target_pos), (0, 0, 0, 1))
                target_poses.append(target_pose)

            # Goal: all blocks are stacked to form a three-layer wall.
            self.add_goal(objs=blocks[3*i:3*(i+1)], matches=np.ones((3, 3)), targ_poses=target_poses[3*i:3*(i+1)], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3., language_goal=self.lang_template)
