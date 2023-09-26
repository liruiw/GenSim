import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class PlaceBluesAroundRed(Task):
    """Pick up the blue blocks one by one and place them around the red block, forming a circle."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place the blue blocks around the red block"
        self.task_completed_desc = "done placing blue blocks around red block."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add red block.
        red_block_size = (0.04, 0.04, 0.04)
        red_block_urdf = 'block/block_for_anchors.urdf'
        red_block_pose = self.get_random_pose(env, red_block_size)
        red_block_id = env.add_object(red_block_urdf, red_block_pose, 'fixed')

        # Add blue blocks.
        blue_blocks = []
        blue_block_size = (0.02, 0.02, 0.02)
        blue_block_urdf = 'block/block_for_anchors.urdf'
        N = 4

        for _ in range(N):
            blue_block_pose = self.get_random_pose(env, blue_block_size)
            blue_block_id = env.add_object(blue_block_urdf, blue_block_pose, color=utils.COLORS['blue'])
            blue_blocks.append(blue_block_id)

        # Calculate target poses for blue blocks to form a circle around the red block.
        radius = 0.06  # radius of the circle
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)  # angles for each blue block
        targ_poses = []
        for angle in angles:
            x = red_block_pose[0][0] + radius * np.cos(angle)
            y = red_block_pose[0][1] + radius * np.sin(angle)
            z = red_block_pose[0][2]
            targ_poses.append(((x, y, z), red_block_pose[1]))

            # Add goal.
            self.add_goal(objs=blue_blocks, matches=np.eye(N), targ_poses=targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1., language_goal=self.lang_template)
