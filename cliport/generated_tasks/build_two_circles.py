import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class BuildTwoCircles(Task):
    """Construct two distinct circles on the tabletop using 10 red and 10 blue blocks.
    Each circle should consist of blocks of the same color, with the blue circle larger and surrounding the red circle."""

    def __init__(self):
        super().__init__()
        self.max_steps = 30
        self.lang_template = "construct two distinct circles on the tabletop using 6 red and 6 blue blocks"
        self.task_completed_desc = "done building two circles."

    def reset(self, env):
        super().reset(env)

        # Add blocks.
        block_urdf = 'block/block.urdf'
        block_size = (0.04, 0.04, 0.04)

        # Add 6 red blocks.
        red_blocks = []
        red_circle_poses = []
        circle_radius = 0.1
        circle_center = (0, 0, block_size[2] / 2)
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        circle_pose = ((0.4, 0.3, 0.0), (0, 0, 0, 1))  # fixed pose
        self.add_corner_anchor_for_pose(env, circle_pose)

        # Define initial and target poses for the red and blue circles.
        for angle in angles:
            pos =  (circle_center[0] + circle_radius * np.cos(angle),
                    circle_center[1] + circle_radius * np.sin(angle),
                    circle_center[2])
            block_pose = (utils.apply(circle_pose, pos), circle_pose[1])
            block_id = env.add_object(block_urdf, self.get_random_pose(env, block_size), color=utils.COLORS['red'])
            red_circle_poses.append(block_pose)
            red_blocks.append(block_id)

        # Add 6 blue blocks.
        blue_blocks = []
        blue_circle_poses = []
        circle_radius = 0.1
        circle_center = (0, 0, block_size[2] / 2)
        circle_pose = ((0.4, -0.3, 0.0), (0,0,0,1))  # fixed pose
        self.add_corner_anchor_for_pose(env, circle_pose)

        for angle in angles:
            pos =  (circle_center[0] + circle_radius * np.cos(angle),
                    circle_center[1] + circle_radius * np.sin(angle),
                    circle_center[2])
            block_pose = (utils.apply(circle_pose, pos), circle_pose[1])
            block_id = env.add_object(block_urdf, self.get_random_pose(env, block_size), color=utils.COLORS['blue'])
            blue_circle_poses.append(block_pose)
            blue_blocks.append(block_id)


        # Goal: each red block is in the red circle, each blue block is in the blue circle.
        self.add_goal(objs=red_blocks, matches=np.ones((6, 6)), targ_poses=red_circle_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal=self.lang_template)
        self.add_goal(objs=blue_blocks, matches=np.ones((6, 6)), targ_poses=blue_circle_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal=self.lang_template)