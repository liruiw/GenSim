import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p


class PlaceRedInGreen(Task):
    """pick up the red blocks and place them into the green bowls amidst other objects."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the red blocks in a green bowl"
        self.task_completed_desc = "done placing blocks in bowls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 4)
        n_blocks = np.random.randint(1, n_bowls + 1)

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, obj_size=bowl_size)
            env.add_object(urdf=bowl_urdf, pose=bowl_pose, category='fixed')
            bowl_poses.append(bowl_pose)

        # Add blocks.
        # x, y, z dimensions for the asset size
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object(block_urdf, block_pose)
            blocks.append(block_id)

        # Goal: each red block is in a different green bowl.
        self.add_goal(objs=blocks, matches=np.ones((len(blocks), len(bowl_poses))), targ_poses=bowl_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)

        # Colors of distractor objects.
        bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
        block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

        # Add distractors.
        n_distractors = 0
        while n_distractors < 6:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = block_colors if is_block else bowl_colors
            pose = self.get_random_pose(env, obj_size=size)
            color = colors[n_distractors % len(colors)]

            obj_id = env.add_object(urdf, pose, color=color)
            n_distractors += 1
