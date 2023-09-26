import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class StackBlockPyramid(Task):
    """Build a pyramid of colored blocks in a color sequence"""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "make the {row} row with {blocks}"
        self.task_completed_desc = "done stacking block pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, category='fixed')

        # Block colors.
        colors = [
            utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['red']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'

        objs = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            objs.append(block_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (bottom row: green, blue, purple).
        language_goal = self.lang_template.format(blocks="the green, blue and purple blocks", row="bottom")
        self.add_goal(objs=objs[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3, language_goal=language_goal)

        # Goal: blocks are stacked in a pyramid (middle row: yellow, orange).
        language_goal = self.lang_template.format(blocks="the yellow and orange blocks", row="middle")      
        self.add_goal(objs=objs[3:5], matches=np.ones((2, 2)), targ_poses=targs[3:5], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2, language_goal=language_goal)

        # Goal: blocks are stacked in a pyramid (top row: red).
        language_goal = self.lang_template.format(blocks="the red block", row="top")
        self.add_goal(objs=objs[5:], matches=np.ones((1, 1)), targ_poses=targs[5:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2]*1, language_goal=language_goal)