import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class AlignPairColoredBlocksAlongLine(Task):
    """Align two pairs of blocks, each pair painted a different color (red and blue), along a marked line on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Place two pairs of blocks, each pair painted a different color (red and blue), along a marked line on the tabletop."
        self.task_completed_desc = "done aligning blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add line.
        line_size = (0.3, 0.01, 0.01)
        line_pose = self.get_random_pose(env, line_size)
        line_template = 'line/line-template.urdf'
        replace = {'DIM': line_size}
        line_urdf = self.fill_template(line_template, replace)
        env.add_object(line_urdf, line_pose, 'fixed')

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_template = 'block/block-template.urdf'
        colors = [utils.COLORS['red'], utils.COLORS['blue']]
        blocks = []
        anchor_base_poses = [(utils.apply(line_pose, (0.04,  0, 0.001)), line_pose[1]),
                        (utils.apply(line_pose, (0.04 * 2,  0, 0.001)), line_pose[1]),
                        (utils.apply(line_pose, (-0.04,  0, 0.041)), line_pose[1]),
                        (utils.apply(line_pose, (-0.04 * 2, 0, 0.041)), line_pose[1])]

        for color in colors:
            for _ in range(2):
                block_pose = self.get_random_pose(env, block_size)
                replace = {'DIM': block_size}
                block_urdf = self.fill_template(block_template, replace)
                block_id = env.add_object(block_urdf, block_pose, color=color)
                blocks.append(block_id)

        # Goal: each pair of similarly colored blocks are touching and both pairs are aligned along the line.
        self.add_goal(objs=blocks, matches=np.ones((4, 4)), targ_poses=anchor_base_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1,
                language_goal=self.lang_template)