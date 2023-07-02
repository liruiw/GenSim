import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class InsertBlocksIntoFixture(Task):
    """Pick up four colored blocks (red, blue, green) and insert them one-by-one into a fixture on the tabletop. The order of insertion starts with the red block, followed by the blue, and green."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} block into the fixture"
        self.task_completed_desc = "done inserting blocks into fixture."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add fixture.
        # x, y, z dimensions for the asset size
        fixture_size = (0.15, 0.15, 0.05)
        fixture_pose = self.get_random_pose(env, fixture_size)
        fixture_urdf = 'insertion/fixture.urdf'
        env.add_object(fixture_urdf, fixture_pose, 'fixed')

        anchor_base_poses = [fixture_pose,
                            (utils.apply(fixture_pose, (0.04,  0, 0.001)), fixture_pose[1]),
                            (utils.apply(fixture_pose, (0,  0.04, 0.001)), fixture_pose[1])]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = ['red', 'blue', 'green']
        blocks = []
        for color in block_colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Goal: each block is inserted into the fixture in the order of red, blue, green, yellow.
        for i, block in enumerate(blocks):
            self.add_goal(objs=[block], matches=np.ones((1, 1)), targ_poses=[anchor_base_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/4)
            self.lang_goals.append(self.lang_template.format(color=block_colors[i]))
