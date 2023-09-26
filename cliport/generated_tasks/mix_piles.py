import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula

class MixPiles(Task):
    """Create two separate piles of ten blocks with different colors. Then, push them into a zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.num_blocks = 10
        self.lang_template = "create two separate piles of ten blocks with different colors. Then, push them into a zone."
        self.task_completed_desc = "done mixing piles."
        self.ee = Spatula
        self.primitive = primitives.push
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add goal zone.
        zone_size = (0.12, 0.12, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        env.add_object('zone/zone.urdf', zone_pose, 'fixed')

        # Get two random colors of piles
        sample_colors, _ = utils.get_colors(self.mode, n_colors=2)

        # Add piles 1.
        piles1 = self.make_piles(env, block_color=sample_colors[0])

        # Add piles 2.
        piles2 = self.make_piles(env, block_color=sample_colors[1])

        # Goal: each block is in the goal zone, alternating between red and blue.
        blocks = piles1 + piles2
        matches = np.ones((len(blocks), 1))
        self.add_goal(objs=blocks, matches=matches, targ_poses=[zone_pose], replace=True,
                      rotations=False, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1,
                          language_goal=self.lang_template)
