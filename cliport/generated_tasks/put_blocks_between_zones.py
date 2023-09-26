import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import copy

class PutBlocksBetweenZones(Task):
    """Arrange four differently colored blocks (red, blue, green, and yellow) between two designated zones on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Arrange the blocks between the zones in the order: red, blue, green, yellow"
        self.task_completed_desc = "done arranging blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone1_pose = self.get_random_pose(env, zone_size)
        zone2_pose = copy.deepcopy(zone1_pose)
        zone2_pose = (utils.apply(zone1_pose, (0,0.1,0)), zone2_pose[1])
        env.add_object(zone_urdf, zone1_pose, 'fixed')
        env.add_object(zone_urdf, zone2_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Goal: blocks are arranged between the zones in the order: red, blue, green, yellow.
        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, 0.1, 0.03)]
        targs = [(utils.apply(zone1_pose, i), zone1_pose[1]) for i in place_pos]

        # Add goal
        self.add_goal(objs=blocks, matches=np.ones((4, 4)), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, symmetries=[np.pi/2]*4, language_goal=self.lang_template)
