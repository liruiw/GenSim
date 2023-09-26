import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import IPython

class PushPilesIntoLetter(Task):
    """Push piles of small objects into a target goal zone shaped in some letters."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "push the pile of blocks fill in the green shape"
        self.task_completed_desc = "done sweeping."
        self.primitive = primitives.push
        self.ee = Spatula
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        num_blocks = 50

        # Add the target letter
        rand_letter = self.get_kitting_shapes(n_objects=1)[0]
        shape = os.path.join(self.assets_root, 'kitting',
                                 f'{rand_letter:02d}.obj')
        zone_pose = self.get_random_pose(env, (0.2,0.2,0.01))
        scale = [0.006, 0.006, 0.00001]  # .0005
        replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': [0.2, 0.5, 0.2]} # green color
        template = 'kitting/object-template-nocollision.urdf'
        urdf = self.fill_template(template, replace)
        letter_zone = env.add_object(urdf, zone_pose, 'fixed')

        # Add pile of small blocks with `make_piles` function
        obj_ids = self.make_piles(env)

        # Sample point from the object shape as the target poses for the piles
        target_poses = self.get_target_sample_surface_points(shape, scale, zone_pose, num_points=num_blocks)

        # Add goal
        self.add_goal(objs=obj_ids, matches=np.ones((num_blocks, num_blocks)), targ_poses=target_poses, replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=2,
                          language_goal=self.lang_template)