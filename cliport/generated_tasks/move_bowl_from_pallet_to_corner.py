import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class MoveBowlFromPalletToCorner(Task):
    """Place the specific bowl from a pallet to a corner."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the {pick} from pallet to {place} corner."
        self.task_completed_desc = "done placing bowl around corner."
        self.additional_reset()


    def reset(self, env):
        super().reset(env)
        n_pallets = 3
        n_objects = 3
        colors, color_names = utils.get_colors(mode=self.mode, n_colors=n_objects)
        # Add pallets and objects
        # x, y, z dimensions for the asset size
        corner_size = (0.12, 0.12, 0)
        corner_urdf = 'corner/corner-template.urdf'
        corner_poses = []

        pallet_size = (0.06, 0.06, 0)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_poses = []
        objects_ids = []
        bowl_shapes = self.get_kitting_shapes(n_objects)

        for i in range(n_pallets):
            # add pallet
            pallet_pose = self.get_random_pose(env, pallet_size)
            pallet_id = env.add_object(pallet_urdf, pallet_pose, category='fixed', color=colors[i])
            pallet_poses.append(pallet_pose)

            # add kit
            bowl_urdf = 'bowl/bowl.urdf'
            bowl_pose = pallet_pose
            bowl_id = env.add_object(bowl_urdf, bowl_pose, color=colors[i])
            objects_ids.append(bowl_id)

            # add corner
            corner_pose = self.get_random_pose(env, pallet_size)
            corner_id = env.add_object(corner_urdf, corner_pose, category='fixed', color=colors[i])
            corner_poses.append(corner_pose)

        # Goal: put a specific kit from a pallet to the top of a corner
        target_idx = np.random.randint(n_pallets)
        pick_name = color_names[target_idx] + " " + 'bowl'
        language_goal = (self.lang_template.format(pick=pick_name, place=color_names[target_idx]))
        self.add_goal(objs=[objects_ids[target_idx]], matches=np.ones((1, 1)), targ_poses=[corner_poses[target_idx]], replace=False,
            rotations=True, metric='pose', params=None, step_max_reward=1 / n_objects, language_goal=language_goal)
