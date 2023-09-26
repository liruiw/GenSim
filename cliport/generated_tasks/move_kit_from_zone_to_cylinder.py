import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class MoveKitFromZoneToCylinder(Task):
    """Place the specific kit from a zone to a cylinder."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the {pick} from zone to {place} cylinder."
        self.task_completed_desc = "done placing kit in zones."
        self.additional_reset()


    def reset(self, env):
        super().reset(env)
        n_zones = 3
        n_objects = 3
        colors, color_names = utils.get_colors(mode=self.mode, n_colors=n_objects)

        # Add zones and objects
        # x, y, z dimensions for the asset size
        cylinder_size = (0.12, 0.12, 0)
        cylinder_template = 'cylinder/cylinder-template.urdf'
        cylinder_poses = []

        zone_size = (0.06, 0.06, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        objects_ids = []
        obj_shapes = self.get_kitting_shapes(n_objects)

        for i in range(n_zones):
            # add zone
            zone_pose = self.get_random_pose(env, zone_size)
            zone_id = env.add_object(zone_urdf, zone_pose, category='fixed', color=colors[i])
            zone_poses.append(zone_pose)

            # add kit
            scale = utils.map_kit_scale((0.03, 0.03, 0.02))
            shape = os.path.join(self.assets_root, 'kitting',
                                     f'{obj_shapes[i]:02d}.obj')
            template = 'kitting/object-template.urdf'
            replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': colors[i]}

            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(template, replace)
            obj_pose = zone_pose
            obj_id = env.add_object(urdf, obj_pose)
            objects_ids.append(obj_id)

            # add cylinder
            cylinder_pose = self.get_random_pose(env, zone_size)
            template = 'kitting/object-template.urdf'
            replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': colors[i]}
            cylinder_urdf = self.fill_template(cylinder_template, replace)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, category='fixed', color=colors[i])
            cylinder_poses.append(cylinder_pose)

        # Goal: put a specific kit from a zone to the top of a cylinder
        target_idx = np.random.randint(n_zones)
        pick_name = color_names[target_idx] + " " + utils.assembling_kit_shapes[obj_shapes[target_idx]]
        language_goal = (self.lang_template.format(pick=pick_name, place=color_names[target_idx]))
        self.add_goal(objs=[objects_ids[target_idx]], matches=np.ones((1, 1)), targ_poses=[cylinder_poses[target_idx]], replace=False,
            rotations=True, metric='pose', params=None, step_max_reward=1 / n_objects, language_goal=language_goal)
