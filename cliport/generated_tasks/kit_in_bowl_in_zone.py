import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import os

class KitInBowlInZone(Task):
    """Pick up each kit and place it on the corresponding colored bowl, which are located in specific positions on a zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place the {} bowl on the zone"
        self.lang_template_2 = "place the {} on the {} bowl"

        self.task_completed_desc = "done placing kits on bowls and bowl on zone."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zone.
        zone_size = (0.2, 0.2, 0.01)
        zone_pose = self.get_random_pose(env, zone_size)
        zone_urdf = 'zone/zone.urdf'
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Define colors.
        kit_colors = ['red']
        bowl_colors = ['blue']

        # Add bowls.
        bowl_size = (0.04, 0.04, 0.06)
        bowls = []
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_pose = self.get_random_pose(env, bowl_size)
        bowl_id = env.add_object(bowl_urdf, bowl_pose)
        bowls.append(bowl_id)

        # Add kits.
        kit_size = utils.map_kit_scale((0.03, 0.03, 0.02))
        obj_shapes = self.get_kitting_shapes(1)
        shape = os.path.join(self.assets_root, 'kitting',
                                 f'{obj_shapes[0]:02d}.obj')
        template = 'kitting/object-template.urdf'
        replace = {'FNAME': (shape,), 'SCALE': kit_size, 'COLOR': kit_colors[0]}

        # IMPORTANT: REPLACE THE TEMPLATE URDF
        kit_urdf = self.fill_template(template, replace)
        kits = []
        kit_pose = self.get_random_pose(env, kit_size)
        kit_id = env.add_object(kit_urdf, kit_pose, color=bowl_colors[0])
        kits.append(kit_id)

        # Goal: place the bowl on top of the zone
        self.add_goal(objs=[bowls[0]], matches=np.ones((1, 1)), targ_poses=[zone_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/2, language_goal=self.lang_template.format(bowl_colors[0]))


        # Goal: place the kit on top of the bowl
        pick_name = kit_colors[0] + " " + utils.assembling_kit_shapes[obj_shapes[0]]
        language_goal = self.lang_template_2.format(pick_name, bowl_colors[0])
        self.add_goal(objs=[kits[0]], matches=np.ones((1, 1)), targ_poses=[zone_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/2, language_goal=language_goal)
