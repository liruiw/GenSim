import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os


class PutKitInBowl(Task):
    """Place the specific kit in a bowl of specified color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the {pick} in a {place} bowl"
        self.task_completed_desc = "done placing kit in bowls."
        self.additional_reset()


    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 4)
        n_objects = np.random.randint(1, n_bowls + 1)
        colors, selected_color_names = utils.get_colors(mode=self.mode, n_colors=2)
        block_urdf = 'stacking/block.urdf'
        block_size = (0.04, 0.04, 0.04)

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, category='fixed', color=colors[1])
            bowl_poses.append(bowl_pose)

        # Add kits.
        objects_ids = []
        obj_shapes = self.get_kitting_shapes(n_objects)

        for i in range(n_objects):
            scale = utils.map_kit_scale((0.03, 0.03, 0.02))
            shape = os.path.join(self.assets_root, 'kitting',
                                     f'{obj_shapes[i]:02d}.obj')
            template = 'kitting/object-template.urdf'
            replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': colors[0]}

            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(template, replace)
            obj_pose = self.get_random_pose(env, block_size)
            obj_id = env.add_object(urdf, obj_pose)
            objects_ids.append(obj_id)

            # Goal: put each block in a different bowl.
            pick_name = selected_color_names[0] + " " + utils.assembling_kit_shapes[obj_shapes[i]]
            language_goal = (self.lang_template.format(pick=pick_name, place=selected_color_names[1]))
            self.add_goal(objs=[obj_id], matches=np.ones((1, 1)), targ_poses=bowl_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / n_objects, language_goal=language_goal)

        # Only one mistake allowed.
        self.max_steps = len(objects_ids) + 1

        # Colors of distractor objects.
        distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            pose = self.get_random_pose(env, size)
            color = colors[n_distractors % len(colors)]

            obj_id = env.add_object(urdf, pose, color=color)
            n_distractors += 1