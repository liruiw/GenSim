import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p


class PackingBoxesPairs(Task):
    """Tightly pack all the boxes of two specified colors inside the brown box."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "pack all the {colors} blocks into the brown box" # should have called it boxes :(
        self.task_completed_desc = "done packing blocks."

        # Tight z-bound (0.0525) to discourage stuffing everything into the brown box
        self.zone_bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.0525]])
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container box.
        zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        replace = {'DIM': zone_size, 'HALF': (zone_size[0] / 2, zone_size[1] / 2, zone_size[2] / 2)}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')

        margin = 0.01
        min_object_dim = 0.05
        bboxes = []

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = utils.TreeNode(None, [], bbox=np.array(root_size))
        utils.KDTree(root, min_object_dim, margin, bboxes)

        # select colors
        all_colors, all_color_names = utils.get_colors(mode=self.mode)
        selected_idx = np.random.choice(range(len(all_colors)), 2, replace=False)

        relevant_color_names = [c for idx, c in enumerate(all_color_names) if idx in selected_idx]
        distractor_colors = [c for idx, c in enumerate(all_color_names) if idx not in selected_idx]

        pack_colors = [c for idx, c in enumerate(all_colors) if idx in selected_idx]
        distractor_colors = [c for idx, c in enumerate(all_colors) if idx not in selected_idx]

        # Add objects in container.
        object_ids = []
        bboxes = np.array(bboxes)
        object_template = 'box/box-template.urdf'
        for bbox in bboxes:
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            pose = (position, (0, 0, 0, 1))
            pose = utils.multiply(zone_pose, pose)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)

            object_ids.append(box_id)
            icolor = np.random.choice(range(len(pack_colors)), 1).squeeze()
            p.changeVisualShape(box_id, -1, rgbaColor=pack_colors[icolor] + [1])

        # Randomly select object in box and save ground truth pose.
        object_volumes = []
        true_poses = []
        for object_id in object_ids:
            true_pose = p.getBasePositionAndOrientation(object_id)
            object_size = p.getVisualShapeData(object_id)[0][3]
            object_volumes.append(np.prod(np.array(object_size) * 100))
            pose = self.get_random_pose(env, object_size)
            p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
            true_poses.append(true_pose)

        # Add distractor objects
        num_distractor_objects = 4
        distractor_bbox_idxs = np.random.choice(len(bboxes), num_distractor_objects)
        for bbox_idx in distractor_bbox_idxs:
            bbox = bboxes[bbox_idx]
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2

            pose = self.get_random_pose(env, size)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)

            icolor = np.random.choice(range(len(distractor_colors)), 1).squeeze()
            if box_id:
                p.changeVisualShape(box_id, -1, rgbaColor=distractor_colors[icolor] + [1])

        # Some scenes might contain just one relevant block that fits in the box.
        if len(relevant_color_names) > 1:
            relevant_desc = f'{relevant_color_names[0]} and {relevant_color_names[1]}'
        else:
            relevant_desc = f'{relevant_color_names[0]}'

        # IMPORTANT: Specify (obj_pts, [(zone_pose, zone_size)]) for target `zone`. obj_pts is a dict
        language_goal = self.lang_template.format(colors=relevant_desc)
        self.add_goal(objs=object_ids, matches=np.eye(len(object_ids)), targ_poses=true_poses, replace=False,
                rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1, language_goal=language_goal)