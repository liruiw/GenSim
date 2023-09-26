import os
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p


class PalletizingBoxes(Task):
    """Pick up homogeneous fixed-sized boxes and stack them in transposed layers on the pallet."""

    def __init__(self):
        super().__init__()
        self.max_steps = 30
        self.lang_template = "stack all the boxes on the pallet"
        self.task_completed_desc = "done stacking boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        zone_size = (0.3, 0.25, 0.25)
        zone_urdf = 'pallet/pallet.urdf'
        rotation = utils.eulerXYZ_to_quatXYZW((0, 0, 0))
        zone_pose = ((0.5, 0.25, 0.02), rotation)
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Add stack of boxes on pallet.
        margin = 0.01
        object_ids = []

        # x, y, z dimensions for the asset size
        stack_size = (0.19, 0.19, 0.19)
        box_template = 'box/box-template.urdf'
        stack_dim = np.int32([2, 3, 3])

        box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim
        for z in range(stack_dim[2]):

            # Transpose every layer.
            stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
            box_size[0], box_size[1] = box_size[1], box_size[0]

            # IMPORTANT: Compute object points and store as a dictionary for the `goal`
            for y in range(stack_dim[1]):
                for x in range(stack_dim[0]):
                    position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
                    position[0] += x * margin - stack_size[0] / 2
                    position[1] += y * margin - stack_size[1] / 2
                    position[2] += z * margin + 0.03
                    pose = (position, (0, 0, 0, 1))
                    pose = utils.multiply(zone_pose, pose)

                    # IMPORTANT: REPLACE THE TEMPLATE URDF
                    urdf = self.fill_template(box_template, {'DIM': box_size})
                    box_id = env.add_object(urdf, pose)
                    object_ids.append(box_id)
                    self.color_random_brown(box_id)

        # Randomly select top box on pallet and save ground truth pose.
        targets = []
        self.steps = []
        boxes = object_ids[:] # make copy
        while boxes:
            _, height, object_mask = self.get_true_image(env)
            top = np.argwhere(height > (np.max(height) - 0.03))
            rpixel = top[int(np.floor(np.random.random() * len(top)))]  # y, x
            box_id = int(object_mask[rpixel[0], rpixel[1]])
            if box_id in boxes:
                position, rotation = p.getBasePositionAndOrientation(box_id)
                rposition = np.float32(position) + np.float32([0, -10, 0])
                p.resetBasePositionAndOrientation(box_id, rposition, rotation)
                self.steps.append(box_id)
                targets.append((position, rotation))
                boxes.remove(box_id)

        self.steps.reverse()  # Time-reversed depalletizing.
        self.add_goal(objs=object_ids, matches=np.eye(len(object_ids)), targ_poses=targets, replace=False,
                rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1, language_goal=self.lang_template)
        self.spawn_box()

    def reward(self):
        reward, info = super().reward()
        self.spawn_box()
        return reward, info