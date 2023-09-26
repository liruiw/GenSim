import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class BlockOnCylinderOnPallet(Task):
    """Pick up each block and place it on the corresponding colored cylinder, which are located in specific positions on a pallet."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place the {} cylinder on the pallet"
        self.lang_template_2 = "place the {} block on the {} cylinder"

        self.task_completed_desc = "done placing blocks on cylinders and cylinder on pallet."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.35, 0.35, 0.01)
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Define colors.
        block_colors = ['red']
        cylinder_colors = ['blue']

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.06)
        cylinder_template = 'cylinder/cylinder-template.urdf'
        cylinders = []


        replace = {'DIM': cylinder_size, 'HALF': (cylinder_size[0] / 2, cylinder_size[1] / 2, cylinder_size[2] / 2), 'COLOR': block_colors[0]}
        cylinder_urdf = self.fill_template(cylinder_template, replace)
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)
        cylinders.append(cylinder_id)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        block_pose = self.get_random_pose(env, block_size)
        block_id = env.add_object(block_urdf, block_pose, color=cylinder_colors[0])
        blocks.append(block_id)

        # Goal: place the cylinder on top of the pallet
        self.add_goal(objs=[cylinders[0]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/2, language_goal=self.lang_template.format(cylinder_colors[0]))


        # Goal: place the block on top of the cylinder
        language_goal = self.lang_template_2.format(block_colors[0], cylinder_colors[0])
        self.add_goal(objs=[blocks[0]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/2, language_goal=language_goal)
