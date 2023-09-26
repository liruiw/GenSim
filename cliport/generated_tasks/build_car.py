import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class BuildCar(Task):
    """Construct a simple car structure using blocks and cylinders."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "Construct a simple car structure using blocks and cylinders. " \
                             "Firstly, create the base of the car by positioning two red blocks side by side. " \
                             "Then, add the car body by stacking a blue block on top of the base. " \
                             "For the wheels, place a black cylinder on each side of the base blocks."
        self.task_completed_desc = "done building car."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        car_pose = ((0.5, 0.0, 0.0), (0,0,0,1))  # fixed pose
        base_length = 0.04
        self.add_corner_anchor_for_pose(env, car_pose)

        # Add base blocks. Use box template so that we can change its size.
        base_size = (0.02, 0.04, 0.02)
        base_block_urdf = "box/box-template.urdf"
        base_block_urdf = self.fill_template(base_block_urdf,  {'DIM': base_size})
        anchor_base_poses = [(utils.apply(car_pose, (base_length / 2,  base_length / 2, 0.001)), car_pose[1]),
                        (utils.apply(car_pose, (-base_length / 2,  base_length / 2, 0.001)), car_pose[1])]
        base_blocks = []
        
        for idx in range(2):
            base_block_pose = self.get_random_pose(env, base_size)
            base_block_id = env.add_object(base_block_urdf, base_block_pose, color=utils.COLORS['red'])
            base_blocks.append(base_block_id)

        # Add car body block.
        body_size = (0.04, 0.02, 0.02)  # x, y, z dimensions for the asset size
        body_block_urdf = "box/box-template.urdf"
        body_block_urdf = self.fill_template(body_block_urdf,  {'DIM': body_size})
        body_block_pose = self.get_random_pose(env, body_size)
        body_block_id = env.add_object(body_block_urdf, body_block_pose, color=utils.COLORS['blue'])
        anchor_body_poses = [car_pose]

        wheel_length = 0.12
        anchor_wheel_poses = [(utils.apply(car_pose, ( wheel_length / 2,  wheel_length / 2, 0.001)), car_pose[1]),
                              (utils.apply(car_pose, (-wheel_length / 2,  wheel_length / 2, 0.001)), car_pose[1]),
                              (utils.apply(car_pose, ( wheel_length / 2, -wheel_length / 2, 0.001)), car_pose[1]),
                              (utils.apply(car_pose, (-wheel_length / 2, -wheel_length / 2, 0.001)), car_pose[1])]

        # Add wheels.
        wheel_size = (0.02, 0.02, 0.02)  # x, y, z dimensions for the asset size
        wheel_urdf = 'cylinder/cylinder-template.urdf'
        wheel_urdf = self.fill_template(wheel_urdf, {'DIM': wheel_size})

        wheels = []
        for idx in range(4):
            wheel_pose = self.get_random_pose(env, wheel_size)
            wheel_id = env.add_object(wheel_urdf, wheel_pose, color=utils.COLORS['black'])
            wheels.append(wheel_id)

        # Goal: Firstly, create the base of the car by positioning two red blocks side by side.
        self.add_goal(objs=base_blocks,
                      matches=np.ones((2, 2)),
                      targ_poses=anchor_base_poses,
                      replace=False,
                      rotations=True,
                      metric='pose',
                      params=None,
                      step_max_reward=1./3,
                      language_goal="Firstly, create the base of the car by positioning two red blocks side by side.")

        # Then, add the car body by stacking a blue block on top of the base.
        self.add_goal(objs=[body_block_id],
                      matches=np.ones((1, 1)),
                      targ_poses=anchor_body_poses,
                      replace=False,
                      rotations=True,
                      metric='pose',
                      params=None,
                      step_max_reward=1./3,
                      language_goal="Then, add the car body by stacking a blue block on top of the base.")

        # For the wheels, place a black cylinder on each side of the base blocks.
        self.add_goal(objs=wheels,
                      matches=np.ones((4, 4)),
                      targ_poses=anchor_wheel_poses,
                      replace=False,
                      rotations=True,
                      metric='pose',
                      params=None,
                      step_max_reward=1./3,
                      language_goal="For the wheels, place a black cylinder on each side of the base blocks.")

