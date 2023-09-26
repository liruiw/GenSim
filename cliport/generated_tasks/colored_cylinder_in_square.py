import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class ColoredCylinderInSquare(Task):
    """Pick up five differently colored cylinder blocks and arrange them inside the square template on the tabletop. Each block should be placed along the corresponding color edge: red, blue, green, yellow, and orange."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "arrange the {color} cylinder along the {color} edge"
        self.task_completed_desc = "done arranging cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add square template.
        square_size = (0.3, 0.3, 0.005)  # x, y, z dimensions for the asset size
        square_pose = self.get_random_pose(env, square_size)
        square_urdf = 'square/square-template.urdf'
        env.add_object(square_urdf, square_pose, 'fixed')

        # Cylinder colors.
        colors = ['red', 'blue', 'green', 'yellow', 'orange']

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.08)  # x, y, z dimensions for the asset size
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for color in colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Associate placement locations for goals.
        place_pos = [(0.1, 0, 0.04), (-0.1, 0, 0.04), (0, 0.1, 0.04), (0, -0.1, 0.04), (0, 0, 0.04)]
        targs = [(utils.apply(square_pose, i), square_pose[1]) for i in place_pos]

        # Goal: each cylinder is placed along the corresponding color edge.
        for i, cylinder in enumerate(cylinders):
            self.add_goal(objs=[cylinder], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 5,
                          language_goal=self.lang_template.format(color=colors[i]))
