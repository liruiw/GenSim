import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class PlaceBlueOnLineEnds(Task):
    """Pick up each blue box and accurately place it at the end of a green line."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the blue box at the end of the green line"
        self.task_completed_desc = "done placing blue boxes on line ends."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add lines.
        line_size = (0.3, 0.01, 0.01)
        line_template = 'line/line-template.urdf'
        replace = {'DIM': line_size}
        line_urdf = self.fill_template(line_template, replace)

        line_colors = ['green']
        line_poses = []

        line_pose = self.get_random_pose(env, line_size)
        color = utils.COLORS[line_colors[0]]
        env.add_object(line_urdf, line_pose, 'fixed', color=color)
        line_poses.append(utils.apply(line_pose, (-0.15,0,0)))
        line_poses.append(utils.apply(line_pose, (0.15,0,0)))

        # Add blue boxes.
        box_size = (0.04, 0.04, 0.04)
        box_urdf = 'box/box-template.urdf'
        box_color = utils.COLORS['blue']
        boxes = []
        for _ in range(2):
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=box_color)
            boxes.append(box_id)

        # Goal: each blue box is at the end of a different colored line.
        for i in range(2):
            language_goal = self.lang_template.format(line_colors[0])
            self.add_goal(objs=[boxes[i]], matches=np.ones((1, 1)), targ_poses=[line_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal=language_goal)
