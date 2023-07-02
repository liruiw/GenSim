"""PyBullet utilities for loading assets."""
import os
import six
import time
import pybullet as p


# BEGIN GOOGLE-EXTERNAL
def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    # Handles most general file open case.
    for _ in range(6):
        try:
            return pybullet_client.loadURDF(file_path, *args, **kwargs)
        except pybullet_client.error as e:
            print("PYBULLET load urdf error!")
            print(e)
            time.sleep(0.1)
    print("missing urdf error. use dummy block.")
    urdf = 'stacking/block.urdf'
    return pybullet_client.loadURDF(urdf, *args, **kwargs)

# END GOOGLE-EXTERNAL
