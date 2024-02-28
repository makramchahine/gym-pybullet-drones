import pybullet as p
import pybullet_data
import os

import shutil
import pybullet_data

object_names = ["sphere2red.urdf", "sphere2blue.urdf", "sphere2green.urdf"]
for object_name in object_names:
    urdf_source_path = f"./gym-pybullet-drones/gym_pybullet_drones/assets/{object_name}"
    urdf_destination_path = os.path.join(pybullet_data.getDataPath(), object_name)

    print(urdf_destination_path)
    shutil.copy(urdf_source_path, urdf_destination_path)

print("Copied assets to pybullet_data folder")