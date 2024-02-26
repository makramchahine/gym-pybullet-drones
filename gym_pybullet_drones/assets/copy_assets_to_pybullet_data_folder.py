import pybullet as p
import pybullet_data
import os

import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

