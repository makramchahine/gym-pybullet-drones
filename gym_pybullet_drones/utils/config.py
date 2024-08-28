import numpy as np

OBJECT_MAP = {
    "red ball": {
        "ply_path": "./output/solid_red_ball/point_cloud/iteration_30000/point_cloud.ply",
        "urdf_path": "sphere2red.urdf",
        "scale": 0.2
    },

    "blue ball": {
        "ply_path": "./output/solid_blue_ball/point_cloud/iteration_30000/point_cloud.ply",
        "urdf_path": "sphere2blue.urdf",
        "scale": 0.2
    },
    "green ball": {
        "urdf_path": "sphere2green.urdf",
        "scale": 0.2
    },
    "yellow ball": {
        "urdf_path": "sphere2yellow.urdf",
        "scale": 0.2
    },
    "purple ball": {
        "urdf_path": "sphere2purple.urdf",
        "scale": 0.2
    },
    "red cube": {
        "urdf_path": "cube2red.urdf",
        "scale": 0.2
    },
    "blue cube": {
        "urdf_path": "cube2blue.urdf",
        "scale": 0.2
    },
    "yellow cube": {
        "urdf_path": "cube2yellow.urdf",
        "scale": 0.2
    },
    "purple cube": {
        "urdf_path": "cube2purple.urdf",
        "scale": 0.2
    },
    "green cube": {
        "urdf_path": "cube2green.urdf",
        "scale": 0.2
    },
    "red donut": {
        "urdf_path": "torus_red.urdf",
        "scale": 0.3
    },
    "red pyramid": {
        "urdf_path": "pyramid_red.urdf",
        "orientation": [np.pi / 2, 0, 0],
        "scale": 0.2
    },
    "blue pyramid": {
        "urdf_path": "pyramid_blue.urdf",
        "orientation": [np.pi / 2, 0, 0],
        "scale": 0.2
    },
    "yellow pyramid": {
        "urdf_path": "pyramid_yellow.urdf",
        "orientation": [np.pi / 2, 0, 0],
        "scale": 0.2
    },
    "purple pyramid": {
        "urdf_path": "pyramid_purple.urdf",
        "orientation": [np.pi / 2, 0, 0],
        "scale": 0.2
    },
    "green pyramid": {
        "urdf_path": "pyramid_green.urdf",
        "orientation": [np.pi / 2, 0, 0],
        "scale": 0.2
    },
    "jeep": {
        "urdf_path": "jeep.urdf",
        "scale": 0.2 / 1.5,
        "orientation": [np.pi / 2, 0, 0]
    },
    "horse": {
        "urdf_path": "horse.urdf",
        "scale": 1 / 5000
    },
    "dog": {
        "urdf_path": "dog.urdf",
        "scale": 1 / 100
    },
    "palmtree": {
        "urdf_path": "palmtree.urdf",
        "scale": 1 / 2000
    },
    "watermelon": {
        "urdf_path": "watermelon.urdf",
        "scale": 1 / 100
    },
    "traffic_light": {
        "urdf_path": "traffic_light.urdf",
        "scale": 1 / 100,
        "orientation": [np.pi / 2, 0, 0]
    },
    "robot": {
        "urdf_path": "robot.urdf",
        "scale": 1 / 100,
    },
    "rocket": {
        "urdf_path": "rocket.urdf",
        "scale": 1 / 1000,
    },
    "pikachu": {
        "urdf_path": "pikachu.urdf",
        "scale": 1 / 10,
    },
    "house_interior": {
        "urdf_path": "house_interior.urdf",
        "scale": 1 / 1000,
    }
}

URDF_MAP = {key: value["urdf_path"] for key, value in OBJECT_MAP.items()}
ORIENTATION_MAP = {key: value.get("orientation", [0, 0, 0]) for key, value in OBJECT_MAP.items()}
SCALING_MAP = {key: value["scale"] for key, value in OBJECT_MAP.items()}