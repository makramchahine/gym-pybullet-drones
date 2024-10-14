from path_templates.schemas import InitConditionsSchema
import numpy as np
import random

def generate_init_conditions_fly_and_turn(object_color) -> InitConditionsSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- Approach and Turn

    """
    max_yaw_offset = 0.175 * np.pi
    if random.random() < 0.75:
        start_heights = [0.1 + random.uniform(0, 1)]
        theta_offset = random.choice([max_yaw_offset, -max_yaw_offset])
    else:
        start_heights = [0.1 + random.choice([0, 1])]
        theta_offset = random.uniform(max_yaw_offset, -max_yaw_offset)

    target_heights = [0.1 + 0.5]
    theta_environment = random.random() * 2 * np.pi
    objects_relative = [(random.uniform(1, 2), 0)]
    
    init_conditions_schema = InitConditionsSchema()
    init_conditions = {
        "task_name": "fly_and_turn",
        "start_heights": start_heights,
        "target_heights": target_heights,
        "start_dist": objects_relative[0][0],
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_relative": objects_relative,
        "objects_color": [object_color],
        "objects_relative_target": objects_relative,
        "objects_color_target": [object_color],
    }
    init_conditions = init_conditions_schema.load(init_conditions)

    return init_conditions

def generate_init_conditions_2choice(object_color) -> InitConditionsSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- Fly to Correct Choice

    """
    max_yaw_offset = 0.1 * np.pi
    if random.random() < 0.75:
        start_heights = [0.1 + random.uniform(0, 1)]
        theta_offset = random.choice([max_yaw_offset, -max_yaw_offset])
    else:
        start_heights = [0.1 + random.choice([0, 1])]
        theta_offset = random.uniform(max_yaw_offset, -max_yaw_offset)

    target_heights = [0.1 + 0.5]
    theta_environment = random.random() * 2 * np.pi
    
    start_dist = random.uniform(1, 2)
    orthogonal_dist = 0.2
    if random.random() < 0.5:
        correct_side = "R"
        objects_relative = [(start_dist, -orthogonal_dist), (start_dist, orthogonal_dist)]
    else:
        correct_side = "L"
        objects_relative = [(start_dist, orthogonal_dist), (start_dist, -orthogonal_dist)]
    
    objects_color = [object_color, "B" if object_color == "R" else "R"]
    objects_color_target = [object_color]
    objects_relative_target = objects_relative[0:1]
    
    init_conditions_schema = InitConditionsSchema()
    init_conditions = {
        "task_name": "2choice",
        "start_heights": start_heights,
        "target_heights": target_heights,
        "start_dist": objects_relative[0][0],
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_relative": objects_relative,
        "objects_color": objects_color,
        "objects_relative_target": objects_relative_target,
        "objects_color_target": objects_color_target,
        "correct_side": correct_side
    }
    init_conditions = init_conditions_schema.load(init_conditions)

    return init_conditions

function_map = {
    "fly_and_turn": generate_init_conditions_fly_and_turn,
    "2choice": generate_init_conditions_2choice
}

def get_init_conditions_func(task_name):
    return function_map[task_name]
