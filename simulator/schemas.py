import os
import json
from marshmallow import Schema, fields, validate

class InitConditionsSchema(Schema):
    task_name = fields.String(required=True, validate=validate.OneOf(["2choice", "fly_and_turn", "closed_loop_inference"]))
    start_heights = fields.List(fields.Float(required=True))
    target_heights = fields.List(fields.Float(required=True))
    start_dist = fields.Float(required=False)
    theta_offset = fields.Float(required=True)
    theta_environment = fields.Float(required=True)
    # Objects to display (PyBullet coordinates)
    objects_relative = fields.List(fields.Tuple((fields.Float(), fields.Float())), required=True)
    objects_color = fields.List(fields.String(), required=True)
    # Objects to approach (PyBullet coordinates)
    objects_relative_target = fields.List(fields.Tuple((fields.Float(), fields.Float())), required=False)
    objects_color_target = fields.List(fields.String(), required=False)
    correct_side = fields.String(required=False)

class InitConditionsClosedLoopInferenceSchema(InitConditionsSchema):
    task_name = fields.String(required=True, validate=validate.Equal("closed_loop_inference"))
    gs_objects_relative = fields.List(fields.Tuple((fields.Float(), fields.Float())), required=True)
    PYBULLET_TO_GS_SCALING_FACTOR = fields.Float(required=True)

def parse_init_conditions(sim_dir):
    init_conditions_schema = InitConditionsSchema()
    with open(os.path.join(sim_dir, 'init_conditions.json'), 'r') as f:
        init_conditions = json.load(f)
    init_conditions = init_conditions_schema.load(init_conditions)

    return init_conditions