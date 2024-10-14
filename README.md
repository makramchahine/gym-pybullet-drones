# gym-pybullet-drones

This repo is a part of a bigger collection codebase for multienv training and evaluation in simulation. Here we provide utility functions to generate synthetic trajectories for training, and evaluation functions to evaluate the performance of a policy in simulation. This repo is agnostic to the type of policy network. 

Note: This is a fork of the original [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) repository.

To use this repo, you need to add it to your PYTHONPATH. You can do this by running the following command in your terminal from the root directory of this repo:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

```

# Usage

## Generating Synthetic Trajectories
```python
python scripts/generate_synthetic_trajectories.py
```

## Evaluation