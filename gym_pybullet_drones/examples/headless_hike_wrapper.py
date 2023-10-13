
import itertools
import math
import random
import time
from functools import partial
import joblib
from tqdm import tqdm

from headless_hike import run_one

if __name__ == "__main__":
    samples = 300
    OBJECTS = ["R", "B"]
    TOTAL_OBJECTS = OBJECTS
    NUM_INITIALIZATIONS = samples // 2

    TOTAL_OBJECTS = OBJECTS * NUM_INITIALIZATIONS
    LOCATIONS_REL = [[(random.uniform(1, 2), 0)] for _ in range(len(TOTAL_OBJECTS))]


    total_list = []
    for i, (obj, loc) in enumerate(zip(TOTAL_OBJECTS, LOCATIONS_REL)):
        total_list.append((obj, loc))
    assert len(total_list) == NUM_INITIALIZATIONS * (2), f"len(total_list): {len(total_list)}"
    random.shuffle(total_list)

    # run_func = partial(run, **vars(ARGS))

    futures = []
    returns = []
    joblib.Parallel(n_jobs=16)(joblib.delayed(run_one)(d[0]) for d in tqdm(total_list))