"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as the images in imagenet2012/val_map.txt.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np

dtype_map = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64
}

"""
Calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json.
Assumes that loadgen's query index is in the same order as the images in imagenet2012/val_map.txt.

Args:
    mlperf_accuracy_file (str): Path to mlperf_log_accuracy.json.
    imagenet_val_file (str): Path to imagenet val_map.txt.
    verbose (bool): If True, prints detailed messages.
    dtype (str): Data type of the label. Choices are 'float32', 'int32', 'int64'.
    scenario (str): The scenario under which the accuracy was calculated.

Returns:
    dict: A dictionary containing 'good', 'total', and 'scenario'.
"""

def calculate_accuracy(mlperf_accuracy_file, imagenet_val_file, verbose=False, dtype='float32', scenario='Unknown'):

    imagenet = []
    with open(imagenet_val_file, "r") as f:
        for line in f:
            cols = line.strip().split()
            imagenet.append((cols[0], int(cols[1])))

    with open(mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    good = 0
    for j in results:
        idx = j['qsl_idx']

        # De-duplicate in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)

        # Get the expected label and image
        img, label = imagenet[idx]

        # Reconstruct label from mlperf accuracy log
        data = np.frombuffer(bytes.fromhex(j['data']), dtype_map[dtype])
        found = int(data[0])
        if label == found:
            good += 1
        else:
            if verbose:
                print("{}, expected: {}, found {}".format(img, label, found))

    accuracy = 100.0 * good / len(seen)
    print("accuracy={:.3f}%, good={}, total={}".format(accuracy, good, len(seen)))
    if verbose:
        print("Found and ignored {} duplicates".format(len(results) - len(seen)))

    # Return the dictionary
    result_dict = {"good": good, "total": len(seen), "scenario": str(scenario)}
    return result_dict
