from __future__ import division, print_function, unicode_literals

import json
import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_coco_map(mlperf_accuracy_file,
                       coco_dir,
                       verbose=False,
                       output_file="coco-results.json",
                       use_inv_map=False,
                       remove_48_empty_images=False):
    
    cocoGt = COCO(os.path.join(coco_dir, "annotations", "instances_val2017.json"))

    if use_inv_map:
        # First label in inv_map is not used
        inv_map = [0] + cocoGt.getCatIds()

    with open(mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    # If requested, handle datasets that had empty images removed
    if remove_48_empty_images:
        im_ids = []
        for cat_id in cocoGt.getCatIds():
            im_ids += cocoGt.catToImgs[cat_id]
        im_ids = list(set(im_ids))
        image_map = [cocoGt.imgs[id_i] for id_i in im_ids]
    else:
        image_map = cocoGt.dataset["images"]

    detections = []
    image_ids = set()
    seen = set()
    no_results = 0

    for entry in results:
        idx = entry["qsl_idx"]
        # De-duplicate in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)

        data = np.frombuffer(bytes.fromhex(entry["data"]), np.float32)
        if len(data) < 7:
            # Handle images that had no results
            image = image_map[idx]
            image_ids.add(image["id"])
            no_results += 1
            if verbose:
                print("no results: {}, idx={}".format(image["coco_url"], idx))
            continue

        # Process each detection in the current result entry
        for i in range(0, len(data), 7):
            image_idx, ymin, xmin, ymax, xmax, score, label = data[i: i + 7]
            image = image_map[idx]
            if int(image_idx) != idx:
                print(
                    "ERROR: loadgen({}) and payload({}) disagree on image_idx".format(
                        idx, int(image_idx)
                    )
                )
            image_id = image["id"]
            height, width = image["height"], image["width"]

            # Scale back to original image coordinates
            ymin *= height
            xmin *= width
            ymax *= height
            xmax *= width

            if use_inv_map:
                label = inv_map[int(label)]
            else:
                label = int(label)

            detections.append(
                {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(xmax - xmin),
                        float(ymax - ymin),
                    ],
                    "score": float(score),
                }
            )
            image_ids.add(image_id)

    # Save detections to file
    with open(output_file, "w") as fp:
        json.dump(detections, fp, sort_keys=True, indent=4)

    # Evaluate with COCO
    cocoDt = cocoGt.loadRes(output_file)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.imgIds = list(image_ids)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    mAP = 100.0 * cocoEval.stats[0]

    if verbose:
        print("mAP={:.3f}%".format(mAP))
        print("found {} results".format(len(results)))
        print("found {} images".format(len(image_ids)))
        print("found {} images with no results".format(no_results))
        print("ignored {} dupes".format(len(results) - len(seen)))

    result_dict = {
        "mAP": mAP,
        "no_results": no_results,
        "total_images": len(image_ids),
        "duplicate_count": len(results) - len(seen),
        "scenario": "COCO Object Detection"
    }

    return result_dict
