import json
from collections import defaultdict
from typing import DefaultDict, List

import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from deteval import GroundTruth, Prediction

from .bbox import Bbox


@pytest.fixture(scope="session")
def eps() -> float:
    return 1e-6


@pytest.fixture(scope="session")
def ground_truths() -> DefaultDict[str, List[GroundTruth[int, Bbox]]]:
    ground_truths = defaultdict(list)
    with open("instances_val2014.json") as f:
        for g in json.load(f)["annotations"]:
            if g["category_id"] != 88:
                continue
            ground_truths[g["image_id"]].append(
                GroundTruth(
                    category=g["category_id"],
                    region=Bbox(
                        left=g["bbox"][0],
                        top=g["bbox"][1],
                        right=g["bbox"][0] + g["bbox"][2],
                        bottom=g["bbox"][1] + g["bbox"][3],
                    ),
                )
            )
    return ground_truths


@pytest.fixture(scope="session")
def predictions() -> DefaultDict[str, List[Prediction[int, Bbox]]]:
    predictions = defaultdict(list)
    with open("instances_val2014_fakebbox100_results.json") as f:
        for p in json.load(f):
            if p["category_id"] != 88:
                continue
            predictions[p["image_id"]].append(
                Prediction(
                    category=p["category_id"],
                    region=Bbox(
                        left=p["bbox"][0],
                        top=p["bbox"][1],
                        right=p["bbox"][0] + p["bbox"][2],
                        bottom=p["bbox"][1] + p["bbox"][3],
                    ),
                    score=p["score"],
                )
            )
    return predictions


@pytest.fixture(scope="session")
def cocoeval() -> COCOeval:
    ground_truths = COCO("instances_val2014.json")
    predictions = ground_truths.loadRes("instances_val2014_fakebbox100_results.json")

    cocoeval = COCOeval(ground_truths, predictions, "bbox")
    cocoeval.params.imgIds = [
        image_id
        for image_id in predictions.getImgIds()
        if len(predictions.getAnnIds(imgIds=[image_id])) > 0
    ]
    cocoeval.evaluate()
    cocoeval.accumulate()

    return cocoeval
