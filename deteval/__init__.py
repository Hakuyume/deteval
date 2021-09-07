from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Generic, Iterator, List, Protocol, Tuple, TypeVar

R = TypeVar("R", bound="Region")


class Region(Protocol):
    def iou(self: R, other: R) -> float:
        ...


@dataclass
class GroundTruth(Generic[R]):
    category: int
    region: R


@dataclass
class Prediction(Generic[R]):
    category: int
    region: R
    score: float


@dataclass
class ConfusionMatrix:
    total: int = 0
    true_positives: List[float] = field(default_factory=list)
    false_positives: List[float] = field(default_factory=list)


def calculate_matrix(
    inputs: Iterator[Tuple[List[GroundTruth[R]], List[Prediction[R]]]],
    iou_thresholds: List[float],
) -> DefaultDict[Tuple[float, int], ConfusionMatrix]:
    matrices: DefaultDict[Tuple[float, int], ConfusionMatrix] = defaultdict(
        ConfusionMatrix
    )

    for gts, preds in inputs:
        matches_all = []
        for gt_index, gt in enumerate(gts):
            for pred_index, pred in enumerate(preds):
                if pred.category == gt.category:
                    matches_all.append(
                        (gt.region.iou(pred.region), gt_index, pred_index)
                    )

        matches_all = sorted(matches_all, reverse=True)
        matches = []
        gt_indices = set()
        pred_indices = set()
        for (iou, gt_index, pred_index) in matches_all:
            if gt_index in gt_indices:
                continue
            if pred_index in pred_indices:
                continue
            matches.append((iou, gt_index, pred_index))
            gt_indices.add(gt_index)
            pred_indices.add(pred_index)

        for iou_threshold in iou_thresholds:
            for gt in gts:
                matrices[(iou_threshold, gt.category)].total += 1

            for (iou, gt_index, pred_index) in matches:
                matrix = matrices[(iou_threshold, gts[gt_index].category)]
                if iou >= iou_threshold:
                    matrix.true_positives.append(preds[pred_index].score)
                else:
                    matrix.false_positives.append(preds[pred_index].score)

            for pred_index in range(len(preds)):
                if pred_index not in pred_indices:
                    matrix = matrices[(iou_threshold, preds[pred_index].category)]
                    matrix.false_positives.append(preds[pred_index].score)

    return matrices
