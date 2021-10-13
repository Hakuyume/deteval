from typing import List, Mapping

import numpy
from pycocotools.cocoeval import COCOeval

from deteval import GroundTruth, Prediction, calculate_matrix

from .bbox import Bbox


def test_calculate_matrix(
    eps: float,
    ground_truths: Mapping[str, List[GroundTruth[int, Bbox]]],
    predictions: Mapping[str, List[Prediction[int, Bbox]]],
    cocoeval: COCOeval,
) -> None:
    cocoparams = cocoeval.eval["params"]

    matrices = calculate_matrix(
        ((ground_truths[image_id], preds) for image_id, preds in predictions.items()),
        list(cocoparams.iouThrs),
    )

    for (iou_threshold, category), matrix in matrices.items():
        t = numpy.flatnonzero(numpy.absolute(cocoparams.iouThrs - iou_threshold) < eps)[
            0
        ]
        k = cocoparams.catIds.index(category)
        a = cocoparams.areaRngLbl.index("all")
        m = cocoparams.maxDets.index(100)
        if cocoeval.eval["recall"][t, k, a, m] >= 0:
            numpy.testing.assert_allclose(  # type:ignore
                len(matrix.true_positive_scores) / matrix.total,
                cocoeval.eval["recall"][t, k, a, m],
                atol=eps,
            )
        else:
            assert matrix.total == 0
            assert matrix.true_positive_scores == []
