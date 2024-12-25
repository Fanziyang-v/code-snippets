"""
This code snippet is an implementation of Non-Maximum Suppression(NMS) algorithm.

A bounding box is described by a six-tuple (class, confidence, x1, y1, x2, y2),
where (x1, y1) is the top-left corner coordinate, (x2, y2) is the bottom-right coordinate,
class is the predicted category of the object, and the confidence is the probability that
the bounding box contains an object.

The NMS algorithm works as follows.
1. Discard the bounding boxes with the confidence less than Confidence threshold.
2. Sort the bounding boxes in a descending order of confidence.
3. Loop over all the remaining boxes, starting first with the box that has highest confidence.
4. Calculate the IOU of the current box, with every remaining box that belongs to the same class.
5. If the IOU of the 2 boxes > IoU threshold, remove the box with a lower confidence from our list of boxes.
6. Repeat this operation until we have gone through all the boxes in the list.
"""


def non_maximum_suppression(
    bboxes: list[tuple[int, float, float, float, float, float]],
    conf_threshold: float = 0.8,
    iou_threshold: float = 0.5,
) -> list[tuple]:
    """Non-maximum suppression(NMS) algorithm.

    Args:
        bboxes (list[tuple]): a list of bounding boxes.
        conf_threshold (float, optional): confidence threshold. Defaults to 0.8.
        iou_threshold (float, optional): IoU threshold. Defaults to 0.5.

    Returns:
        list[tuple]: a list of bounding boxes after non-maximum suppression.
    """
    bboxes = [bbox for bbox in bboxes if bbox[1] >= conf_threshold]
    bboxes = sorted(bboxes, key=lambda bbox: bbox[1], reverse=True)
    bboxes_nms = []
    while bboxes:
        chosen_bbox = bboxes.pop(0)
        bboxes = [
            bbox
            for bbox in bboxes
            if bbox[0] != chosen_bbox[0]
            or iou(bbox[2:], chosen_bbox[2:]) < iou_threshold
        ]
        bboxes_nms.append(chosen_bbox)
    return bboxes_nms


def iou(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """Compute IoU of two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _area(bbox1) + _area(bbox2) - intersection
    return intersection / union


def _area(bbox: tuple[float, float, float, float]) -> float:
    """Compute area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (y2 - y1) * (x2 - x1)
