"""
This code snippet shows an example on calculating
Intersection over Union(IoU) of two bounding boxes.

IoU equals to intersection area / union area.
"""

def intersection_over_union(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    # Locate the intersection area of bounding box 1 and 2
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _area(bbox1) + _area(bbox2) - intersection
    return intersection / union


def _area(bbox: list | tuple) -> float:
    """Compute area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (y2 - y1) * (x2 - x1)


if __name__ == "__main__":
    bbox1 = (1, 2, 4, 4)
    bbox2 = (2, 3, 4, 5)
    print(intersection_over_union(bbox1, bbox2)) # 0.25
