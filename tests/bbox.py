from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bbox:
    left: float
    top: float
    right: float
    bottom: float

    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)

    def iou(self, other: Bbox) -> float:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        if left < right and top < bottom:
            i = Bbox(left, top, right, bottom)
            return i.area() / (self.area() + other.area() - i.area())
        else:
            return 0.0
