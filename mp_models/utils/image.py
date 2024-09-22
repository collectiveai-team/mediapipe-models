import cv2

import numpy as np

from mp_models.meta import BBox
from PIL import Image, ImageDraw


def cv2pil(cv_image: np.ndarray) -> Image.Image:
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_image)


def pil2cv(pil_image: Image.Image) -> np.ndarray:
    cv_image = np.array(pil_image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)


def draw_bboxes(
    pil_image: Image.Image,
    bboxes: list[BBox],
    color: str = "blue",
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        draw.rectangle(
            [
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
            ],
            width=width,
            outline=color,
        )
