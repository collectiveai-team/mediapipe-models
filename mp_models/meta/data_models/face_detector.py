import numpy as np

from pydantic import BaseModel, ConfigDict

from .output import BBox, Centroid


class FaceDetectorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    np_image: np.ndarray


class FaceDetection(BaseModel):
    centroid: Centroid | None = None
    bbox: BBox | None = None
