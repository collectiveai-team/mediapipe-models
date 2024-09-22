from pydantic import BaseModel, NonNegativeInt


class BBox(BaseModel):
    x1: NonNegativeInt
    y1: NonNegativeInt
    x2: NonNegativeInt
    y2: NonNegativeInt


class Centroid(BaseModel):
    x: NonNegativeInt
    y: NonNegativeInt
