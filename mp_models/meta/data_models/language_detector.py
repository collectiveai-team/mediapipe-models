from typing import Optional
from pydantic import BaseModel, StrictStr, NonNegativeFloat


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorOutput(BaseModel):
    language: Optional[StrictStr] = None
    confidence: Optional[NonNegativeFloat] = None
