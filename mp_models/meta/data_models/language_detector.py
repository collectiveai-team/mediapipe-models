from pydantic import BaseModel, StrictStr, NonNegativeFloat


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageOutputInput(BaseModel):
    language: StrictStr
    confidence: NonNegativeFloat
