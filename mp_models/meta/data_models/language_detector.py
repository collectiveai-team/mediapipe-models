from pydantic import BaseModel, StrictStr, NonNegativeFloat


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorOutput(BaseModel):
    language: StrictStr | None = None
    confidence: NonNegativeFloat | None = None
