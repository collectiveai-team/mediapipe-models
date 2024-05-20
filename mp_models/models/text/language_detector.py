from mediapipe.tasks import python

from mp_models.meta import (
    MPDetector,
    LanguageDetectorInput,
    LanguageDetectorOutput,
)


class LanguageDetector(MPDetector):
    def __init__(
        self,
        model_url_path: str = "language_detector/language_detector/float32/1/language_detector.tflite",  # noqa
        min_confidence: float = 0.1,
    ):
        super().__init__(model_url_path=model_url_path)
        options = python.text.LanguageDetectorOptions(
            base_options=self.base_options,
            score_threshold=min_confidence,
        )

        self.detector = python.text.LanguageDetector.create_from_options(
            options
        )

    def detect(
        self,
        detector_input: LanguageDetectorInput,
    ) -> list[LanguageDetectorOutput]:
        detections = self.detector.detect(text=detector_input.text).detections
        if not detections:
            return LanguageDetectorOutput()

        return [
            LanguageDetectorOutput(
                language=detection.language_code,
                confidence=detection.probability,
            )
            for detection in detections
        ]
