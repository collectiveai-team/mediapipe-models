from mediapipe.tasks import python

from mp_models.meta import (
    MPDetector,
    LanguageDetectorInput,
    LanguageOutputInput,
)


class LanguageDetector(MPDetector):
    def __init__(
        self,
        model_url_path: str = "language_detector/language_detector/float32/1/language_detector.tflite",  # noqa
    ):
        super().__init__(model_url_path=model_url_path)
        self.detector = python.text.LanguageDetector.create_from_options(
            self.options
        )

    def detect(
        self,
        detector_input: LanguageDetectorInput,
    ) -> LanguageOutputInput:
        detection = self.detector.detect(text=detector_input.text)
        detection = detection.detections[0]

        return LanguageOutputInput(
            language=detection.language_code,
            confidence=detection.probability,
        )
