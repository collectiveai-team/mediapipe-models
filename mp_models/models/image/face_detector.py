import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python.components.containers.detections import Detection

from mp_models.meta import (
    MPDetector,
    FaceDetectorInput,
    FaceDetection,
)


class FaceDetector(MPDetector):
    def __init__(
        self,
        model_url_path: str = "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        device: str = "cpu",
    ):
        super().__init__(
            model_url_path=model_url_path,
            device=device,
        )

        options = python.vision.FaceDetectorOptions(
            base_options=self.base_options,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3,
        )

        self.detector = python.vision.FaceDetector.create_from_options(options)

    def _parse_detections(self, detection: Detection) -> FaceDetection:
        bounding_box = detection.bounding_box
        return FaceDetection(
            centroid={
                "x": bounding_box.origin_x + bounding_box.width // 2,
                "y": bounding_box.origin_y + bounding_box.height // 2,
            },
            bbox={
                "x1": bounding_box.origin_x,
                "y1": bounding_box.origin_y,
                "x2": bounding_box.origin_x + bounding_box.width,
                "y2": bounding_box.origin_y + bounding_box.height,
            },
        )

    def detect(
        self,
        detector_input: FaceDetectorInput,
    ) -> list[FaceDetection]:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=detector_input.np_image,
        )

        detections = self.detector.detect(mp_image).detections
        if not detections:
            return

        return [
            self._parse_detections(detection=detection)
            for detection in detections
        ]
