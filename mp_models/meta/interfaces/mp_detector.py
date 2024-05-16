import os
import requests

import mediapipe as mp

from typing import Optional
from pydantic import BaseModel
from mediapipe.tasks import python
from abc import ABC, abstractmethod

from mp_models.utils.path import create_path


DEVICE_MAP = {
    "cpu": mp.tasks.BaseOptions.Delegate.CPU,
    "gpu": mp.tasks.BaseOptions.Delegate.GPU,
}


class MPDetector(ABC):
    def __init__(
        self,
        model_url_path: str,
        device: str = "cpu",
        local_model_path: str = "/resources/models",
        base_model_url: str = "https://storage.googleapis.com/mediapipe-models",  # noqa
    ):

        self.local_model_path = local_model_path
        create_path(self.local_model_path)

        model_url = f"{base_model_url}/{model_url_path}"
        file_name = os.path.basename(model_url)
        out_path = os.path.join(self.local_model_path, file_name)

        self._download_model(out_path=out_path, model_url=model_url)
        self.options = python.text.LanguageDetectorOptions(
            base_options=python.BaseOptions(
                delegate=DEVICE_MAP.get(device, "cpu"),
                model_asset_path=out_path,
            )
        )

    def _download_model(self, out_path: str, model_url: str) -> None:
        if os.path.isfile(out_path):
            return

        req = requests.get(model_url)
        assert req.status_code == 200

        with open(out_path, "wb") as f:
            f.write(req.content)

    @abstractmethod
    def detect(self, detector_input: BaseModel) -> Optional[BaseModel]:
        pass
