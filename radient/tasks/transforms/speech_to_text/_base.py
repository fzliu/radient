from abc import abstractmethod
from typing import Dict

from radient.tasks.transforms._base import Transform


class SpeechToTextTransform(Transform):

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def transform(self, data: str) -> Dict[str, str]:
        pass
