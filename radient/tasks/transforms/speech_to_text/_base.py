from abc import abstractmethod

from radient.tasks.transforms._base import Transform


class SpeechToTextTransform(Transform):

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def transform(self, data: str) -> dict[str, str]:
        pass
