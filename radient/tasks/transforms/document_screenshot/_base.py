from abc import abstractmethod
from typing import Union

from radient.tasks.transforms._base import Transform


class DocumentScreenshotTransform(Transform):

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def transform(self, data: str) -> dict[str, str]:
        pass
