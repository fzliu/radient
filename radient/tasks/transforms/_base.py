from abc import abstractmethod
from typing import Any

import numpy as np

from radient.tasks._base import Task


class Transform(Task):
    """Transforms are operations that perform multimodal data transformation,
    such as such as turning a video into independent frames. Because these are
    usually I/O-bound operations, batching is not innately supported.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, data: Any) -> Any:
        pass
