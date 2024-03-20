__all__ = [
    "AudioVectorizer"
]

from abc import abstractmethod
from typing import Any, List

from radient.base import Vector, Vectorizer
from radient.util import fully_qualified_name


class AudioVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def standardize_input(cls, text: Any) -> str:
        pass