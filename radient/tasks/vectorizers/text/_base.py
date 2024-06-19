__all__ = [
    "TextVectorizer"
]

from abc import abstractmethod
from typing import Any, List

from radient.tasks.vectorizers._base import Vectorizer
from radient.utils import fully_qualified_name
from radient.vector import Vector


class TextVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(self, text: Any, **kwargs) -> str:
        if not isinstance(text, str):
            return str(text)
        return text
