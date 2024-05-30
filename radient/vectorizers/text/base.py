__all__ = [
    "TextVectorizer"
]

from abc import abstractmethod
from typing import Any, List

from radient.utils import fully_qualified_name
from radient.vectorizers.base import Vector
from radient.vectorizers.base import Vectorizer



class TextVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(self, text: Any) -> str:
        if not isinstance(text, str):
            return str(text)
        return text
