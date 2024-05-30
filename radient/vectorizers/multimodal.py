__all__ = [
    "multimodal_vectorizer"
]

from typing import Any, Set, List, Type, Optional

from radient.vectorizers.base import Vectorizer
from radient.vectorizers.audio.imagebind import ImageBindAudioVectorizer
from radient.vectorizers.image.imagebind import ImageBindImageVectorizer
from radient.vectorizers.text.imagebind import ImageBindTextVectorizer

IMAGEBIND_VECTORIZERS = {
    ImageBindAudioVectorizer,
    ImageBindImageVectorizer,
    ImageBindTextVectorizer
}


class MultimodalVectorizer(Vectorizer):

    def __init__(self, vectypes: Set[Type], **kwargs):
        super().__init__()
        self._vectorizers = {}
        for VecType in vectypes:
            vectorizer = VecType(**kwargs)
            self._vectorizers[vectorizer.vtype] = vectorizer

    def modalities(self) -> List[str]:
        return list(self._vectorizers.keys())

    def _vectorize(self, data: Any, modality: str, **kwargs):
        return self._vectorizers[modality](data)

    def accelerate(self, **kwargs):
        for vectorizer in self._vectorizers.values():
            vectorizer.accelerate(**kwargs)


def multimodal_vectorizer(method: Optional[str], **kwargs) -> MultimodalVectorizer:
    """Creates a text vectorizer specified by `method`.
    """

    if method in ("imagebind",):
        return MultimodalVectorizer(IMAGEBIND_VECTORIZERS, **kwargs)
    else:
        raise NotImplementedError