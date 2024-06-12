__all__ = [
    "multimodal_vectorizer"
]

from typing import Any, Set, List, Type, Optional

from radient.tasks.vectorizers._base import Vectorizer
from radient.tasks.vectorizers.audio.imagebind import ImageBindAudioVectorizer
from radient.tasks.vectorizers.image.imagebind import ImageBindImageVectorizer
from radient.tasks.vectorizers.text.imagebind import ImageBindTextVectorizer

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

    def _vectorize(self, data: Any, **kwargs):
        modality = kwargs.get("modality")
        vector = self._vectorizers[modality](data)
        vector.putmeta("modality", modality)
        return vector

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