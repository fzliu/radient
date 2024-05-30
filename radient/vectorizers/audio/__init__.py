__all__ = [
    "TorchaudioAudioVectorizer",
    "audio_vectorizer"
]

from typing import Optional

from radient.vectorizers.audio.base import AudioVectorizer
from radient.vectorizers.audio.imagebind import ImageBindAudioVectorizer
from radient.vectorizers.audio.torchaudio import TorchaudioAudioVectorizer


def audio_vectorizer(method: str = "torchaudio", **kwargs) -> AudioVectorizer:
    """Creates an image vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in ("torchaudio",):
        return TorchaudioAudioVectorizer(**kwargs)
    elif method in ("imagebind",):
        return ImageBindAudioVectorizer(**kwargs)
    else:
        raise NotImplementedError
