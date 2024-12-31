__all__ = [
    "SpeechToTextTransform",
    "WhisperSpeechToTextTransform"
]

from typing import Optional

from radient.tasks.transforms.speech_to_text._base import SpeechToTextTransform
from radient.tasks.transforms.speech_to_text.whisper import WhisperSpeechToTextTransform


def speech_to_text_transform(method: str = "whisper", **kwargs) -> SpeechToTextTransform:
    """Creates a Huggingface pipeline for ASR specified by `method`.
    """

    if method in ("whisper", None):
        return WhisperSpeechToTextTransform(**kwargs)
    else:
        raise NotImplementedError
