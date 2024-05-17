__all__ = [
    "ImageBindImageVectorizer"
]

from typing import Any, List

import numpy as np

from radient.utils import LazyImport
from radient.vector import Vector
from radient.vectorizers.accelerate import export_to_onnx, ONNXForward
from radient.vectorizers.audio.base import AudioVectorizer

imagebind_model = LazyImport("imagebind.models", attribute="imagebind_model", package_name="git+https://github.com/facebookresearch/ImageBind@main")
torch = LazyImport("torch")
transforms = LazyImport("torchvision", attribute="transforms")
waveform2melspec = LazyImport("imagebind.data", attribute="waveform2melspec")

CLIP_DURATION = 2
NUM_MEL_BINS = 128
TARGET_LENGTH = 204


class ImageBindAudioVectorizer(AudioVectorizer):
    """Computes image embeddings using FAIR's ImageBind model.
    """

    def __init__(self, model_name = "imagebind_huge", **kwargs):
        super().__init__()
        self._model_name = model_name
        # TODO(fzliu): remove non-audio trunks from this model
        self._model = getattr(imagebind_model, model_name)(pretrained=True)
        self._model.eval()
        self._normalize = transforms.Normalize(mean=-4.268, std=9.138)

    def _transform(self, waveform: np.ndarray):
        output = []
        start = 0
        # Split the waveform into clips of duration CLIP_DURATION. Each
        # waveform is then converted into its Mel spectrum representation.
        while True:
            end = start + self.sample_rate * CLIP_DURATION
            # Ignore the last clip if it's too short.
            if end >= waveform.size(1) - 1:
                end = waveform.size(1) - 1
                break
            mel_spec = waveform2melspec(
                waveform[:,start:end],
                self.sample_rate,
                NUM_MEL_BINS,
                TARGET_LENGTH
            )
            output.append(self._normalize(mel_spec))
            start = end
        return torch.stack(output, dim=0)

    def _preprocess(self, audio: Any) -> np.ndarray:
        audio = super()._preprocess(audio)
        audio = self._transform(audio).unsqueeze(0)
        return audio

    def _vectorize(self, audio: np.ndarray) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            output = self._model({imagebind_model.ModalityType.AUDIO: audio})
            vector = output[imagebind_model.ModalityType.AUDIO].squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self, **kwargs):
        modality = imagebind_model.ModalityType.AUDIO
        inputs = ({modality: torch.randn((1, 400))}, {})
        onnx_model_path = export_to_onnx(
            self,
            inputs,
            axes_names=["batch_size", "seq_len"],
            input_names=[modality],
            output_names=[modality],
            model_type="pytorch"
        )

        self._model.forward = ONNXForward(
            onnx_model_path,
            output_names=names,
        )

    @property
    def sample_rate(self):
        return 16_000
    

