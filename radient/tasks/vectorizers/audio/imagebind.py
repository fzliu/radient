__all__ = [
    "ImageBindAudioVectorizer"
]

from typing import Any, List

import numpy as np

from radient.tasks.accelerate import export_to_onnx, ONNXForward
from radient.tasks.vectorizers._imagebind import create_imagebind_model
from radient.tasks.vectorizers._imagebind import imagebind_model
from radient.tasks.vectorizers.audio._base import AudioVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

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
        self._model = create_imagebind_model(model_name=model_name, modality="audio")
        self._model.eval()
        self._normalize = transforms.Normalize(mean=-4.268, std=9.138)

    def _transform(self, waveform: np.ndarray, **kwargs):
        output = []
        # Split the waveform into clips of duration CLIP_DURATION. Each
        # waveform is then converted into its Mel spectrum representation.
        waveform = torch.from_numpy(waveform)
        samples_per_clip = self.sample_rate * CLIP_DURATION
        for n in np.arange(0, waveform.shape[1], samples_per_clip):
            end = n + samples_per_clip
            mel_spec = waveform2melspec(
                waveform[:,n:end],
                self.sample_rate,
                NUM_MEL_BINS,
                TARGET_LENGTH
            )
            output.append(self._normalize(mel_spec))
        return torch.stack(output, dim=0)

    def _preprocess(self, audio: Any, **kwargs) -> np.ndarray:
        audio = super()._preprocess(audio)
        audio = self._transform(audio).unsqueeze(0)
        return audio

    def _vectorize(self, audio: np.ndarray, **kwargs) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            output = self._model({imagebind_model.ModalityType.AUDIO: audio})
            vector = output[imagebind_model.ModalityType.AUDIO].squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self):
        modality = imagebind_model.ModalityType.AUDIO
        inputs = ({modality: torch.randn((1, 400))}, {})
        input_names = output_names = [modality]
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
            output_names=output_names,
        )

    @property
    def sample_rate(self):
        return 16_000
    

