__all__ = [
    "TimmImageVectorizer"
]

from typing import Any, List

import numpy as np

from radient.util.lazy_import import LazyImport
from radient.vector import Vector
from radient.vectorizers.audio.base import AudioVectorizer
from radient.vectorizers.accelerate import export_to_onnx, ONNXForward

torchaudio = LazyImport("torchaudio")
torch = LazyImport("torch")


class TorchaudioAudioVectorizer(AudioVectorizer):
    """Computes audio embeddings using `torchaudio`.
    """

    def __init__(
        self,
        model_name: str = "HUBERT_BASE",
        reduce_method: str = "avg_pool",
        **kwargs
    ):
        super().__init__()
        self._model_name = model_name
        self._reduce_method = reduce_method
        bundle = getattr(torchaudio.pipelines, model_name)
        self._sample_rate = bundle.sample_rate
        self._model = bundle.get_model()

    def _vectorize(self, audios: List[Any]) -> List[Vector]:
        vectors = []
        for audio in audios:
            waveform, sr = audio
            with torch.inference_mode():
                waveform = torchaudio.functional.resample(waveform, sr, self._sample_rate)
                output = self._model.forward(waveform)
                features = output[0] if isinstance(output, tuple) else output
                # Torchaudio vectorizers output a list of features, so we
                # reduce the features to a single 1D vector using the method
                # specified by the user.
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                if self._reduce_method == "avg_pool":
                    vector = np.mean(features, axis=(0,1))
                vectors.append(vector.view(Vector))
        return vectors

    def accelerate(self, **kwargs):
        # Torchaudio-based vectorizers take an optional `lengths` parameter,
        # which we ignore here.
        onnx_model_path = export_to_onnx(
            self,
            torch.randn((1, 400)),
            axes_names=["batch_size", "seq_len"],
            input_names=["waveforms"],
            output_names=["features"],
            model_type="pytorch"
        )
        self._model.forward = ONNXForward(
            onnx_model_path
        )
