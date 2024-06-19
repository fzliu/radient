__all__ = [
    "TorchaudioAudioVectorizer"
]

from typing import List, Tuple

import numpy as np

from radient.tasks.accelerate import export_to_onnx, ONNXForward
from radient.tasks.vectorizers.audio._base import AudioVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

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

    def _vectorize(self, audio: np.ndarray, **kwargs) -> List[Vector]:
        with torch.inference_mode():
            output = self._model.forward(torch.from_numpy(audio))
            features = output[0] if isinstance(output, tuple) else output
            if isinstance(features, torch.Tensor):
                features = features.numpy()

            # Torchaudio vectorizers output a list of features, so we
            # optionally reduce the features to a single 1D vector using
            # the method specified by the function caller.
            if self._reduce_method == "avg_pool":
                output = np.mean(features, axis=(0,1)).view(Vector)
            else:
                output = [v.view(Vector) for v in np.mean(features, axis=0)]

            return output

    def accelerate(self):
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

    @property
    def sample_rate(self):
        return self._sample_rate
