__all__ = [
    "ImageBindTextVectorizer"
]

from typing import Any, List

import numpy as np

import urllib.request

from radient.utils import LazyImport, download_cache_file
from radient.vector import Vector
from radient.vectorizers.accelerate import export_to_onnx, ONNXForward
from radient.vectorizers.text.base import TextVectorizer

imagebind_model = LazyImport("imagebind.models", attribute="imagebind_model", package_name="git+https://github.com/facebookresearch/ImageBind@main")
SimpleTokenizer = LazyImport("imagebind.models.multimodal_preprocessors", attribute="SimpleTokenizer", package_name="git+https://github.com/facebookresearch/ImageBind@main")
torch = LazyImport("torch")

IMAGEBIND_VOCAB_URL = "https://github.com/facebookresearch/ImageBind/raw/main/bpe/bpe_simple_vocab_16e6.txt.gz"


class ImageBindTextVectorizer(TextVectorizer):
    """Computes image embeddings using FAIR's ImageBind model.
    """

    def __init__(self, model_name = "imagebind_huge", **kwargs):
        super().__init__()
        self._model_name = model_name
        # TODO(fzliu): remove non-text trunks from this model
        self._model = getattr(imagebind_model, model_name)(pretrained=True)
        self._model.eval()
        vocab_path = download_cache_file(IMAGEBIND_VOCAB_URL)
        self._tokenizer = SimpleTokenizer(bpe_path=vocab_path)

    def _vectorize(self, text: str) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            tokens = self._tokenizer(text).unsqueeze(0)
            output = self._model({imagebind_model.ModalityType.TEXT: tokens})
            vector = output[imagebind_model.ModalityType.TEXT].squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self, **kwargs):
        modality = imagebind_model.ModalityType.TEXT
        inputs = ({modality: self._tokenize("a")}, {})
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
    

