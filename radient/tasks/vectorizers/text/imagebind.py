__all__ = [
    "ImageBindTextVectorizer"
]

from typing import Any, List

import numpy as np

import urllib.request

from radient.tasks.accelerate import export_to_onnx, ONNXForward
from radient.tasks.vectorizers._imagebind import create_imagebind_model
from radient.tasks.vectorizers._imagebind import imagebind_model
from radient.tasks.vectorizers.text._base import TextVectorizer
from radient.utils import download_cache_file
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

SimpleTokenizer = LazyImport("imagebind.models.multimodal_preprocessors", attribute="SimpleTokenizer", package_name="git+https://github.com/fzliu/ImageBind@main")
torch = LazyImport("torch")

IMAGEBIND_VOCAB_URL = "https://github.com/fzliu/ImageBind/raw/main/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz"



class ImageBindTextVectorizer(TextVectorizer):
    """Computes image embeddings using FAIR's ImageBind model.
    """

    def __init__(self, model_name = "imagebind_huge", **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = create_imagebind_model(model_name=model_name, modality="text")
        self._model.eval()
        vocab_path = download_cache_file(IMAGEBIND_VOCAB_URL)
        self._tokenizer = SimpleTokenizer(bpe_path=vocab_path)

    def _vectorize(self, text: str, **kwargs) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            tokens = self._tokenizer(text).unsqueeze(0)
            output = self._model({imagebind_model.ModalityType.TEXT: tokens})
            vector = output[imagebind_model.ModalityType.TEXT].squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self):
        modality = imagebind_model.ModalityType.TEXT
        input_names = output_names = [modality]
        inputs = ({modality: self._tokenizer("a")}, {})
        onnx_model_path = export_to_onnx(
            self,
            inputs,
            axes_names=["batch_size"],
            input_names=input_names,
            output_names=output_names,
            model_type="pytorch"
        )

        self._model.forward = ONNXForward(
            onnx_model_path,
            output_names=output_names,
        )

    @property
    def sample_rate(self):
        return 16_000
    

