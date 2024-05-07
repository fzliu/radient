__all__ = [
    "SBERTTextVectorizer"
]

from typing import List

from radient.base import Vector
from radient.util import LazyImport
from radient.text.base import TextVectorizer
from radient.accelerate import export_to_onnx, ONNXForward

SentenceTransformer = LazyImport("sentence_transformers", attribute="SentenceTransformer", package="sentence-transformers")
torch = LazyImport("torch")


class SBERTTextVectorizer(TextVectorizer):
    """Text vectorization with `sentence-transformers`.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, **kwargs)

    def vectorize(self, texts: List[str]) -> List[Vector]:
        #TODO(fzliu): token length check
        texts = TextVectorizer.standardize_inputs(texts)
        vectors = self._model.encode(texts)
        return [v.view(Vector) for v in vectors]

    @property
    def model_name(self):
        return self._model_name

    def accelerate(self, **kwargs):
        # Store the model in ONNX format to maximize compatibility with
        # different backends. Since `sentence-transformers` takes a single
        # dictionary input in its underlying `forward` call, the export
        # function will need to take a second empty dictionary as kwargs.
        # Output names are acquired by running the `encode` function and
        # specifying all outputs.
        model_args = (self._model.tokenize(["a"]), {})
        input_names = list(model_args[0].keys())
        output_names = list(self._model.encode("a", output_value=None).keys())
        onnx_model_path = export_to_onnx(
            self,
            model_args,
            axes_names=["batch_size", "max_seq_len"],
            input_names=input_names,
            output_names=output_names,
            model_type="pytorch"
        )

        # Monkey-patch the the underlying model's `forward` function to run the
        # optimized ONNX model rather than the torch version.
        self._model.forward = ONNXForward(
            onnx_model_path,
            output_names=output_names,
            output_class=torch.tensor,
        )
