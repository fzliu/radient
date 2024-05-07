__all__ = [
    "TimmImageVectorizer"
]

from typing import Any, List

from radient.base import Vector
from radient.util import LazyImport
from radient.image.base import ImageVectorizer
from radient.accelerate import export_to_onnx, ONNXForward

timm = LazyImport("timm")
torch = LazyImport("torch")


class TimmImageVectorizer(ImageVectorizer):
    """Computes image embeddings using `timm`.
    """

    def __init__(self, model_name: str = "resnet50", **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = timm.create_model(model_name, pretrained=True, **kwargs)
        self._model.reset_classifier(0)
        self._model.eval()
        data_config = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_config)

    def vectorize(self, images: List[Any]) -> List[Vector]:
        vectors = []
        with torch.inference_mode():
            for image in images:
                image = ImageVectorizer.standardize_input(image)
                x = self._transform(image.convert("RGB")).unsqueeze(0)
                vectors.append(self._model(x).numpy().squeeze())
        return vectors

    def accelerate(self, **kwargs):
        # `timm` models take a single 4D tensor (`B x C x H x W`) as input.
        onnx_model_path = export_to_onnx(
            self,
            torch.randn((1, 3, 224, 224)),
            axes_names=["batch_size"],
            model_type="pytorch"
        )

        # Monkey-patch the the underlying model's `forward` function to run the
        # optimized ONNX model rather than the torch version.
        self._model.forward = ONNXForward(
            onnx_model_path,
            output_class=torch.tensor,
        )

