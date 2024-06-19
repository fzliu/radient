__all__ = [
    "TimmImageVectorizer"
]

from typing import Any, List

from radient.tasks.accelerate import export_to_onnx, ONNXForward
from radient.tasks.vectorizers.image._base import ImageVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

Image = LazyImport("PIL", attribute="Image", package_name="Pillow")
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

    def _vectorize(self, image: Image, **kwargs) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            x = self._transform(image.convert("RGB")).unsqueeze(0)
            vector = self._model(x).squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self):
        # `timm` models take a single 4D tensor (`B x C x H x W`) as input.
        onnx_model_path = export_to_onnx(
            self,
            torch.randn((1, 3, 224, 224)),
            axes_names=["batch_size"],
            input_names=["images"],
            output_names=["vectors"],
            model_type="pytorch"
        )

        self._model.forward = ONNXForward(
            onnx_model_path
        )

