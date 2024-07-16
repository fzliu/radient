__all__ = [
    "ImageBindImageVectorizer"
]

from typing import Any, List

from radient.tasks.accelerate import export_to_onnx, ONNXForward
from radient.tasks.vectorizers._imagebind import create_imagebind_model
from radient.tasks.vectorizers._imagebind import imagebind_model
from radient.tasks.vectorizers.image._base import ImageVectorizer
from radient.utils.lazy_import import LazyImport
from radient.vector import Vector

Image = LazyImport("PIL", attribute="Image", package_name="Pillow")
torch = LazyImport("torch")
transforms = LazyImport("torchvision", attribute="transforms")


class ImageBindImageVectorizer(ImageVectorizer):
    """Computes image embeddings using FAIR's ImageBind model.
    """

    def __init__(self, model_name: str = "imagebind_huge", **kwargs):
        super().__init__()
        self._model_name = model_name
        # TODO(fzliu): remove non-image trunks from this model
        self._model = create_imagebind_model(model_name=model_name, modality="image")
        self._model.eval()
        self._transform = transforms.Compose([
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _vectorize(self, image: Image, **kwargs) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            x = self._transform(image.convert("RGB")).unsqueeze(0)
            output = self._model({imagebind_model.ModalityType.VISION: x})
            vector = output[imagebind_model.ModalityType.VISION].squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self):
        modality = imagebind_model.ModalityType.VISION
        inputs = ({modality: torch.randn((1, 3, 224, 224))}, {})
        input_names = output_names = [modality]
        onnx_model_path = export_to_onnx(
            self,
            inputs,
            axes_names=["batch_size"],
            input_names=input_names,
            output_names=output_names,
            model_type="pytorch"
        )
        # TODO(fzliu): delete all tensors from model
        self._model.forward = ONNXForward(
            onnx_model_path,
            output_names=output_names,
        )

