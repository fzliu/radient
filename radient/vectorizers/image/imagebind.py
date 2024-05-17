__all__ = [
    "ImageBindImageVectorizer"
]

from typing import Any, List

from radient.utils import LazyImport
from radient.vector import Vector
from radient.vectorizers.accelerate import export_to_onnx, ONNXForward
from radient.vectorizers.image.base import ImageVectorizer

Image = LazyImport("PIL", attribute="Image")
imagebind_model = LazyImport("imagebind.models", attribute="imagebind_model", package_name="git+https://github.com/facebookresearch/ImageBind@main")
torch = LazyImport("torch")
transforms = LazyImport("torchvision", attribute="transforms")


class ImageBindImageVectorizer(ImageVectorizer):
    """Computes image embeddings using FAIR's ImageBind model.
    """

    def __init__(self, model_name = "imagebind_huge", **kwargs):
        super().__init__()
        self._model_name = model_name
        # TODO(fzliu): remove non-image trunks from this model
        self._model = getattr(imagebind_model, model_name)(pretrained=True)
        self._model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _vectorize(self, image: Image) -> Vector:
        # TODO(fzliu): dynamic batching
        with torch.inference_mode():
            x = self._transform(image.convert("RGB")).unsqueeze(0)
            output = self._model({imagebind_model.ModalityType.VISION: x})
            vector = output[imagebind_model.ModalityType.VISION].squeeze()
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
        return vector.view(Vector)

    def accelerate(self, **kwargs):
        modality = imagebind_model.ModalityType.VISION
        inputs = ({modality: torch.randn((1, 3, 224, 224))}, {})
        onnx_model_path = export_to_onnx(
            self,
            inputs,
            axes_names=["batch_size"],
            input_names=[modality],
            output_names=[modality],
            model_type="pytorch"
        )
        # TODO(fzliu): delete all tensors from model
        self._model.forward = ONNXForward(
            onnx_model_path,
            output_names=names,
        )

