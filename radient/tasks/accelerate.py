from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from radient.utils import fully_qualified_name
from radient.utils.lazy_import import LazyImport
from radient.tasks.vectorizers._base import Vectorizer

torch = LazyImport("torch")
onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime", package_name="onnxruntime-gpu")


def export_to_onnx(
    vectorizer: Vectorizer,
    model_args: Union[Tuple[Any, ...], Any],
    axes_names: Sequence[str] = [],
    input_names: Sequence[str] = [],
    output_names: Sequence[str] = [],
    model_type: Optional[str] = None
):
    """Attempts to export a model in ONNX format for use with `onnxruntime`.
    Switches export implementation based on torch, tensorflow, or scikit-learn
    models.
    """

    # If a model type/library was not specified, attempt to programmatically
    # determine it using the object's fully qualified name. This doesn't work
    # for child classes (e.g. inheriting `nn.Module` or `nn.Sequential`) yet.
    if not model_type:
        model_qualified_name = fully_qualified_name(vectorizer.model)
        if "torch.nn" in model_type:
            model_type = "pytorch"
        elif "tensorflow" in model_type:
            model_type = "tensorflow"
        elif "sklearn.feature_extraction" in model_type:
            model_type = "sklearn"
        else:
            raise NotImplementedError

    # Model path example:
    # "~/.radient/accelerated_models/<method_name>/<model_name>.onnx"
    onnx_model_path = Path.home() / ".radient" / "accelerated_models"
    onnx_model_path /= vectorizer.vtype
    onnx_model_path /= vectorizer.model_name + ".onnx"
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_model_path = str(onnx_model_path)

    if model_type in ("pytorch", "torch"):
        # Generate dynamic axes on-the-fly.
        dynamic_axes = {}
        if input_names and output_names:
            #symbolic_names = {0: "batch_size", 1: "max_seq_len"}
            symbolic_names = dict(zip(range(len(axes_names)), axes_names))
            dynamic_axes.update({k: symbolic_names for k in input_names})
            dynamic_axes.update({k: symbolic_names for k in output_names})
        torch.onnx.export(
            vectorizer.model,
            model_args,
            onnx_model_path,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
    elif model_type in ("tensorflow", "tf"):
        raise NotImplementedError
    elif model_type in ("scikit-learn", "sklearn"):
        raise NotImplementedError
    else:
        raise NotImplementedError

    return onnx_model_path


class ONNXForward(object):
    """Callable object that runs forward inference on an ONNX model.
    """

    def __init__(
        self,
        model_path: str,
        output_names: Optional[List[str]] = None,
        output_class: Optional[Callable] = None,
        providers: Optional[List[str]] = None
    ):
        super().__init__()
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._output_names = output_names
        self._output_class = output_class

    def __call__(
        self,
        inputs: Union[Dict, Sequence, np.ndarray]
    ) -> List[Union[Dict, np.ndarray]]:
        inputs_ = {}
        input_names = [node.name for node in self._session.get_inputs()]
        if isinstance(inputs, dict):
            # For dictionary inputs, ONNX has a tendency to append a `.N` for
            # tensors that have the identical names in Pytorch model
            # definitions. For example:
            #
            # `attention_mask` -> `attention_mask.3`
            #
            # We automatically detect and compensate for these changes here.
            for name, feat in inputs.items():
                is_match = lambda x: name == x.split(".")[0]
                nms = [nm for nm in input_names if is_match(nm)]
                assert len(nms) == 1, "found conflicting input names"
                inputs_[nms[0]] = np.array(feat)
        elif isinstance(inputs, list):
            inputs = [np.array(item) for item in inputs]
            inputs_ = dict(zip(input_names, inputs))
        else:
            inputs_ = {input_names[0]: np.array(inputs)}

        # Optionally cast model outputs to the desired type, e.g. torch.Tensor.
        result = self._session.run(self._output_names, inputs_)
        if self._output_class:
            result = [self._output_class(arr) for arr in result]

        if self._output_names:
            # If output names were specified, return the result as a
            # dictionary rather than a list.
            assert len(result) == len(self._output_names), "length mismatch"
            result_ = dict(zip(self._output_names, result))
        elif len(result) == 1:
            result_ = result[0]
        else:
            result_ = result
        return result_
