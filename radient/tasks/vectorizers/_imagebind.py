from radient.utils.lazy_import import LazyImport

imagebind_model = LazyImport("imagebind.models", attribute="imagebind_model", package_name="git+https://github.com/fzliu/ImageBind@main")

IMAGEBIND_MODULE_NAMES = (
    "modality_preprocessors",
    "modality_trunks",
    "modality_heads",
    "modality_postprocessors"
)


def create_imagebind_model(modality: str, model_name: str = "imagebind_huge"):
    """Wrapper around `imagebind_model` to load a specific modality. Modalities
    aside from the one specified are removed from the model. (It's might be
    better to get this code merged into the original ImageBind repo.)
    """

    if modality == "image":
        modality = imagebind_model.ModalityType.VISION
    elif modality == "text":
        modality = imagebind_model.ModalityType.TEXT
    elif modality == "audio":
        modality = imagebind_model.ModalityType.AUDIO

    model = getattr(imagebind_model, model_name)(pretrained=True)

    # Delete unnecessary modality trunks, preprocessors, and postprocessors
    # from the model.
    for module_name in IMAGEBIND_MODULE_NAMES:
        for modality_type in imagebind_model.ModalityType.__dict__.values():
            if modality_type != modality:
                module = getattr(model, module_name)
                delattr(module, modality_type)

    return model

