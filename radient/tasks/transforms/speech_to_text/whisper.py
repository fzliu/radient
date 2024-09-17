from typing import Dict, Union

from radient.tasks.transforms.speech_to_text._base import SpeechToTextTransform
from radient.utils.lazy_import import LazyImport

AutoModelForSpeechSeq2Seq = LazyImport("transformers", attribute="AutoModelForSpeechSeq2Seq", package_name="transformers")
AutoProcessor = LazyImport("transformers", attribute="AutoProcessor", package_name="transformers")
pipeline = LazyImport("transformers", attribute="pipeline", package_name="transformers")
torch = LazyImport("torch")


class WhisperSpeechToTextTransform(SpeechToTextTransform):

    def __init__(self,
        model_id: str = "openai/whisper-large-v3",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):

        # Create model and preprocessor.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device="cpu"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # Instantiate ASR pipeline.
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device="cpu",
        )

    def transform(self, data: str) -> Dict[str, str]:
        result = self._pipeline(data)
        return {"data": results["text"], "modality": "text"}
