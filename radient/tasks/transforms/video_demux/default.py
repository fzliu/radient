from typing import Dict, List

import numpy as np

from radient.tasks.transforms.video_demux._base import VideoDemuxTransform
from radient.utils.lazy_import import LazyImport

cv2 = LazyImport("cv2", package_name="opencv-python")
librosa = LazyImport("librosa")
sf = LazyImport("soundfile")


class DefaultVideoDemuxTransform(VideoDemuxTransform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, data: str) -> Dict[str, List[str]]:
        """Extracts frames and audio snippets from a video file.
        """

        video_path = data
        output_path = self._make_output_dir()

        # Grab the total number of frames as well as the video's FPS to
        # determine the interval in frames and stopping condition.
        video_capture = cv2.VideoCapture(video_path)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_interval = video_capture.get(cv2.CAP_PROP_FPS) * self._interval

        frames = {"data": [], "modality": "image"}
        for i, n in enumerate(np.arange(0, frame_count, frame_interval)):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(n))
            retval, frame = video_capture.read()
            if not retval:
                break
            frame_path = str(output_path / f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
            frames["data"].append(frame_path)

        video_capture.release()

        # Extract audio snippet as raw data. With the raw audio, we store it
        # in `.wav` format with the original sample rate.
        audios = {"data": [], "modality": "audio"}
        waveform, sample_rate = librosa.load(video_path, sr=None, mono=False)
        sample_interval = int(sample_rate * self._interval)
        if len(waveform.shape) == 1:
            y = np.expand_dims(y, axis=0)
        for i, n in enumerate(np.arange(0, waveform.shape[1], sample_interval)):
            n_end = n + sample_interval
            audio_path = str(output_path / f"audio_{i:04d}.wav")
            sf.write(audio_path, waveform[:,n:n_end].T, sample_rate)
            audios["data"].append(audio_path)

        return [frames, audios]
