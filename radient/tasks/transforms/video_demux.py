from pathlib import Path
from typing import Any, Dict, List
import uuid

import numpy as np

from radient.tasks.transforms._base import Transform
from radient.utils.lazy_import import LazyImport

cv2 = LazyImport("cv2", package_name="opencv-python")
ffmpeg = LazyImport("ffmpeg", package_name="ffmpeg-python")
librosa = LazyImport("librosa")
sf = LazyImport("soundfile")


class VideoDemuxTransform(Transform):

    def __init__(self,
        interval: float = 2.0,
        output_directory: str = "~/.radient/data/video_demux",
        **kwargs
    ):
        super().__init__()
        self._interval = interval
        output_directory = Path(output_directory).expanduser()
        self._output_directory = output_directory

    def transform(
        self, 
        data: str
    ) -> Dict[str, List[str]]:
        """Extracts frames and audio snippets from a video file.
        """

        # The full output directory comes from a combination of the user's
        # specification plus a unique identifier for the current run.
        # TODO(fzliu): switch to an incremental identifier e.g. UUIDv7
        output_path = Path(self._output_directory) / str(uuid.uuid4())
        output_path.mkdir(parents=True, exist_ok=True)
        video_path = data

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
        for i, n in enumerate(range(0, waveform.shape[1], sample_interval)):
            n_end = n + sample_interval
            audio_path = str(output_path / f"audio_{i:04d}.wav")
            sf.write(audio_path, waveform[:,n:n_end].T, sample_rate)
            audios["data"].append(audio_path)

        return [frames, audios]

    def _transform_ffmpeg(self, data: str) -> Dict:
        """For test/debugging purposes only.
        """

        video_path = data
        frames = []
        audios = []

        # The video duration determines how many intervals we have, while the
        # sample rate is used for the downstream audio vectorizer.
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
        sample_rate = int(probe["streams"][1]["sample_rate"])

        for i in range(int(duration/self._interval)):
            start_time = i * self._interval

            # Extract a single frame, which we then convert into a PIL Image.
            frame_data = ffmpeg.input(
                video_path,
                ss=start_time,
                vframes=1
            ).output(
                "pipe:", format="image2", vcodec="mjpeg"
            ).run(capture_stdout=True, capture_stderr=True)[0]
            frames.append(Image.open(BytesIO(frame_data)))

            # Extract audio snippet as raw data. With the raw audio, we output
            # a (waveform, sample_rate) pair.
            audio_data = ffmpeg.input(
                video_path,
                ss=start_time,
                t=self._interval
            ).output(
                "pipe:", format="wav"
            ).run(capture_stdout=True, capture_stderr=True)[0]
            waveform = np.frombuffer(audio_data, np.int16)
            audio_list.append((waveform, sample_rate))

        return {"audio": audios, "image": frames}
