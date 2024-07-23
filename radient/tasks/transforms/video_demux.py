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

    def transform(self,
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

        # Default to `ffmpeg` if it is available. If not, fall back to
        # OpenCV + librosa.
        try:
            print("a")
            subprocess.call("ffprobe")
            print("b")
            subprocess.call("ffmpeg")
            print("c")
            return self._transform_ffmpeg(output_path, video_path)
        except Exception as e:
            pass
        return self._transform_fallback(output_path, video_path)

    def _transform_ffmpeg(self,
        output_path: str,
        video_path: str
    ) -> Dict[str, List[str]]:
        """For test/debugging purposes only.
        """

        frames = {"data": [], "modality": "image"}
        audios = {"data": [], "modality": "audio"}

        # Get video information using ffprobe.
        video_info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,r_frame_rate,sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True).stdout.split()
        frame_count = int(video_info[0])
        frame_rate = eval(video_info[1])
        frame_interval = frame_rate * self._interval
        sample_interval = int(sample_rate * self._interval)

        # Extract frames as PNG images.
        for i, n in enumerate(np.arange(0, frame_count, frame_interval)):
            frame_time = n / frame_rate
            #frame_path = str(output_path / f"frame_{i:04d}.png")
            frame_path = str(output_path / f"frame_{n:04d}.png")
            subprocess.run(["ffmpeg", "-ss", str(frame_time), "-i", video_path, "-vframes", "1", frame_path])
            frames["data"].append(frame_path)

        # Extract audio snippet as raw data.
        for i in range(0, frame_count, int(frame_interval)):
            audio_path = str(output_path / f"audio_{i:04d}.wav")
            start_time = i / frame_rate
            subprocess.run(["ffmpeg", "-ss", str(start_time), "-i", video_path, 
                            "-t", str(self._interval), "-q:a", "0", "-map", "a", audio_path])
            audios["data"].append(audio_path)

        return [frames, audios]

    def _transform_fallback(self, 
        output_path: str,
        video_path: str
    ) -> Dict[str, List[str]]:

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
