import shutil
import subprocess

import numpy as np

from radient.tasks.transforms.video_demux._base import VideoDemuxTransform


class FFmpegVideoDemuxTransform(VideoDemuxTransform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, data: str):
        """Extracts frames and audio snippets from a video file.
        """

        # Ensure that FFmpeg is installed and available from the command line.
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError(f"ffmpeg not found, try specifying 'method': 'default' in params")
        if not shutil.which("ffprobe"):
            raise FileNotFoundError(f"ffmpeg not found, try specifying 'method': 'default' in params")

        frames = {"data": [], "type": "image"}
        audios = {"data": [], "type": "audio"}
        video_path = data
        output_path = self._make_output_dir()

        # Grab video information using ffprobe.
        frame_info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries",
             "stream=r_frame_rate,nb_read_packets", "-of",
             "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True).stdout.split()
        frame_rate = eval(frame_info[0])
        frame_count = eval(frame_info[1])
        frame_interval = frame_rate * self._interval
        audio_info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of",
             "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True).stdout.split()

        for i, n in enumerate(np.arange(0, frame_count, frame_interval)):
            start_time = n / frame_rate

            # Extract frames.
            frame_path = str(output_path / f"frame_{i:04d}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start_time),
                            "-i", video_path, "-vframes", "1", frame_path])
            frames["data"].append(frame_path)

            # Extract audio.
            audio_path = str(output_path / f"audio_{i:04d}.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start_time),
                            "-i", video_path, "-t", str(self._interval),
                            "-q:a", "0", "-map", "a", audio_path])
            audios["data"].append(audio_path)

        return [frames, audios]
