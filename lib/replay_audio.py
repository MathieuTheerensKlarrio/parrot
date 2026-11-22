"""Audio recordings made by the analysis menu.

Every new recording is stored as a folder of short .wav chunks inside the replay
audio folder, each chunk named after the seconds since the recording started
(e.g. 1.234.wav). Chunks saved directly in the replay audio folder, by play mode
or by older Parrot versions, are listed as one extra recording.
"""
import os
import time
from typing import List, NamedTuple


class AudioRecording(NamedTuple):
    name: str
    folder: str
    chunks: List[str]  # chunk paths, sorted by time

    @property
    def seconds(self) -> float:
        return chunk_time(self.chunks[-1]) if self.chunks else 0.0


def chunk_time(path: str) -> float:
    return float(os.path.basename(path)[:-len(".wav")])


def sorted_chunks(folder: str) -> List[str]:
    """The .wav chunks in a folder, ordered by their timestamp; other files are ignored."""
    chunks = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not file.endswith(".wav") or not os.path.isfile(path):
            continue
        try:
            chunks.append((chunk_time(path), path))
        except ValueError:
            continue
    return [path for _, path in sorted(chunks)]


def list_recordings(audio_folder: str) -> List[AudioRecording]:
    """All recordings with at least one chunk, newest first."""
    if not os.path.isdir(audio_folder):
        return []

    recordings = []
    for name in os.listdir(audio_folder):
        folder = os.path.join(audio_folder, name)
        if os.path.isdir(folder):
            chunks = sorted_chunks(folder)
            if chunks:
                recordings.append((os.path.getmtime(folder), AudioRecording(name, folder, chunks)))

    loose_chunks = sorted_chunks(audio_folder)
    if loose_chunks:
        recordings.append((os.path.getmtime(loose_chunks[-1]),
                           AudioRecording("unsorted files in " + os.path.basename(audio_folder), audio_folder, loose_chunks)))

    return [recording for _, recording in sorted(recordings, key=lambda x: x[0], reverse=True)]


def new_recording_folder(audio_folder: str) -> str:
    """Create an empty folder for a new recording, named after the current date and time."""
    base = os.path.join(audio_folder, time.strftime("%Y-%m-%d_%H-%M-%S"))
    folder, suffix = base, 2
    while os.path.exists(folder):
        folder = base + "_" + str(suffix)
        suffix += 1
    os.makedirs(folder)
    return folder
