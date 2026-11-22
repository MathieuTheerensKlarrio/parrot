"""Score a model against the segmented recordings in the recordings folder.

Each source recording is cut into the same feature windows that training uses
(see lib.wav.load_wav_data_from_srt), labelled by its .srt segmentation, and
every window is classified by the model.
"""
import os
import random
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config.config import BACKGROUND_LABEL, FEATURE_ENGINEERING_TYPE, MICROPHONE_SEPARATOR, RECORD_SECONDS, \
    SLIDING_WINDOW_AMOUNT, RATE
from lib.wav import load_wav_data_from_srt

PREDICTION_BATCH_SIZE = 512


def label_of_directory(directory_name: str) -> str:
    if MICROPHONE_SEPARATOR:
        return directory_name.split(MICROPHONE_SEPARATOR)[0]
    return directory_name


def pick_srt_file(shared_key: str, srt_files: List[str]) -> Optional[str]:
    """The segmentation training would use: .MANUAL.srt first, then the highest .vN.srt."""
    if shared_key + ".MANUAL.srt" in srt_files:
        return shared_key + ".MANUAL.srt"
    best_version, best_file = -1, None
    pattern = re.compile(re.escape(shared_key) + r"\.v(\d+)\.srt$")
    for srt_file in srt_files:
        match = pattern.match(srt_file)
        if match and int(match.group(1)) > best_version:
            best_version, best_file = int(match.group(1)), srt_file
    return best_file


def find_segmented_recordings(label_directory: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Pairs of (source wav, srt) in a label directory, plus the source files that have no usable srt."""
    source_directory = os.path.join(label_directory, "source")
    segments_directory = os.path.join(label_directory, "segments")
    if not os.path.isdir(source_directory) or not os.path.isdir(segments_directory):
        return [], []

    srt_files = [x for x in os.listdir(segments_directory) if x.endswith(".srt")]
    pairs, missing = [], []
    for source_file in sorted(os.listdir(source_directory)):
        if not source_file.endswith(".wav"):
            continue
        srt_file = pick_srt_file(source_file[:-len(".wav")], srt_files)
        if srt_file is None:
            missing.append(os.path.join(source_directory, source_file))
        else:
            pairs.append((os.path.join(source_directory, source_file), os.path.join(segments_directory, srt_file)))
    return pairs, missing


def predict_labels(classifier, windows: List, threshold: int) -> List[str]:
    """The winning class per window, or the background label when the winner is below the threshold percentage."""
    predicted = []
    classes = list(classifier.classes_)
    for start in range(0, len(windows), PREDICTION_BATCH_SIZE):
        probabilities = np.asarray(classifier.predict_proba(windows[start:start + PREDICTION_BATCH_SIZE]))
        for row in probabilities:
            winner = int(np.argmax(row))
            predicted.append(classes[winner] if row[winner] * 100 >= threshold else BACKGROUND_LABEL)
    return predicted


def settings_mismatches(classifier) -> List[str]:
    """Audio settings the model was trained with that differ from the current config."""
    current = {"RATE": RATE, "RECORD_SECONDS": RECORD_SECONDS, "SLIDING_WINDOW_AMOUNT": SLIDING_WINDOW_AMOUNT}
    return [f"{key}: model {classifier.get_setting(key, value)}, config {value}"
            for key, value in current.items() if classifier.get_setting(key, value) != value]


def evaluate_recordings(classifier, recordings_folder: str, threshold: int = 0, include_background: Optional[bool] = None,
                        progress: Optional[Callable[[str], None]] = None, seed: int = 0) -> Dict:
    """Classify every labelled window of the recordings for the labels the model knows.

    Background windows are only scored when the model has a background class; per recording
    at most as many background windows are sampled as it has labelled windows.

    Returns a dict with "true" and "predicted" label lists, "per_label" window counts
    per label, and "missing_srt" source files that could not be scored.
    """
    classes = list(classifier.classes_)
    if include_background is None:
        include_background = BACKGROUND_LABEL in classes
    feature_type = classifier.get_setting("FEATURE_ENGINEERING_TYPE", FEATURE_ENGINEERING_TYPE)
    rng = random.Random(seed)

    result = {"true": [], "predicted": [], "per_label": {}, "missing_srt": []}
    for directory_name in sorted(os.listdir(recordings_folder)):
        label = label_of_directory(directory_name)
        label_directory = os.path.join(recordings_folder, directory_name)
        if label == BACKGROUND_LABEL or label not in classes or not os.path.isdir(label_directory):
            continue

        pairs, missing = find_segmented_recordings(label_directory)
        result["missing_srt"].extend(missing)
        for source_file, srt_file in pairs:
            if progress is not None:
                progress(f"{directory_name}/{os.path.basename(source_file)}")
            windows = load_wav_data_from_srt(srt_file, source_file, feature_type, False)
            true_labels = [label] * len(windows)
            if include_background and len(windows) > 0:
                background = load_wav_data_from_srt(srt_file, source_file, feature_type, False, False, True)
                if len(background) > len(windows):
                    background = rng.sample(background, len(windows))
                windows = windows + background
                true_labels += [BACKGROUND_LABEL] * len(background)
            if len(windows) == 0:
                continue

            result["true"].extend(true_labels)
            result["predicted"].extend(predict_labels(classifier, windows, threshold))
            for true_label in set(true_labels):
                result["per_label"][true_label] = result["per_label"].get(true_label, 0) + true_labels.count(true_label)
    return result
