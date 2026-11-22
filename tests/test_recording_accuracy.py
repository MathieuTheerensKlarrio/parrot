import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config.config import BACKGROUND_LABEL, CURRENT_VERSION
from lib.recording_accuracy import evaluate_recordings, pick_srt_file, find_segmented_recordings

FIXTURES = os.path.join(ROOT, "tests", "fixtures")


class FixedClassifier:
    """Always predicts the same probabilities, whatever the input."""

    def __init__(self, classes, probabilities):
        self.classes_ = classes
        self.probabilities = probabilities
        self.rows_seen = 0

    def predict_proba(self, data):
        self.rows_seen += len(data)
        return np.tile(self.probabilities, (len(data), 1))

    def get_setting(self, key, default):
        return default


class TestPickSrtFile(unittest.TestCase):
    def test_manual_wins(self):
        self.assertEqual(pick_srt_file("a", ["a.v2.srt", "a.MANUAL.srt", "a.v3.srt"]), "a.MANUAL.srt")

    def test_highest_version_wins(self):
        self.assertEqual(pick_srt_file("a", ["a.v2.srt", "a.v10.srt", "a.v3.srt"]), "a.v10.srt")

    def test_misnamed_or_other_files_are_ignored(self):
        self.assertIsNone(pick_srt_file("a", ["a_v3.srt", "ab.v3.srt"]))


class TestEvaluateRecordings(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        label_dir = os.path.join(self.folder, "pop")
        os.makedirs(os.path.join(label_dir, "source"))
        os.makedirs(os.path.join(label_dir, "segments"))
        shutil.copy(os.path.join(FIXTURES, "recordings", "pop", "source", "pop.wav"),
                    os.path.join(label_dir, "source", "pop.wav"))
        shutil.copy(os.path.join(FIXTURES, "segmentation", "pop.srt"),
                    os.path.join(label_dir, "segments", f"pop.v{CURRENT_VERSION}.srt"))
        # A source file without segmentation and a label the model does not know
        shutil.copy(os.path.join(FIXTURES, "recordings", "pop", "source", "pop.wav"),
                    os.path.join(label_dir, "source", "unsegmented.wav"))
        os.makedirs(os.path.join(self.folder, "unknown", "source"))

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_scores_pop_windows_and_balanced_background(self):
        classifier = FixedClassifier(["pop", BACKGROUND_LABEL], [0.9, 0.1])
        result = evaluate_recordings(classifier, self.folder)

        pop_windows = result["per_label"]["pop"]
        self.assertGreater(pop_windows, 0)
        self.assertLessEqual(result["per_label"][BACKGROUND_LABEL], pop_windows)
        self.assertEqual(len(result["true"]), len(result["predicted"]))
        self.assertEqual(classifier.rows_seen, len(result["true"]))
        self.assertEqual(set(result["predicted"]), {"pop"})
        self.assertEqual(result["missing_srt"], [os.path.join(self.folder, "pop", "source", "unsegmented.wav")])

    def test_below_threshold_counts_as_background(self):
        classifier = FixedClassifier(["pop", BACKGROUND_LABEL], [0.6, 0.4])
        result = evaluate_recordings(classifier, self.folder, threshold=70)
        self.assertEqual(set(result["predicted"]), {BACKGROUND_LABEL})

    def test_no_background_windows_without_background_class(self):
        classifier = FixedClassifier(["pop", "other"], [0.9, 0.1])
        result = evaluate_recordings(classifier, self.folder)
        self.assertEqual(set(result["true"]), {"pop"})

    def test_nothing_to_score(self):
        classifier = FixedClassifier(["other", BACKGROUND_LABEL], [0.9, 0.1])
        result = evaluate_recordings(classifier, self.folder)
        self.assertEqual(result["true"], [])


class TestFindSegmentedRecordings(unittest.TestCase):
    def test_folder_without_source_or_segments(self):
        folder = tempfile.mkdtemp()
        try:
            self.assertEqual(find_segmented_recordings(folder), ([], []))
        finally:
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
