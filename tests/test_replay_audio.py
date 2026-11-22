import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.replay_audio import list_recordings, new_recording_folder, sorted_chunks


def touch(path, mtime=None):
    with open(path, "wb"):
        pass
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestReplayAudio(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_chunks_are_sorted_by_time_not_name(self):
        for name in ["10.000.wav", "2.500.wav", "0.030.wav", "notes.txt", "not-a-time.wav"]:
            touch(os.path.join(self.folder, name))
        self.assertEqual([os.path.basename(x) for x in sorted_chunks(self.folder)],
                         ["0.030.wav", "2.500.wav", "10.000.wav"])

    def test_recordings_are_listed_newest_first_with_loose_files(self):
        now = time.time()
        for name, age in [("old", 300), ("new", 10)]:
            os.makedirs(os.path.join(self.folder, name))
            touch(os.path.join(self.folder, name, "0.030.wav"))
            touch(os.path.join(self.folder, name, "4.250.wav"))
            os.utime(os.path.join(self.folder, name), (now - age, now - age))
        os.makedirs(os.path.join(self.folder, "empty"))
        touch(os.path.join(self.folder, "1.000.wav"), now - 100)

        recordings = list_recordings(self.folder)
        self.assertEqual([r.name for r in recordings],
                         ["new", "unsorted files in " + os.path.basename(self.folder), "old"])
        self.assertEqual(recordings[0].seconds, 4.25)
        self.assertEqual(recordings[1].folder, self.folder)

    def test_missing_folder_has_no_recordings(self):
        self.assertEqual(list_recordings(os.path.join(self.folder, "missing")), [])

    def test_new_recording_folders_never_collide(self):
        first = new_recording_folder(self.folder)
        second = new_recording_folder(self.folder)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.isdir(first) and os.path.isdir(second))


if __name__ == "__main__":
    unittest.main()
