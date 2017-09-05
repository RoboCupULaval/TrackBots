# Under MIT License, see LICENSE.txt
import unittest

from tracker.tracker_sender import Tracker


class TrackerTestCase(unittest.TestCase):

    def setUp(self):
        address = ('127.0.0.1', 10031)
        self.T = Tracker(address)

    def assertChoseChose(self):
        pass


class TrackerXXXTest(TrackerTestCase):

    def test_chose(self):
        pass
