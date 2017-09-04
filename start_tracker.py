

from tracker.tracker_sender import Tracker

vision_address = ('224.5.23.2', 10044)

tracker = Tracker(vision_address)
tracker.start()
tracker.debug(('127.0.0.1', 20021))
