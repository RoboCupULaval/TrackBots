
from tracker.tracker_sender import Tracker

vision_host = '224.5.23.2'
vision_port = 10031

tracker = Tracker(vision_host, vision_port)
tracker.start()
tracker.debug('127.0.0.1', 20021)