import pickle
import socket
from tracker.tracker import Tracker
from tracker.constants import TrackerConst
import sched
import time
from tracker.debug.debug_tracker import Debug

vision_address = ('224.5.23.2', 10006)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.connect(TrackerConst.TRACKER_ADDRESS)

tracker = Tracker(vision_address)
tracker.start()

send_rate = 30

debug = Debug(tracker, ('127.0.0.1', 20021))
debug.start()


def send_loop():
    if tracker.is_running:
        sc.enter(1 / send_rate, 3, send_loop)
        packet = tracker.get_track_frame()
        try:
            sock.send(pickle.dumps(packet))
        except ConnectionRefusedError:
            pass


sc = sched.scheduler(time.time, time.sleep)
sc.enter(1 / send_rate, 3, send_loop)
sc.run()
