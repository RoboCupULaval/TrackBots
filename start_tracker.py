import signal
from tracker.tracker import Tracker
from tracker.constants import TrackerConst
from tracker.debug.debug_tracker import Debug
from tracker.send_packet import send_packets, get_socket, send_udp


vision_address = ('224.5.23.2', 10006)
tracker = Tracker(vision_address)
tracker.start()

signal.signal(signal.SIGINT, lambda *args: tracker.stop())

debug = Debug(tracker, ('127.0.0.1', 20021))
debug.start()

sock = get_socket(TrackerConst.TRACKER_ADDRESS)
send_packets(sock,
             send_rate=30,
             send_func=send_udp,
             packet=lambda: tracker.track_frame,
             stop_condition=lambda: not tracker.is_running)


