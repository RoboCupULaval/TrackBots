
from tracker.tracker_receiver import TrackerReceiver
import time

tracker_host = '224.5.23.2'
tracker_port = 21111

tracker_receiver = TrackerReceiver(tracker_host, tracker_port)
tracker_receiver.start()

while True:
    print(tracker_receiver.get())
    time.sleep(0.05)
