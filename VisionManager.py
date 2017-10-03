from typing import Tuple

from .tracker.tracker import Tracker
from .tracker.debug.uidebug_sender import UIDebugSender
from multiprocessing import Pipe, Connection


class VisionManager:
    def __init__(self, connection: Connection):

        if not isinstance(connection, Connection):
            raise TypeError("Wrong type of pipe!")
        self.connection = connection
        self.tracker = None
        self.uidebug_sender = None

    def initialize_tracker(self, vision_address: Tuple[str, int]):
        self.tracker = Tracker(vision_address = ('224.5.23.2', 10006))

    def initalize_uidebug_sender(self, vision_address: Tuple[str, int]):
        # to be called after the tracker as been initialized please
        self.uidebug_sender = UIDebugSender(self.tracker, vision_address)

    def update_commands(self):
        # todo implement
        pass

    def wait_for_addresses(self):

        while (1):
            recv = self.connection.recv()
            print(recv)