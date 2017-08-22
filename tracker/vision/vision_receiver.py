import queue
import socket
import threading
from ipaddress import ip_address
from socketserver import ThreadingMixIn, UDPServer, BaseRequestHandler
import struct
import time

from tracker.constants import TeamColor
from tracker.field import Field
from tracker.observations import BallObservation, RobotObservation
from tracker.proto.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket


class VisionReceiver(ThreadingMixIn, UDPServer):

    allow_reuse_address = True

    def __init__(self, host_ip, port_number):

        self.host = host_ip
        self.port = port_number

        self._is_ready = False

        self.running_thread = threading.Thread(target=self.serve_forever)
        self.running_thread.daemon = True

        self.field = Field()
        self.observations = queue.Queue()

        handler = self.get_udp_handler(self.observations, self.field)
        super().__init__(('', port_number), handler)

        if ip_address(host_ip).is_multicast:
            self.socket.setsockopt(socket.IPPROTO_IP,
                                   socket.IP_ADD_MEMBERSHIP,
                                   struct.pack("=4sl", socket.inet_aton(host_ip), socket.INADDR_ANY))

    def start(self):
        self.running_thread.start()

        waiting_time = time.time()
        while not self.has_received_geometry():
            if time.time() - waiting_time > 1:
                print('Waiting for geometry from {}:{}'.format(self.host, self.port))
                waiting_time = time.time()
            time.sleep(0)

        self._is_ready = True
        self.clear_queue()
        print('Geometry packet received.')

    def get_udp_handler(self, observations, field):

        class ThreadedUDPRequestHandler(BaseRequestHandler):

            def handle(self):
                data = self.request[0]
                packet = SSL_WrapperPacket()
                packet.ParseFromString(data)

                if packet.HasField('detection'):
                    timestamp = packet.detection.t_capture
                    camera_id = packet.detection.camera_id
                    frame_number = packet.detection.frame_number

                    for ball_info in packet.detection.balls:
                        observations.put(BallObservation(ball_info, timestamp, camera_id, frame_number))

                    for robot_info in packet.detection.robots_blue:
                        observations.put(RobotObservation(robot_info, TeamColor.BLUE,
                                                          timestamp, camera_id, frame_number))

                    for robot_info in packet.detection.robots_yellow:
                        observations.put(RobotObservation(robot_info, TeamColor.YELLOW,
                                                          timestamp, camera_id, frame_number))

                if packet.HasField('geometry'):
                    field.update(packet.geometry)
                    field.last_update = time.time()

        return ThreadedUDPRequestHandler

    def get(self):
        return self.observations.get()

    def clear_queue(self):
        with self.observations.mutex:
            self.observations.queue.clear()

    def has_received_geometry(self):
        return False if self.field.geometry is None else True

    def is_ready(self):
        return self._is_ready
