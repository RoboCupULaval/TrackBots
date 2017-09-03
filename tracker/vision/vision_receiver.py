import queue
import socket
import threading
from ipaddress import ip_address
from socketserver import ThreadingMixIn, UDPServer, BaseRequestHandler
import struct
import time

from tracker.field import Field
from tracker.observations import DetectionFrame, BallObservation, RobotObservation
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
        self.observations.queue.clear()
        print('Geometry packet received.')

    def get_udp_handler(self, observations, field):

        class ThreadedUDPRequestHandler(BaseRequestHandler):

            def handle(self):
                data = self.request[0]
                packet = SSL_WrapperPacket()
                packet.ParseFromString(data)

                if packet.HasField('detection'):

                    balls = []
                    for ball in packet.detection.balls:
                        ball_fields = VisionReceiver.parse_proto(ball)
                        balls.append(BallObservation(**ball_fields))

                    robots_blue = []
                    for robot in packet.detection.robots_blue:
                        robot_fields = VisionReceiver.parse_proto(robot)
                        robots_blue.append(RobotObservation(**robot_fields))

                    robots_yellow = []
                    for robot in packet.detection.robots_yellow:
                        robot_fields = VisionReceiver.parse_proto(robot)
                        robots_yellow.append(RobotObservation(**robot_fields))

                    frame_fields = VisionReceiver.parse_proto(packet.detection)
                    frame_fields['balls'] = balls
                    frame_fields['robots_blue'] = robots_blue
                    frame_fields['robots_yellow'] = robots_yellow

                    observations.put(DetectionFrame(**frame_fields))

                if packet.HasField('geometry'):
                    field.update(packet.geometry)

        return ThreadedUDPRequestHandler

    @staticmethod
    def parse_proto(proto_packet):
        return dict(map(lambda f: (f[0].name, f[1]), proto_packet.ListFields()))

    def get(self):
        return self.observations.get()

    def has_received_geometry(self):
        return False if self.field.geometry is None else True

    @property
    def is_ready(self):
        return self._is_ready
