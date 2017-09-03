
import socket
from tracker.field import Field
from tracker.observations import DetectionFrame, BallObservation, RobotObservation
from tracker.proto.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket


class VisionReceiver:

    def __init__(self, address):

        self.host = address[0]
        self.port = address[1]

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(address)

        self._is_ready = False

        self.last_detection_frame = None
        self.field = Field()

    def start(self):
        print('Waiting for geometry from {}:{}'.format(self.host, self.port))
        while not self.has_received_geometry():
            self._listen()

        self._is_ready = True
        print('Geometry packet received.')

    def _listen(self):

        data, _ = self.sock.recvfrom(2048)
        packet = SSL_WrapperPacket()
        packet.ParseFromString(data)

        if packet.HasField('detection'):
            self.parse_detection(packet)

        if packet.HasField('geometry'):
            self.field.update(packet.geometry)

    def parse_detection(self, packet):
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

        self.last_detection_frame = DetectionFrame(**frame_fields)

    @staticmethod
    def parse_proto(proto_packet):
        return dict(map(lambda f: (f[0].name, f[1]), proto_packet.ListFields()))

    def get(self):
        self._listen()
        return self.last_detection_frame

    def has_received_geometry(self):
        return False if self.field.geometry is None else True

    @property
    def is_ready(self):
        return self._is_ready
