
from tracker.constants import TeamColor
from tracker.proto.messages_robocup_ssl_detection_pb2 import SSL_DetectionBall, SSL_DetectionRobot

import numpy as np


class Observation:

    def __init__(self, timestamp: float, camera_id: int, frame_number: int):
        assert(isinstance(camera_id, int))
        assert(isinstance(timestamp, float))
        assert(isinstance(frame_number, int) and frame_number >= 0)

        self.camera_id = camera_id
        self.timestamp = timestamp
        self.frame_number = frame_number


class BallObservation(Observation):

    def __init__(self, info: SSL_DetectionBall, timestamp: float, camera_id: int, frame_number: int):
        self.states = np.array([info.x, info.y]).T
        self.confidence = info.confidence
        super().__init__(timestamp, camera_id, frame_number)


class RobotObservation(Observation):

    def __init__(self, info: SSL_DetectionRobot, team_color: TeamColor, timestamp: float, camera_id: int, frame_number: int):

        assert(team_color in TeamColor)

        self.robot_id = info.robot_id
        self.team_color = team_color
        self.states = np.array([info.x, info.y, info.orientation])
        self.confidence = info.confidence
        super().__init__(timestamp, camera_id, frame_number)