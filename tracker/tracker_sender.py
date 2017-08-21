
import logging
import pickle
import queue
import sched
import signal
import socket
import struct
import threading
import time
from abc import abstractmethod
from enum import Enum
from ipaddress import ip_address
from socketserver import BaseRequestHandler
from socketserver import ThreadingMixIn, UDPServer

import numpy as np

from tracker.proto.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket
from tracker.proto.messages_robocup_ssl_detection_pb2 import SSL_DetectionBall, SSL_DetectionRobot

from tracker.debug_command import DebugCommand
from tracker.proto.messages_tracker_wrapper_pb2 import TRACKER_WrapperPacket


class TeamColor(Enum):
    def __str__(self):
        return 'blue' if self == TeamColor.BLUE else 'yellow'
    YELLOW = 0
    BLUE = 1


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
        self.ball_id = 0
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


class KalmanFilter:

    def __init__(self):

        self.is_active = False
        self.last_timestamp = 0
        self.last_observation = None
        self.last_prediction = None

        self.F = self.transition_model()
        self.H = self.observation_model()

        self.state_number = np.size(self.F, 1)
        self.observable_state = np.sum(self.H)

        self.B = self.control_input_model()

        self.R = self.observation_covariance()
        self.Q = self.process_covariance()
        self.P = self.initial_state_covariance()

        self.S = np.zeros((self.observable_state, self.observable_state))
        self.x = np.zeros(self.state_number)
        self.u = np.zeros(self.observable_state)

    @abstractmethod
    def get_position(self):
        pass

    @abstractmethod
    def get_velocity(self):
        pass

    @abstractmethod
    def transition_model(self, dt):
        pass

    @abstractmethod
    def control_input_model(self, dt):
        self.B = np.zeros((self.state_number, self.observable_state))

    @abstractmethod
    def observation_model(self):
        pass

    @abstractmethod
    def initial_state_covariance(self):
        pass

    @abstractmethod
    def process_covariance(self):
        pass

    @abstractmethod
    def observation_covariance(self):
        pass

    @abstractmethod
    def update(self, obs) -> None:
        pass

    @abstractmethod
    def predict(self, dt):
        pass


class BallFilter(KalmanFilter):

    INITIAL_BALL_CONFIDENCE = 20

    INITIAL_STATE_COVARIANCE = 1000

    POSITION_PROCESS_COVARIANCE = 1
    VELOCITY_PROCESS_COVARIANCE = 10

    POSITION_OBSERVATION_COVARIANCE = 2

    def __init__(self):
        self.confidence = BallFilter.INITIAL_BALL_CONFIDENCE
        super().__init__()

    def get_position(self):
        if not self.is_active:
            return None
        else:
            return np.array([self.x[0], self.x[2]]).flatten()

    def get_velocity(self):
        if not self.is_active:
            return None
        else:
            return np.array([self.x[1], self.x[3]]).flatten()

    def transition_model(self, dt=0):
        return np.array([[1, dt, 0,  0],   # Position x
                         [0,  1, 0,  0],   # Speed x
                         [0,  0, 1, dt],   # Position y
                         [0,  0, 0,  1]])  # Speed y

    def observation_model(self):
        return np.array([[1, 0, 0, 0],   # Position x
                         [0, 0, 1, 0]])  # Position y

    def control_input_model(self, dt=0):
        return np.zeros((self.state_number, self.observable_state))

    def initial_state_covariance(self):
        return BallFilter.INITIAL_STATE_COVARIANCE * np.eye(self.state_number)

    def process_covariance(self):
        return np.diag([BallFilter.POSITION_PROCESS_COVARIANCE,
                        BallFilter.VELOCITY_PROCESS_COVARIANCE,
                        BallFilter.POSITION_PROCESS_COVARIANCE,
                        BallFilter.VELOCITY_PROCESS_COVARIANCE])

    def observation_covariance(self):
        return BallFilter.POSITION_OBSERVATION_COVARIANCE * np.eye(self.observable_state)

    def update(self, obs):
        self.is_active = True

        dt = obs.timestamp - self.last_timestamp
        if dt < 0:
            return
        self.last_timestamp = obs.timestamp

        self.last_observation = obs

        self.F = self.transition_model(dt)
        y = obs.states - np.dot(self.H, self.x)
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.state_number) - np.dot(K, self.H)), self.P)

    def predict(self, dt=0):
        self.F = self.transition_model(dt)
        self.B = self.control_input_model(dt)
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def increase_confidence(self):
        self.confidence += 1
        if self.confidence > 100:
            self.confidence = 100

    def decrease_confidence(self):
        self.confidence *= 0.995
        if self.confidence < 0:
            self.confidence = 0


class RobotFilter(KalmanFilter):

    INITIAL_BALL_CONFIDENCE = 20

    INITIAL_STATE_COVARIANCE = 1000

    POSITION_PROCESS_COVARIANCE = 1
    VELOCITY_PROCESS_COVARIANCE = 10
    ORIENTATION_PROCESS_COVARIANCE = 0.05
    ANGULAR_VELOCITY_PROCESS_COVARIANCE = 0.5

    POSITION_OBSERVATION_COVARIANCE = 2
    ORIENTATION_OBSERVATION_COVARIANCE = 0.05

    def __init__(self):
        self.tau = [0.3, 0.3, 0.3]
        super().__init__()

    def get_position(self):
        if not self.is_active:
            return None
        else:
            return np.array([self.x[0], self.x[2], self.x[4]]).flatten()

    def get_velocity(self):
        if not self.is_active:
            return None
        else:
            return np.array([self.x[1], self.x[3], self.x[5]]).flatten()

    def get_orientation(self):
        if not self.is_active:
            return None
        else:
            return self.x[4]

    def transition_model(self, dt=0):
        return np.array([[1,                    dt, 0,  0,                    0,  0],   # Position x
                         [0,  1 - dt / self.tau[0], 0,  0,                    0,  0],   # Speed x
                         [0,                     0, 1, dt,                    0,  0],   # Position y
                         [0,                     0, 0,  1 - dt / self.tau[1], 0,  0],   # Speed y
                         [0,                     0, 0,  0,                    1, dt],   # Position Theta
                         [0,                     0, 0,  0,                    0,  1 - dt / self.tau[2]]])  # Speed Theta

    def observation_model(self):
        return np.array([[1, 0, 0, 0, 0, 0],   # Position x
                         [0, 0, 1, 0, 0, 0],   # Position y
                         [0, 0, 0, 0, 1, 0]])  # Orientation

    def control_input_model(self, dt=0):
        return np.array([[0,                0,                0],  # Position x
                         [dt / self.tau[0], 0,                0],  # Speed x
                         [0,                0,                0],  # Position y
                         [0,                dt / self.tau[1], 0],  # Speed y
                         [0,                0,                0],  # Position Theta
                         [0,                0,                dt / self.tau[2]]])  # Speed Theta

    def initial_state_covariance(self):
        return RobotFilter.INITIAL_STATE_COVARIANCE * np.eye(self.state_number)

    def process_covariance(self):
        return np.diag([RobotFilter.POSITION_PROCESS_COVARIANCE,
                        RobotFilter.VELOCITY_PROCESS_COVARIANCE,
                        RobotFilter.POSITION_PROCESS_COVARIANCE,
                        RobotFilter.VELOCITY_PROCESS_COVARIANCE,
                        RobotFilter.ORIENTATION_PROCESS_COVARIANCE,
                        RobotFilter.ANGULAR_VELOCITY_PROCESS_COVARIANCE])

    def observation_covariance(self):
        return np.diag([RobotFilter.POSITION_OBSERVATION_COVARIANCE,
                        RobotFilter.POSITION_OBSERVATION_COVARIANCE,
                        RobotFilter.ORIENTATION_OBSERVATION_COVARIANCE])

    def update(self, obs):

        self.is_active = True

        dt = obs.timestamp - self.last_timestamp
        if dt < 0:
            return
        self.last_timestamp = obs.timestamp
        self.last_observation = obs

        self.F = self.transition_model(dt)
        y = obs.states - np.dot(self.H, self.x)
        y[2] = RobotFilter.wrap_to_pi(y[2])
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.state_number) - np.dot(K, self.H)), self.P)

    def predict(self, dt=0):
        if not self.is_active:
            return
        self.F = self.transition_model(dt)
        self.B = self.control_input_model(dt)
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


class Field:

    def __init__(self):
        self.last_update = None

        self.geometry = None
        self._field = None

        self._line_width = None
        self._field_length = None
        self._field_width = None
        self._boundary_width = None
        self._referee_width = None
        self._goal_width = None
        self._goal_depth = None
        self._goal_wall_width = None
        self._center_circle_radius = None
        self._defense_radius = None
        self._defense_stretch = None
        self._free_kick_from_defense_dist = None
        self._penalty_spot_from_field_line_dist = None
        self._penalty_line_from_spot_dist = None

    def update(self, geometry_packet):
        self.geometry = geometry_packet
        self._field = geometry_packet.field
        # self.parse_field_packet()

    def parse_field_packet(self):  # This is legacy, not good!
        self._line_width = self._field.line_width
        self._field_length = self._field.field_length
        self._field_width = self._field.field_width
        self._boundary_width = self._field.boundary_width
        self._referee_width = self._field.referee_width
        self._goal_width = self._field.goal_width
        self._goal_depth = self._field.goal_depth
        self._goal_wall_width = self._field.goal_wall_width
        self._center_circle_radius = self._field.center_circle_radius
        self._defense_radius = self._field.defense_radius
        self._defense_stretch = self._field.defense_stretch
        self._free_kick_from_defense_dist = self._field.free_kick_from_defense_dist
        self._penalty_spot_from_field_line_dist = self._field.penalty_spot_from_field_line_dist
        self._penalty_line_from_spot_dist = self._field.penalty_line_from_spot_dist

    def is_point_inside_field(self, point):
        if self._field is not None:
            if abs(point[0]) > self._field_length/2 or abs(point[1]) > self._field_width/2:
                return True
            else:
                return False
        else:
            return None


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


class Tracker:

    TRACKER_HOST = '127.0.0.1'
    TRACKER_PORT = 21111
    SEND_DELAY = 0.02

    MAX_BALL_ON_FIELD = 1
    MAX_ROBOT_PER_TEAM = 12

    BALL_CONFIDENCE_THRESHOLD = 1
    BALL_SEPARATION_THRESHOLD = 500

    STATE_PREDICTION_TIME = 0.1

    MAX_UNDETECTED_DELAY = 2

    def __init__(self, vision_host_ip, vision_port_number):

        self.logger = logging.getLogger('TrackerServer')
        logging.basicConfig(level=logging.INFO, format='%(message)s')

        self.thread_terminate = threading.Event()
        signal.signal(signal.SIGINT, self._sigint_handler)
        self.tracker_thread = threading.Thread(target=self.tracker_main_loop)

        self.server = udp_socket(Tracker.TRACKER_HOST, Tracker.TRACKER_PORT)
        self.last_sending_time = time.time()

        self.vision_server = VisionReceiver(vision_host_ip, vision_port_number)
        self.logger.info('VisionReceiver created. ({}:{})'.format(vision_host_ip, vision_port_number))

        self.debug_terminate = threading.Event()

        self.blue_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self.yellow_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self.balls = []
        self.considered_balls = []

        self.current_timestamp = None

    def start(self):
        self.vision_server.start()
        while not self.vision_server.is_ready():
            time.sleep(0.1)
        self.tracker_thread.start()

    def tracker_main_loop(self):
        while not self.thread_terminate.is_set():

            obs = self.vision_server.get()
            self.current_timestamp = obs.timestamp

            if type(obs) is RobotObservation:
                self.update_teams_with_observation(obs)
            elif type(obs) is BallObservation:
                self.update_balls_with_observation(obs)

            self.remove_undetected_robots()

            self.update_balls_confidence()
            self.select_best_balls()

            self.predict_next_states()

            if time.time() - self.last_sending_time > Tracker.SEND_DELAY:
                self.last_sending_time = time.time()
                self.send_packet()

            time.sleep(0)

    def predict_next_states(self):
        for robot in self.yellow_team + self.blue_team:
            robot.predict(Tracker.STATE_PREDICTION_TIME)

        for ball in self.considered_balls:
            ball.predict(Tracker.STATE_PREDICTION_TIME)

    def remove_undetected_robots(self):
        for robot in self.yellow_team + self.blue_team:
            if robot.last_timestamp + Tracker.MAX_UNDETECTED_DELAY < self.current_timestamp:
                robot.is_active = False

    def update_teams_with_observation(self, obs):
        if obs.team_color is TeamColor.BLUE:
            self.blue_team[obs.robot_id].update(obs)
        else:
            self.yellow_team[obs.robot_id].update(obs)

    def update_balls_with_observation(self, obs):

        closest_ball = self.find_closest_ball_to_observation(obs)

        if closest_ball is None:  # No ball or every balls are too far.
            self.considered_balls.append(BallFilter())
            self.considered_balls[-1].update(obs)
            self.logger.info('New ball detected: {}.'.format(id(self.considered_balls[-1])))
        else:
            closest_ball.update(obs)
            closest_ball.increase_confidence()

        # remove ball far outside field

    def find_closest_ball_to_observation(self, obs):

        position_differences = self.compute_distances_ball_to_observation(obs)

        closest_ball = None
        if position_differences is not None:
            min_diff = float(min(position_differences))
            if min_diff < Tracker.BALL_SEPARATION_THRESHOLD:
                closest_ball_idx = position_differences.index(min_diff)
                closest_ball = self.considered_balls[closest_ball_idx]

        return closest_ball

    def compute_distances_ball_to_observation(self, obs):
        position_differences = []
        for ball in self.considered_balls:
            if ball.last_prediction is not None:
                position_differences.append(float(np.linalg.norm(ball.get_position() - obs.states)))
            elif ball.last_observation is not None:  # If we never predict the state, we still need to compare it
                position_differences.append(float(np.linalg.norm(ball.last_observation.states - obs.states)))
            else:  # This should never happens if ball are updated when create
                position_differences.append(float('inf'))

        if not position_differences:
            position_differences = None
        elif len(position_differences) == 1 and position_differences[0] == float('inf'):
            position_differences = None

        return position_differences

    def select_best_balls(self):
        if len(self.considered_balls) > 0:
            self.considered_balls.sort(key=lambda x: x.confidence, reverse=True)
            max_ball = min(Tracker.MAX_BALL_ON_FIELD, len(self.considered_balls))
            self.balls = self.considered_balls[0:max_ball]
        else:
            self.balls.clear()

    def update_balls_confidence(self):
        for ball in self.considered_balls:
            ball.decrease_confidence()
            if ball.confidence < Tracker.BALL_CONFIDENCE_THRESHOLD:
                self.considered_balls.remove(ball)
                self.logger.info('Removing ball {}.'.format(id(ball)))

    def generate_packet(self):

        wrapper_packet = TRACKER_WrapperPacket()
        
        detection_packet = wrapper_packet.detection
        detection_packet.timestamp = self.current_timestamp

        active_blue_robot = [robot for robot in self.blue_team if robot.is_active]
        active_yellow_robot = [robot for robot in self.yellow_team if robot.is_active]

        for idx, robot in enumerate(active_blue_robot):
            robot_position = robot.get_position()
            robot_velocity = robot.get_velocity()

            blue_robot = detection_packet.blue_team.add()

            blue_robot.confidence = 100
            blue_robot.robot_id = idx
            blue_robot.x = float(robot_position[0])
            blue_robot.y = float(robot_position[1])
            blue_robot.vx = float(robot_velocity[0])
            blue_robot.vy = float(robot_velocity[1])
            blue_robot.orientation = float(robot_position[2])
            blue_robot.angular_speed = float(robot_velocity[2])

        for idx, robot in enumerate(active_yellow_robot):
            robot_position = robot.get_position()
            robot_velocity = robot.get_velocity()

            yellow_robot = detection_packet.yellow_team.add()

            yellow_robot.confidence = 100
            yellow_robot.robot_id = idx
            yellow_robot.x = float(robot_position[0])
            yellow_robot.y = float(robot_position[1])
            yellow_robot.vx = float(robot_velocity[0])
            yellow_robot.vy = float(robot_velocity[1])
            yellow_robot.orientation = float(robot_position[2])
            yellow_robot.angular_speed = float(robot_velocity[2])

        for idx, ball in enumerate(self.balls):
            ball_position = ball.get_position()
            ball_velocity = ball.get_velocity()

            ball_packet = detection_packet.balls.add()
            ball_packet.confidence = ball.confidence
            ball_packet.ball_id = idx
            ball_packet.x = float(ball_position[0])
            ball_packet.y = float(ball_position[1])
            ball_packet.vx = float(ball_velocity[0])
            ball_packet.vy = float(ball_velocity[1])
        
        return wrapper_packet

    def send_packet(self):
        
        packet = self.generate_packet()
        
        try:
            self.server.send(packet.SerializeToString())
        except ConnectionRefusedError:
            pass

    def stop(self):
        self.thread_terminate.set()
        self.tracker_thread.join()
        self.thread_terminate.clear()
        self.debug_terminate.set()

    def _sigint_handler(self, *args):
        self.stop()

    def debug(self, host, port):

        ui_server = udp_socket(host, port)
        self.logger.info('UI server create. ({}:{})'.format(host, port))

        debug_fps = 20
        ui_commands = []

        def add_robot_position_commands(pos, color=(0, 255, 0), color_angle=(0, 0, 0), radius=90):

            player_center = (pos[0], pos[1])
            data_circle = {'center': player_center,
                           'radius': radius,
                           'color': color,
                           'is_fill': True,
                           'timeout': 0.08}
            ui_commands.append(DebugCommand(3003, data_circle))

            end_point = np.array([pos[0], pos[1]]) + 90 * np.array([np.cos(pos[2]), np.sin(pos[2])])
            end_point = (end_point[0], end_point[1])
            data_line = {'start': player_center,
                         'end': end_point,
                         'color': color_angle,
                         'timeout': 0.08}
            ui_commands.append(DebugCommand(3001, data_line))

        def add_balls_position_commands(pos, color=(255, 127, 80)):
            player_center = (pos[0], pos[1])
            data_circle = {'center': player_center,
                           'radius': 150,
                           'color': color,
                           'is_fill': True,
                           'timeout': 0.06}
            ui_commands.append(DebugCommand(3003, data_circle))

        def send_ui_commands():
            for robot in self.yellow_team:
                if robot.last_observation is not None:
                    pos_raw = robot.last_observation.states
                    add_robot_position_commands(pos_raw, color=(255, 0, 0), radius=10, color_angle=(255, 0, 0))
                if robot.get_position() is not None:
                    pos_filter = robot.get_position()
                    add_robot_position_commands(pos_filter, color=(255, 255, 0))

            for robot in self.blue_team:
                if robot.last_observation is not None:
                    pos_raw = robot.last_observation.states
                    add_robot_position_commands(pos_raw, color=(255, 0, 0), radius=10, color_angle=(255, 0, 0))
                if robot.get_position() is not None:
                    pos_filter = robot.get_position()
                    add_robot_position_commands(pos_filter, color=(0, 0, 255))

            for ball in self.considered_balls:
                if ball.get_position() is not None:
                    pos_filter = ball.get_position()
                    add_balls_position_commands(pos_filter, color=(255, 0, 0))

            for ball in self.balls:
                if ball.last_observation is not None:
                    pos_raw = ball.last_observation.states
                    add_balls_position_commands(pos_raw, color=(255, 255, 255))
                if ball.get_position() is not None:
                    pos_filter = ball.get_position()
                    add_balls_position_commands(pos_filter, color=(255, 100, 0))

            try:
                for cmd in ui_commands:
                    ui_server.send(pickle.dumps(cmd.get_packet()))
            except ConnectionRefusedError:
                pass

            ui_commands.clear()

        def print_info():
            if 0.95 < time.time() % 1 < 1:
                print('Balls confidence:',
                      ' '.join('{:.1f}'.format(ball.confidence) for ball in self.considered_balls),
                      'Balls: ',
                      ' '.join('{}'.format(id(ball)) for ball in self.balls))

        def scheduled_loop(scheduler):

            if not self.debug_terminate.is_set():
                scheduler.enter(1/debug_fps, 3, scheduled_loop, (scheduler,))
                send_ui_commands()
                print_info()

        sc = sched.scheduler(time.time, time.sleep)
        sc.enter(1/debug_fps, 1, scheduled_loop, (sc,))
        sc.run()


def udp_socket(host, port):
    skt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection_info = (host, port)
    skt.connect(connection_info)
    return skt

if __name__ == '__main__':

    vision_host = '224.5.23.2'
    vision_port = 10026

    tracker = Tracker(vision_host, vision_port)
    tracker.start()
    tracker.debug('127.0.0.1', 20021)
