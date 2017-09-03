
import logging
import pickle
import sched
import signal
import socket
import threading
import time

import numpy as np

from tracker.debug.debug_command import DebugCommand
from tracker.filters.ball_kalman_filter import BallFilter
from tracker.filters.robot_kalman_filter import RobotFilter
from tracker.proto.messages_tracker_wrapper_pb2 import TRACKER_WrapperPacket
from tracker.vision.vision_receiver import VisionReceiver
from tracker.constants import TrackerConst


class Tracker:

    TRACKER_ADDRESS = TrackerConst.TRACKER_ADDRESS
    SEND_DELAY = TrackerConst.SEND_DELAY

    MAX_BALL_ON_FIELD = TrackerConst.MAX_BALL_ON_FIELD
    MAX_ROBOT_PER_TEAM = TrackerConst.MAX_ROBOT_PER_TEAM

    BALL_CONFIDENCE_THRESHOLD = TrackerConst.BALL_CONFIDENCE_THRESHOLD
    BALL_SEPARATION_THRESHOLD = TrackerConst.BALL_SEPARATION_THRESHOLD

    STATE_PREDICTION_TIME = TrackerConst.STATE_PREDICTION_TIME

    MAX_UNDETECTED_DELAY = TrackerConst.MAX_UNDETECTED_DELAY

    def __init__(self, vision_address):

        self.logger = logging.getLogger('TrackerServer')
        logging.basicConfig(level=logging.INFO, format='%(message)s')

        self.thread_terminate = threading.Event()
        signal.signal(signal.SIGINT, self._sigint_handler)
        self.tracker_thread = threading.Thread(target=self.tracker_main_loop)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.connect(Tracker.TRACKER_ADDRESS)

        self.last_sending_time = time.time()

        self.vision_server = VisionReceiver(vision_address)
        self.logger.info('VisionReceiver created. ({}:{})'.format(*vision_address))

        self.debug_terminate = threading.Event()

        self.blue_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self.yellow_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self.balls = []
        self.considered_balls = []

        self.current_timestamp = None

    def start(self):
        self.vision_server.start()
        while not self.vision_server.is_ready:
            time.sleep(0.1)
        self.tracker_thread.start()

    def tracker_main_loop(self):
        while not self.thread_terminate.is_set():

            detection_frame = self.vision_server.get()
            self.current_timestamp = detection_frame.t_capture

            for robot_obs in detection_frame.robots_blue:
                obs_state = np.array([robot_obs.x, robot_obs.y, robot_obs.orientation])
                self.blue_team[robot_obs.robot_id].update(obs_state, detection_frame.t_capture)

            for robot_obs in detection_frame.robots_yellow:
                obs_state = np.array([robot_obs.x, robot_obs.y, robot_obs.orientation])
                self.yellow_team[robot_obs.robot_id].update(obs_state, detection_frame.t_capture)

            for ball_obs in detection_frame.balls:
                obs_state = np.array([ball_obs.x, ball_obs.y])
                closest_ball = self.find_closest_ball_to_observation(obs_state)

                if closest_ball is None:  # No ball or every balls are too far.
                    self.considered_balls.append(BallFilter())
                    self.considered_balls[-1].update(obs_state, detection_frame.t_capture)
                    self.logger.info('New ball detected: {}.'.format(id(self.considered_balls[-1])))
                else:
                    closest_ball.update(obs_state, detection_frame.t_capture)

            self.remove_undetected_robots()

            self.update_balls_confidence()
            self.select_best_balls()

            self.predict_next_states()

            if time.time() - self.last_sending_time > Tracker.SEND_DELAY:
                self.last_sending_time = time.time()
                self.send_packet()

    def predict_next_states(self):
        for robot in self.yellow_team + self.blue_team:
            robot.predict(Tracker.STATE_PREDICTION_TIME)

        for ball in self.considered_balls:
            ball.predict(Tracker.STATE_PREDICTION_TIME)

    def remove_undetected_robots(self):
        for robot in self.yellow_team + self.blue_team:
            if robot.last_t_capture + Tracker.MAX_UNDETECTED_DELAY < self.current_timestamp:
                robot.is_active = False

    def find_closest_ball_to_observation(self, obs):

        position_differences = self.compute_distances_ball_to_observation(obs)

        closest_ball = None
        if position_differences is not None:
            min_diff = float(min(position_differences))
            if min_diff < Tracker.BALL_SEPARATION_THRESHOLD:
                closest_ball_idx = position_differences.index(min_diff)
                closest_ball = self.considered_balls[closest_ball_idx]

        return closest_ball

    def compute_distances_ball_to_observation(self, obs_state):
        position_differences = []
        for ball in self.considered_balls:
            if ball.last_prediction is not None:
                position_differences.append(float(np.linalg.norm(ball.get_position() - obs_state)))
            elif ball.last_observation is not None:  # If we never predict the state, we still need to compare it
                position_differences.append(float(np.linalg.norm(ball.last_observation - obs_state)))
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
            self.sock.send(packet.SerializeToString())
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

        ui_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ui_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ui_server.connect(Tracker.TRACKER_ADDRESS)

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
                    pos_raw = robot.last_observation
                    add_robot_position_commands(pos_raw, color=(255, 0, 0), radius=10, color_angle=(255, 0, 0))
                if robot.get_position() is not None:
                    pos_filter = robot.get_position()
                    add_robot_position_commands(pos_filter, color=(255, 255, 0))

            for robot in self.blue_team:
                if robot.last_observation is not None:
                    pos_raw = robot.last_observation
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
                    pos_raw = ball.last_observation
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
