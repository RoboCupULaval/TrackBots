
import logging
import signal
import threading
import time

import numpy as np

from tracker.filters.robot_kalman_filter import RobotFilter
from tracker.multiballservice import MultiBallService
from tracker.vision.vision_receiver import VisionReceiver
from tracker.constants import TrackerConst
from tracker.track_frame import TrackFrame, Robot, Ball

logging.basicConfig(level=logging.INFO, format='%(message)s')


class Tracker:

    TRACKER_ADDRESS = TrackerConst.TRACKER_ADDRESS

    MAX_BALL_ON_FIELD = TrackerConst.MAX_BALL_ON_FIELD
    MAX_ROBOT_PER_TEAM = TrackerConst.MAX_ROBOT_PER_TEAM

    STATE_PREDICTION_TIME = TrackerConst.STATE_PREDICTION_TIME

    MAX_UNDETECTED_DELAY = TrackerConst.MAX_UNDETECTED_DELAY

    def __init__(self, vision_address):

        self.logger = logging.getLogger('Tracker')
        
        self.thread_terminate = threading.Event()
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._thread = threading.Thread(target=self.tracker_main_loop)

        self.last_sending_time = time.time()

        self.vision_receiver = VisionReceiver(vision_address)
        self.logger.info('VisionReceiver created. ({}:{})'.format(*vision_address))

        self.blue_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self.yellow_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self.balls = MultiBallService(Tracker.MAX_BALL_ON_FIELD)

        self.current_timestamp = None

    @property
    def is_running(self):
        return self._thread.is_alive()

    def start(self):
        self.vision_receiver.start()
        self._thread.start()

    def tracker_main_loop(self):
        while not self.thread_terminate.is_set():

            detection_frame = self.vision_receiver.get()
            self.current_timestamp = detection_frame.t_capture

            for robot_obs in detection_frame.robots_blue:
                obs_state = np.array([robot_obs.x, robot_obs.y, robot_obs.orientation])
                self.blue_team[robot_obs.robot_id].update(obs_state, detection_frame.t_capture)
                self.blue_team[robot_obs.robot_id].predict(Tracker.STATE_PREDICTION_TIME)

            for robot_obs in detection_frame.robots_yellow:
                obs_state = np.array([robot_obs.x, robot_obs.y, robot_obs.orientation])
                self.yellow_team[robot_obs.robot_id].update(obs_state, detection_frame.t_capture)
                self.yellow_team[robot_obs.robot_id].predict(Tracker.STATE_PREDICTION_TIME)

            for ball_obs in detection_frame.balls:
                self.balls.update_with_observation(ball_obs, detection_frame.t_capture)

            self.remove_undetected_robot()

    def remove_undetected_robot(self):
        for robot in self.yellow_team + self.blue_team:
            if robot.last_t_capture + Tracker.MAX_UNDETECTED_DELAY < self.current_timestamp:
                robot.is_active = False

    def get_track_frame(self):

        track_fields = dict()
        track_fields['timestamp'] = self.current_timestamp

        track_fields['robots_blue'] = []
        for robot_id, robot in enumerate(self.blue_team):
            if robot.is_active:
                robot_fields = Tracker.get_fields(robot, robot_id)
                track_fields['robots_blue'].append(Robot(**robot_fields))

        track_fields['robots_yellow'] = []
        for robot_id, robot in enumerate(self.yellow_team):
            if robot.is_active:
                robot_fields = Tracker.get_fields(robot, robot_id)
                track_fields['robots_yellow'].append(Robot(**robot_fields))

        track_fields['balls'] = []
        for idx, ball in enumerate(self.balls):
            ball_fields = Tracker.get_fields(ball, idx)
            track_fields['balls'].append(Ball(**ball_fields))

        return TrackFrame(**track_fields)

    @staticmethod
    def get_fields(entity, idx):
        fields = dict()
        fields['pose'] = tuple(entity.get_pose())
        fields['velocity'] = tuple(entity.get_velocity())
        fields['id'] = idx
        return fields

    def stop(self):
        self.thread_terminate.set()
        self._thread.join()
        self.thread_terminate.clear()

    def _sigint_handler(self, *args):
        self.stop()
