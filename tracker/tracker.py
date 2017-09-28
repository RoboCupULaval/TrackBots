
import logging
import threading
import time
from typing import Union
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
        self._thread = threading.Thread(target=self.tracker_main_loop)

        self.last_sending_time = time.time()

        self.vision_receiver = VisionReceiver(vision_address)
        self.logger.info('VisionReceiver created. ({}:{})'.format(*vision_address))

        self._blue_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self._yellow_team = [RobotFilter() for _ in range(Tracker.MAX_ROBOT_PER_TEAM)]
        self._balls = MultiBallService(Tracker.MAX_BALL_ON_FIELD)

        self._current_timestamp = None

    @property
    def is_running(self):
        return self._thread.is_alive()

    @property
    def current_timestamp(self):
        return self._current_timestamp

    def start(self):
        self.vision_receiver.start()
        self._thread.start()

    def tracker_main_loop(self):
        while not self.thread_terminate.is_set():

            detection_frame = self.vision_receiver.get()
            self._current_timestamp = detection_frame.t_capture

            for robot_obs in detection_frame.robots_blue:
                obs_state = np.array([robot_obs.x, robot_obs.y, robot_obs.orientation])
                self._blue_team[robot_obs.robot_id].update(obs_state, detection_frame.t_capture)
                self._blue_team[robot_obs.robot_id].predict(Tracker.STATE_PREDICTION_TIME)

            for robot_obs in detection_frame.robots_yellow:
                obs_state = np.array([robot_obs.x, robot_obs.y, robot_obs.orientation])
                self._yellow_team[robot_obs.robot_id].update(obs_state, detection_frame.t_capture)
                self._yellow_team[robot_obs.robot_id].predict(Tracker.STATE_PREDICTION_TIME)

            for ball_obs in detection_frame.balls:
                self._balls.update_with_observation(ball_obs, detection_frame.t_capture)

            self.remove_undetected_robot()

    def remove_undetected_robot(self):
        for robot in self._yellow_team + self._blue_team:
            if robot.last_t_capture + Tracker.MAX_UNDETECTED_DELAY < self.current_timestamp:
                robot.is_active = False

    @property
    def track_frame(self) -> TrackFrame:
        track_fields = dict()
        track_fields['timestamp'] = self.current_timestamp
        track_fields['robots_blue'] = self.blue_team
        track_fields['robots_yellow'] = self.yellow_team
        track_fields['balls'] = self.balls

        return TrackFrame(**track_fields)

    @property
    def balls(self) -> Ball:
        return Tracker.format_list(self._balls, Ball)

    @property
    def blue_team(self) -> Robot:
        return Tracker.format_list(self._blue_team, Robot)

    @property
    def yellow_team(self) -> Robot:
        return Tracker.format_list(self._yellow_team, Robot)

    @staticmethod
    def format_list(entities: list, entity_format: Union[Robot, Ball]):
        format_list = []
        for entity_id, entity in enumerate(entities):
            if entity.is_active:
                fields = dict()
                fields['pose'] = tuple(entity.pose)
                fields['velocity'] = tuple(entity.velocity)
                fields['id'] = entity_id
                format_list.append(entity_format(**fields))
        return format_list

    def stop(self):
        self.thread_terminate.set()
        self._thread.join()
        self.thread_terminate.clear()
