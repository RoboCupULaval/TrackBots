import socket
import pickle
import time
import sched
import numpy as np
from tracker.debug.debug_command import DebugCommand
import threading


class Debug:

    def __init__(self, tracker, address):
        self.ui_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ui_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ui_server.connect(address)

        self.tracker = tracker
        self.tracker.logger.info('UI server create. ({}:{})'.format(*address))

        self._thread = threading.Thread(target=self.running_thread, daemon=True)

        self.debug_fps = 20
        self.ui_commands = []

    def start(self):
        self._thread.start()

    def add_robot_position_commands(self, pos, color=(0, 255, 0), color_angle=(0, 0, 0), radius=90):

        player_center = (pos[0], pos[1])
        data_circle = {'center': player_center,
                       'radius': radius,
                       'color': color,
                       'is_fill': True,
                       'timeout': 0.08}
        self.ui_commands.append(DebugCommand(3003, data_circle))

        end_point = np.array([pos[0], pos[1]]) + 90 * np.array([np.cos(pos[2]), np.sin(pos[2])])
        end_point = (end_point[0], end_point[1])
        data_line = {'start': player_center,
                     'end': end_point,
                     'color': color_angle,
                     'timeout': 0.08}
        self.ui_commands.append(DebugCommand(3001, data_line))

    def add_balls_position_commands(self, pos, color=(255, 127, 80)):
        player_center = (pos[0], pos[1])
        data_circle = {'center': player_center,
                       'radius': 150,
                       'color': color,
                       'is_fill': True,
                       'timeout': 0.06}
        self.ui_commands.append(DebugCommand(3003, data_circle))

    def send_ui_commands(self):
        for robot in self.tracker._yellow_team:
            if robot.last_observation is not None:
                pos_raw = robot.last_observation
                self.add_robot_position_commands(pos_raw, color=(255, 0, 0), radius=10, color_angle=(255, 0, 0))
            if robot.pose is not None:
                pos_filter = robot.pose
                self.add_robot_position_commands(pos_filter, color=(255, 255, 0))

        for robot in self.tracker._blue_team:
            if robot.last_observation is not None:
                pos_raw = robot.last_observation
                self.add_robot_position_commands(pos_raw, color=(255, 0, 0), radius=10, color_angle=(255, 0, 0))
            if robot.pose is not None:
                pos_filter = robot.pose
                self.add_robot_position_commands(pos_filter, color=(0, 0, 255))

        for ball in self.tracker._balls.considered_balls:
            if ball.pose is not None:
                pos_filter = ball.pose
                self.add_balls_position_commands(pos_filter, color=(255, 0, 0))

        for ball in self.tracker._balls:
            if ball.last_observation is not None:
                pos_raw = ball.last_observation
                self.add_balls_position_commands(pos_raw, color=(255, 255, 255))
            if ball.pose is not None:
                pos_filter = ball.pose
                self.add_balls_position_commands(pos_filter, color=(255, 100, 0))

        try:
            for cmd in self.ui_commands:
                self.ui_server.send(pickle.dumps(cmd.get_packet()))
        except ConnectionRefusedError:
            pass

            self.ui_commands.clear()

    def print_info(self):
        if 0.95 < time.time() % 1 < 1:
            print('Balls confidence:',
                  ' '.join('{:.1f}'.format(ball.confidence) for ball in self.tracker._balls.considered_balls),
                  'Balls: ',
                  ' '.join('{}'.format(id(ball)) for ball in self.tracker._balls))
            print(self.tracker.track_frame)

    def scheduled_loop(self, scheduler):

        if self.tracker.is_running:
            scheduler.enter(1 / self.debug_fps, 3, self.scheduled_loop, (scheduler,))
            self.send_ui_commands()
            self.print_info()

    def running_thread(self):
        sc = sched.scheduler(time.time, time.sleep)
        sc.enter(1 / self.debug_fps, 1, self.scheduled_loop, (sc,))
        sc.run()
