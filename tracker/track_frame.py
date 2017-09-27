from collections import namedtuple

_track_fields = 'timestamp, balls, robots_blue, robots_yellow'
_robot_fields = 'id, pose, velocity'
_ball_fields = 'id, pose, velocity'

Ball = namedtuple('Ball', _ball_fields)
Robot = namedtuple('Robot', _robot_fields)
TrackFrame = namedtuple('TrackFrame', _track_fields)
