
import time


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
        self.last_update = time.time()
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
