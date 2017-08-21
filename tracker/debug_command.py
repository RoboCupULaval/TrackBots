# Under MIT License, see LICENSE.txt

SENDER_NAME = "TRACKER"


class DebugCommand:

    def __init__(self, p_type_, p_data, p_link=None, p_version="1.0"):
        self._packet = dict()
        self._packet['name'] = SENDER_NAME
        self._packet['version'] = p_version
        self._packet['type'] = p_type_
        self._packet['link'] = p_link
        self._packet['data'] = p_data

    def get_packet(self):
        return self._packet
