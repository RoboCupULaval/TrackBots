
import queue
import socket
import struct
import threading
import time
from ipaddress import ip_address
from socketserver import BaseRequestHandler
from socketserver import ThreadingMixIn, UDPServer

from tracker.proto.messages_tracker_wrapper_pb2 import TRACKER_WrapperPacket


class TrackerReceiver(ThreadingMixIn, UDPServer):

    allow_reuse_address = True

    def __init__(self, host_ip, port_number):

        self.host = host_ip
        self.port = port_number

        self.running_thread = threading.Thread(target=self.serve_forever)
        self.running_thread.daemon = True

        self.packet_list = queue.Queue()
        handler = self.get_udp_handler(self.packet_list)
        super().__init__(('', port_number), handler)

        if ip_address(host_ip).is_multicast:
            self.socket.setsockopt(socket.IPPROTO_IP,
                                   socket.IP_ADD_MEMBERSHIP,
                                   struct.pack("=4sl", socket.inet_aton(host_ip), socket.INADDR_ANY))

    def start(self):
        self.running_thread.start()

    def get_udp_handler(self, packet_list):

        class ThreadedUDPRequestHandler(BaseRequestHandler):

            def handle(self):
                data = self.request[0]
                packet = TRACKER_WrapperPacket()
                packet.ParseFromString(data)
                packet_list.put(packet)

        return ThreadedUDPRequestHandler

    def get(self):
        return self.packet_list.get()

    def clear_queue(self):
        with self.packet_list.mutex:
            self.packet_list.queue.clear()


if __name__ == '__main__':

    tracker_host = '224.5.23.2'
    tracker_port = 21111

    tracker_receiver = TrackerReceiver(tracker_host, tracker_port)
    tracker_receiver.start()

    while True:
        print(tracker_receiver.get())
        time.sleep(0.05)
