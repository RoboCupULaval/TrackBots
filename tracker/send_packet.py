import sched
import time
import pickle
import socket


def send_udp(s, packet):
    try:
        s.send(pickle.dumps(packet))
    except ConnectionRefusedError:
        pass


def send_packets(channel, *,
                 send_rate=30,
                 packet=lambda: None,
                 send_func=lambda *args: None,
                 stop_condition=lambda: False):

    def send_loop():
        if not stop_condition():
            sc.enter(1 / send_rate, 3, send_loop)
            send_func(channel, packet())

    sc = sched.scheduler(time.time, time.sleep)
    sc.enter(1 / send_rate, 3, send_loop)
    sc.run()


def get_socket(address):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.connect(address)
    return sock
