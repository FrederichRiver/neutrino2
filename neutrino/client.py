import socket
import time


class SocketClient(object):
    def __init__(self, name='127.0.0.1', port=0) -> None:
        self.socket = socket.socket()
        if name is None:
            name = socket.gethostname()
        self.name = name
        forbid_port = [0, 21, 22, 23, 80, 8080]
        if port in forbid_port:
            self.port = 12001

    def connect(self):
        self.socket.connect((self.name, self.port))

    def send(self, msg: str):
        self.socket.send(str.encode(msg))

    def recieve(self) -> str:
        data = self.socket.recv(1024)
        return data.decode()

    def close(self):
        self.socket.close()


def test():
    print('func test')


def frame(func):
    event = SocketClient()
    event.connect()
    event.send('Q|test')
    dt = event.recieve()
    event.close()
    del event
    time.sleep(int(5))
    while True:
        func()
        event = SocketClient()
        event.connect()
        event.send(f'E|{func.__name__}')
        dt = event.recieve()
        print(dt)
        event.close()
        del event
        time.sleep(int(30))


frame(test)
