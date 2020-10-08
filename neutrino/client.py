import socket


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


event = SocketClient()
event.connect()
event.send('E|test')
event.recieve()
event.close()
