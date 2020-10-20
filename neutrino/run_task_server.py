import sys
import os
import signal
from mars.task_server import SocketServer
from mars.utils import deamon
from dev_global.env import LOG_FILE
from polaris.mysql8 import GLOBAL_HEADER


__version__ = (1, 0, 0)


PROG_NAME = 'socket_server'


def socket_server():
    serv = SocketServer(GLOBAL_HEADER)
    serv.run()


if __name__ == '__main__':
    # This is main function
    # Arguments format is like 'netrino args'
    # Neutrino receives args like start stop or other.
    PID_FILE = f"/tmp/{PROG_NAME}.pid"
    if len(sys.argv) != 2:
        print(f"{PROG_NAME} start|stop|help")
        raise SystemExit(1)
    if sys.argv[1] == 'start':
        if os.path.exists(PID_FILE):
            print(f"{PROG_NAME} is already running.")
        else:
            deamon(PROG_NAME)
            socket_server()
    elif sys.argv[1] == 'stop':
        if os.path.exists(PID_FILE):
            sys.stdout.flush()
            with open(LOG_FILE, 'a') as write_null:
                os.dup2(write_null.fileno(), 1)
            with open(PID_FILE) as f:
                os.kill(int(f.read()), signal.SIGTERM)
        else:
            raise SystemExit(1)
    else:
        print('Unknown command {!r}'.format(sys.argv[1]))
        raise SystemExit(1)
