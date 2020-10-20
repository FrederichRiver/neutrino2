from mars.utils import frame_work
from venus.api_stock_event import event_download_stock_data
from dev_global.env import LOG_FILE
import sys
import os
import signal

PROG_NAME = 'event_download_stock_data'


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
            frame_work(event_download_stock_data, PROG_NAME)
    elif sys.argv[1] == 'stop':
        if os.path.exists(PID_FILE):
            sys.stdout.flush()
            with open(LOG_FILE, 'a') as write_null:
                os.dup2(write_null.fileno(), 1)
            with open(PID_FILE) as f:
                os.kill(int(f.read()), signal.SIGTERM)
        else:
            raise SystemExit(1)
    elif sys.argv[1] == 'test':
        event_download_stock_data()
    else:
        print('Unknown command {!r}'.format(sys.argv[1]))
        raise SystemExit(1)
