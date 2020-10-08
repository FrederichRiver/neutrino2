#!/usr/bin/python38
import atexit
import os
import signal
import sys
import time
import re


__version__ = (1, 0, 1)


def ticktick(pid_file, log_file):
    # This is a daemon programe, which will start after
    # system booted.
    #
    # It is defined to start by rc.local.
    #
    # fork a sub process from father
    if os.path.exists(pid_file):
        raise RuntimeError('Neutrino is already running')
    # the first fork.
    if os.fork() > 0:
        raise SystemExit(0)

    os.chdir('/')
    os.umask(0)
    os.setsid()
    # Second fork
    if os.fork() > 0:
        raise SystemExit(0)
    # Flush I/O buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # with open(log_file, 'rb', 0) as read_null:
    # os.dup2(read_null.fileno(), sys.stdin.fileno())
    with open(log_file, 'a') as write_null:
        # Redirect to 1 which means stdout
        os.dup2(write_null.fileno(), 1)
    with open(log_file, 'a') as error_null:
        # Redirect to 2 which means stderr
        os.dup2(error_null.fileno(), 2)
    if pid_file:
        with open(pid_file, 'w+') as f:
            f.write(str(os.getpid()))
        atexit.register(os.remove, pid_file)

    def sigterm_handler(signo, frame):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, sigterm_handler)


def main():
    PID_FILE = "/tmp/ticktick.pid"
    LOG_FILE = "/tmp/tick.log"
    ticktick(PID_FILE, LOG_FILE)
    while True:
        # pack files
        pack_log()
        time.sleep(86350)


def pack_log():
    source_path = '/opt/neutrino/'
    target_path = "/var/ftp/pub/log/"
    zip_list = []
    # get log file(s) in source path -> zip_list
    os.chdir(source_path)
    file_list = os.listdir(source_path)
    for f in file_list:
        if re.match(r'^neutrino_log-\d{4}-\d{2}-\d{2}.log', f):
            zip_list.append(f)
    compress_file_list = ' '.join(zip_list)
    # tar -zcf compress_file source_file(s)
    prefix = 'neutrino_log'
    zip_time = time.strftime('%Y-%m-%d')
    zip_file_name = f"{prefix}-{zip_time}.tar.gz"
    tar_cmd = f"tar -zcf {zip_file_name} {compress_file_list}"
    os.system(tar_cmd)
    # remove log files.
    for f in zip_list:
        os.system(f"rm {f}")
    os.system(f"mv {zip_file_name} {target_path}")


if __name__ == '__main__':
    # This is main function
    # Arguments format is like 'netrino args'
    # Neutrino receives args like start stop or other.
    PID_FILE = "/tmp/ticktick.pid"
    LOG_FILE = "/tmp/tick.log"
    if len(sys.argv) != 2:
        print("tick start|stop|help")
        raise SystemExit(1)
    if sys.argv[1] == 'start':
        main()
    elif sys.argv[1] == 'stop':
        if os.path.exists(PID_FILE):
            sys.stdout.flush()
            print("Neutrino is stopping.")
            with open(PID_FILE) as f:
                os.kill(int(f.read()), signal.SIGTERM)
        else:
            print("Neutrino is not running.")
            raise SystemExit(1)
    elif sys.argv[1] == 'help':
        print('help!')
    else:
        print('Unknown command {!r}'.format(sys.argv[1]))
        raise SystemExit(1)
