#!/usr/bin/python38
import atexit
import os
import signal
import sys
import time
from dev_global.env import LOG_FILE, PID_FILE, TASK_FILE, MANUAL, GLOBAL_HEADER
from polaris.mysql8 import mysqlBase
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from mars.task_manager import taskManager2
from threading import Thread
from mars.log_manager import log_decorator2, system_loop, info_log, error_log


__version__ = '1.6.14'


@log_decorator2
def neutrino(pid_file, log_file):
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


def logfile_monitor(log_file):
    # A parallel programe which monitoring the log file.
    # If log file is not exists, it will create one and
    # relocalize the file.
    while True:
        if os.path.exists(log_file):
            time.sleep(10)
        else:
            create_file = open(log_file, 'a')
            create_file.close()
            with open(log_file, 'a') as write_null:
                os.dup2(write_null.fileno(), 1)
            with open(log_file, 'a') as error_null:
                os.dup2(error_null.fileno(), 2)
            info_log(f"Neutrino started with pid {os.getpid()}.")


@log_decorator2
def neptune_pipeline(taskfile=None):
    # init task manager and main
    if not taskfile:
        raise FileNotFoundError(taskfile)
    mysql = mysqlBase(GLOBAL_HEADER)
    jobstores = {
        'default': SQLAlchemyJobStore(tablename='apscheduler_jobs', engine=mysql.engine)
            }
    Neptune = taskManager2(
        taskfile=taskfile,
        task_manager='Neptune',
        jobstores=jobstores,
        executors={'default': ThreadPoolExecutor(20)},
        job_defaults={'max_instance': 5})
    Neptune.start()
    while True:
        Neptune.task_solver.load_event()
        task_list = Neptune.check_task_list()
        Neptune.task_manage(task_list)
        time.sleep(300)


def print_info(info_file):
    infotext = ''
    with open(info_file) as r:
        infotext = r.read()
    print(infotext)


@system_loop
def main():
    neutrino(PID_FILE, LOG_FILE)
    info_log(f"Neutrino id is {os.getpid()}.")
    lm = Thread(target=logfile_monitor, args=(LOG_FILE,), name='lm', daemon=True)
    lm.start()
    neptune_pipeline(TASK_FILE)


if __name__ == '__main__':
    # This is main function
    # Arguments format is like 'netrino args'
    # Neutrino receives args like start stop or other.
    if len(sys.argv) != 2:
        print("neutrino start|stop|help")
        raise SystemExit(1)
    if sys.argv[1] == 'start':
        main()
    elif sys.argv[1] == 'stop':
        if os.path.exists(PID_FILE):
            sys.stdout.flush()
            with open(LOG_FILE, 'a') as write_null:
                os.dup2(write_null.fileno(), 1)
                info_log("Neutrino is stopped.")
            with open(PID_FILE) as f:
                os.kill(int(f.read()), signal.SIGTERM)
        else:
            error_log("Neutrino is not running.")
            raise SystemExit(1)
    elif sys.argv[1] == 'clear':
        with open(LOG_FILE, 'w') as f:
            pass
    elif sys.argv[1] == 'help':
        print_info(MANUAL)
    elif sys.argv[1] == 'log':
        print_info(LOG_FILE)
    elif sys.argv[1] == 'version':
        print(__version__)
    else:
        print('Unknown command {!r}'.format(sys.argv[1]))
        raise SystemExit(1)
