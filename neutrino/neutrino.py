#!/usr/bin/python38
import os
import signal
import sys
import time
from dev_global.env import LOG_FILE, PID_FILE, TASK_FILE, MANUAL
from polaris.mysql8 import mysqlBase, GLOBAL_HEADER
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from mars.task_manager import taskManager2
from threading import Thread
from mars.log_manager import info_log, error_log
from dev_global.basic import deamon


__version__ = '1.7.15'


PROG_NAME = 'Neutrino'


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
            info_log(f"{PROG_NAME} started with pid {os.getpid()}.")


def task_pipeline(taskfile=None, task_pipeline_name='Default'):
    # init task manager and main
    if not taskfile:
        raise FileNotFoundError(taskfile)
    mysql = mysqlBase(GLOBAL_HEADER)
    jobstores = {
        'default': SQLAlchemyJobStore(tablename=f"{task_pipeline_name}_task_sched", engine=mysql.engine)
            }
    task_manager = taskManager2(
        taskfile=taskfile,
        task_manager=task_pipeline_name,
        jobstores=jobstores,
        executors={'default': ThreadPoolExecutor(20)},
        job_defaults={'max_instance': 5})
    task_manager.start()
    info_log(f"{PROG_NAME} started with pid {os.getpid()}.")
    while True:
        task_manager.task_report()
        task_manager.task_solver.load_event()
        task_list = task_manager.check_task_list()
        task_manager.task_manage(task_list)
        time.sleep(300)


# @system_loop
def main(pid_file: str, log_file: str, task_file: str, prog_name: str):
    deamon(pid_file, log_file, prog_name)
    LM = Thread(target=logfile_monitor, args=(log_file,), name='neu_lm', daemon=True)
    LM.start()
    task_pipeline(task_file, prog_name)


if __name__ == '__main__':
    # This is main function
    # Arguments format is like 'netrino args'
    # Neutrino receives args like start stop or other.
    if len(sys.argv) != 2:
        print(f"{PROG_NAME} start|stop|help")
        raise SystemExit(1)
    if sys.argv[1] == 'start':
        main(PID_FILE, LOG_FILE, TASK_FILE, 'neutrino')
    elif sys.argv[1] == 'stop':
        if os.path.exists(PID_FILE):
            sys.stdout.flush()
            with open(LOG_FILE, 'a') as write_null:
                os.dup2(write_null.fileno(), 1)
                info_log(f"{PROG_NAME} is stopped.")
            with open(PID_FILE) as f:
                os.kill(int(f.read()), signal.SIGTERM)
        else:
            error_log(f"{PROG_NAME} is not running.")
            raise SystemExit(1)
    elif sys.argv[1] == 'clear':
        with open(LOG_FILE, 'w') as f:
            pass
    elif sys.argv[1] == 'help':
        os.system(f"cat {MANUAL}")
        # print_info(MANUAL)
    elif sys.argv[1] == 'log':
        os.system(f"cat {LOG_FILE}")
        # print_info(LOG_FILE)
    elif sys.argv[1] == 'version':
        print(__version__)
    else:
        print('Unknown command {!r}'.format(sys.argv[1]))
        raise SystemExit(1)
