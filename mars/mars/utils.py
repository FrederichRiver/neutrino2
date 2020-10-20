#!/usr/bin/python3
import atexit
import os
import signal
import sys
import json
import logging
import pandas as pd
import datetime
import re
import functools
import psutil
from dev_global.env import LOG_FILE, TIME_FMT
from .task_server import SocketClient
import time


def frame_work(func, prog_name):
    deamon(prog_name)
    event = SocketClient()
    event.connect()
    event.send(f"Q|{func.__name__}")
    dt = event.recieve()
    event.close()
    del event
    time.sleep(int(dt))
    while True:
        func()
        event = SocketClient()
        event.connect()
        event.send(f'E|{func.__name__}')
        dt = event.recieve()
        event.close()
        del event
        time.sleep(int(dt))
    return 1


def interval_work(func, prog_name):
    deamon(prog_name)
    event = SocketClient()
    event.connect()
    event.send(f"Q|{func.__name__}")
    dt = event.recieve()
    event.close()
    del event
    time.sleep(1800)
    while True:
        func()
        event = SocketClient()
        event.connect()
        event.send(f'Q|{func.__name__}')
        dt = event.recieve()
        event.close()
        del event
        time.sleep(1800)
    return 1


def deamon(prog_name):
    # This is a daemon programe, which will start after
    # system booted.
    #
    # It is defined to start by rc.local.
    #
    # fork a sub process from father
    pid_file = f"/tmp/{prog_name}.pid"
    log_file = f"/opt/neutrino/{prog_name}.log"
    if os.path.exists(pid_file):
        raise RuntimeError(f"{prog_name} is already running")
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


def read_json(key, js_file):
    try:
        with open(js_file, 'r') as f:
            result = f.read()
            j = json.loads(result)
        item = j[key]
    except Exception:
        item = None
    return key, item


def drop_space(input_str):
    result = input_str.replace(' ', '')
    return result


def read_url(key, url_file):
    """
    It is a method base on read_json, returns a url.
    """
    _, url = read_json(key, url_file)
    return url


def record_base(text, level=logging.INFO):
    logging.basicConfig(filename=LOG_FILE,
                        level=logging.INFO,
                        filemode='a',
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    if level == logging.INFO:
        logging.info(text)
    elif level == logging.WARNING:
        logging.warn(text)
    elif level == logging.ERROR:
        logging.error(text)


INFO = functools.partial(record_base, level=logging.INFO)
ERROR = functools.partial(record_base, level=logging.ERROR)
WARN = functools.partial(record_base, level=logging.WARNING)


class Resource(object):
    def __init__(self):
        self.cpu = 0.0
        self.memory = 0.0
        self.period = 0.0

    def _query_info(self):
        mem = psutil.virtual_memory()
        self.memory = mem.percent
        self.cpu = psutil.cpu_percent(1)
        return self.cpu, self.memory

    def status(self):
        self._query_info()
        if self.memory < 85:
            return self.memory
        else:
            return 0

    def system_report(self):
        # Report system infomation.
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        MB = 1024*1024
        GB = 1024*MB
        sys_info = (
            f"<CPU>: {psutil.cpu_count()}\n"
            f"<Total Memory>: {round(mem.total/MB, 2)}MB\n"
            f"<Total Disk>: {round(disk.total/GB, 2)}GB")
        return sys_info


def trans(x):
    """
    Used for sql generating.
    float or int : value -> str / nan -> 0
    datetime     : value -> '2020-01-22' / nan -> NULL
    str          : value -> 'value'
    """
    if isinstance(x, int) or isinstance(x, float):
        if pd.isnull(x):
            return '0'
        else:
            return str(x)
    elif isinstance(x, datetime.datetime):
        if re.match(r"\d{4}\-\d{2}\-\d{2}", str(x)):
            return f"'{str(x)}'"
        else:
            return "NULL"
    elif isinstance(x, str):
        return f"'{str(x)}'"
    else:
        return f"'{str(x)}'"


def set_date_as_index(df):
    df['date'] = pd.to_datetime(df['date'], format=TIME_FMT)
    df.set_index('date', inplace=True)
    # exception 1, date index not exists.
    # exception 2, date data is not the date format.
    return df


def data_clean(df):
    for index, col in df.iteritems():
        try:
            if re.search('date', index):
                df[index] = pd.to_datetime(df[index])
            elif re.search('int', index):
                df[index] = pd.to_numeric(df[index])
            elif re.search('float', index):
                df[index] = pd.to_numeric(df[index])
            elif re.search('char', index):
                pass
            else:
                pass
        except Exception as e:
            ERROR("Error while data cleaning.")
            ERROR(e)
    return df


def str2number(in_str):
    import re
    if isinstance(in_str, str):
        in_str = in_str.replace(',', '')
        f = re.search(r'(\-|\+)?\d+(\.[0-9]+)?', in_str)
        d = re.match(r'\d{4}\-\d{2}\-\d{2}', in_str)
        if d:
            result = in_str
        elif f:
            # print(in_str)
            try:
                result = float(f[0])
            except Exception:
                result = 'NULL'
        else:
            result = None
    elif isinstance(in_str, int):
        result = in_str
    elif isinstance(in_str, float):
        result = in_str
    else:
        result = None
    return result
