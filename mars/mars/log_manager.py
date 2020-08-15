#!/usr/bin/python38
import os
import re
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from dev_global.env import LOG_FILE
from functools import wraps


__all__ = ['event_pack_tick_data', ]


# logging.basicConfig(filename=LOG_FILE)
# step 1: create a logger
neulog = logging.getLogger()
# step 2: set logger level
neulog.setLevel(logging.INFO)
# step 3: create a handler
Time_Handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
# step 4: set handler level
Time_Handler.setLevel(logging.INFO)
# step 5: create log format
LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
# step 6: add log format to handler
neu_format = logging.Formatter(LOG_FORMAT)
Time_Handler.setFormatter(neu_format)
# step 7: add handler to logger


def infoLog(msg: str):
    neulog.addHandler(Time_Handler)
    neulog.info(msg)
    neulog.handlers.pop()


def errorLog(msg: str):
    neulog.addHandler(Time_Handler)
    neulog.error(msg)
    neulog.handlers.pop()


"""
neu_format = logging.basicConfig(
                level=logging.INFO, format=LOG_FORMAT,
                datefmt="%Y-%m-%d %H:%M:%S", filename=LOG_FILE)
"""


def log_decorator(func):
    """
    decorate function with return value.
    """
    @wraps(func)
    def wrapper(*args, **kv):
        try:
            result = func(*args, **kv)
        except Exception as e:
            neulog.addHandler(Time_Handler)
            neulog.error(f"<Module> {func.__name__} <Message> {e}")
            neulog.handlers.pop()
            result = None
        return result
    return wrapper


def log_decorator2(func):
    """
    decorate function without return value.
    """
    @wraps(func)
    def wrapper(*args, **kv):
        try:
            func(*args, **kv)
        except Exception as e:
            neulog.addHandler(Time_Handler)
            neulog.error(f"<Module> {func.__name__} <Message> {e}")
            neulog.handlers.pop()
        return func
    return wrapper


def system_loop(func):
    @wraps(func)
    def wrapper(*args, **kv):
        try:
            func(*args, **kv)
        except Exception as e:
            neulog.error(e)
            raise SystemExit(1)
        return func
    return wrapper


def info_log(text):
    logging.basicConfig(filename=LOG_FILE,
                        level=logging.INFO,
                        filemode='a',
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.info(text)


def error_log(text):
    logging.basicConfig(filename=LOG_FILE,
                        level=logging.ERROR,
                        filemode='a',
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.error(text)


class filePack(object):
    def __init__(self):
        if os.environ.get('LOGNAME') == 'friederich':
            self.path = '/home/friederich/Downloads/tmp/'
            self.dest_path = '/home/friederich/Downloads/neutrino/'
        else:
            self.path = '/root/download/'
            self.dest_path = '/root/ftp/'
        self.flag_list = []

    def get_file_flag(self):
        """
        recognite file name like SH000300_20200501.csv
        """
        flag_list = os.listdir(self.path)
        temp_flag_list = []
        for flag in flag_list[:5]:
            result = re.match(r'^(\w{2}\d{6}\_)(\d{8})', flag)
            if result:
                temp_flag_list.append(result[2])
        self.flag_list = list(set(temp_flag_list))

    def package(self, flag):
        result = os.listdir(self.path)
        pack_list = []
        for file_name in result:
            file_pattern = re.compile(r"^\w{2}\d{6}\_" + flag)
            if re.match(file_pattern, file_name):
                pack_list.append(file_name)
        if pack_list:
            pack_file = ' '.join(pack_list)
            os.chdir(self.path)
            os.system('pwd')
            os.system(f"tar -czvf stock_data_{flag}.tar.gz {pack_file}")
            os.system(f"cp stock_data_{flag}.tar.gz {self.dest_path}")
        for data_file in pack_list:
            os.system(f"rm {data_file}")


def event_pack_tick_data():
    from jupiter.log_manager import filePack
    event = filePack()
    event.get_file_flag()
    for flag in event.flag_list:
        event.package(flag)


if __name__ == "__main__":
    infoLog('test')
