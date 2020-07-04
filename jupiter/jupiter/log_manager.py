#!/usr/bin/python38
import os
import re
import logging
from dev_global.env import LOG_FILE
from functools import wraps


__all__ = ['event_pack_tick_data', ]

test_file = '/home/friederich/Dev/test.log'
LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", filename=test_file)
logger = logging.getLogger()


def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kv):
        try:
            result = func(*args, **kv)
        except Exception as e:
            logger.error(e)
        return result
    return wrapper


@log_decorator
def user_test(x: str):
    print(x)
    raise TypeError("TEST")


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
    import sys
    print(sys.version)
    user_test(1)
