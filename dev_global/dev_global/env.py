#!/usr/bin/python38
from polaris.mysql8 import mysqlHeader

"""
global environment varibles

"""

PYTHON_VERSION = 3.8
LOCAL_TIME_ZONE = 'Beijing'
TIME_FMT = '%Y-%m-%d'
LOG_TIME_FMT = "%Y-%m-%d %H:%M:%S"

GITHUB_URL = "https://github.com/FrederichRiver/neutrino2"
EMAIL = "hezhiyuan_tju@163.com"

"""
if os.getenv('SERVER') == 'MARS':
    ROOT_PATH = '/root/'
    SOFT_PATH = '/opt/neutrino/'
else:
    ROOT_PATH = '/home/friederich/Dev/neutrino2/'
    SOFT_PATH = '/home/friederich/Dev/neutrino2/'
"""
ROOT_PATH = '/root/'
SOFT_PATH = '/opt/neutrino/'

encode = 'wAKO0tFJ8ZH38RW4WseZnQ=='

LOG_FILE = SOFT_PATH + 'neutrino.log'
PID_FILE = '/tmp/neutrino.pid'
TASK_FILE = SOFT_PATH + 'config/task.json'
CONF_FILE = SOFT_PATH + 'config/conf.json'
HEAD_FILE = SOFT_PATH + 'config/header.json'
COOKIE_FILE = SOFT_PATH + 'config/cookie.json'
SQL_FILE = SOFT_PATH + 'config/sql.json'
MANUAL = SOFT_PATH + 'config/Neutrino'


GLOBAL_HEADER = mysqlHeader('stock', 'stock2020', 'stock')
VIEWER_HEADER = mysqlHeader('view', 'view2020', 'stock')
