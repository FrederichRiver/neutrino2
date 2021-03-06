#!/usr/bin/python3
import os
import sys
from mars.database_manager import (
    event_mysql_backup, event_initial_database,
    change_stock_template_definition)
__version__ = '1.0.3'


def neutrino_install():
    dest = '/opt/neutrino/'
    obj = '/root/ftp/package/'
    files = [
        'neutrino.py', 'manage_tool.py',
        'config/conf.json', 'config/Neutrino', 'config/task.json',
        'config/cookie.json']
    for fi in files:
        obj_fi = obj + fi
        dest_fi = dest + fi
        if os.path.isfile(obj_fi):
            # print(obj_fi)
            cmd = "cp -u -v %s %s" % (obj_fi, dest_fi)
            os.system(cmd)
        elif os.path.isdir(fi):
            pass  # solve_dir(obj_fi+'/')
        else:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("maint init|backup")
        raise SystemExit(1)
    if sys.argv[1] == "init":
        event_initial_database()
    elif sys.argv[1] == "backup":
        event_mysql_backup()
    elif sys.argv[1] == "install":
        neutrino_install()
    elif sys.argv[1] == "test":
        change_stock_template_definition()
