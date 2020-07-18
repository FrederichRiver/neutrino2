#!/usr/bin/python38
# -*- coding: utf-8 -*-

import imp
import json
import re
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from mars.log_manager import log_decorator, log_decorator2
# modules loaded into module list
import venus.api_stock_event
import taurus.nlp_event
import saturn.finance_event
import saturn.balance_analysis
import mars.database_manager
import mars.log_manager
import mars.mail_manager

__version__ = '1.2.7'


def decoration_print(func):
    """
    decorater of print function.
    """
    def wrap_func():
        print('+-' * 15)
        func()
        print('+-' * 15)
    return wrap_func


class taskManager2(BackgroundScheduler):
    """
    Task manager is a object to manage tasks.
    It will run tasks according to task.json file.
    It will auto load modules without reboot system.
    Process : load_event -> load_task_file
    """
    def __init__(self, taskfile=None, task_manager_name=None, gconfig={}, **options):
        super(BackgroundScheduler, self).__init__(gconfig=gconfig, **options)
        self.task_manager_name = task_manager_name
        self.start_time = datetime.datetime.now()
        if taskfile:
            self.taskfile = taskfile
            self.task_solver = taskSolver(taskfile)
            self._task_list = []
        else:
            # if task file is not found.
            raise FileNotFoundError(taskfile)

    def __str__(self):
        runtime = datetime.datetime.now() - self.start_time
        h, m = timedelta_convert(runtime)
        return f"<Task manager ({self.task_manager_name}) has running for {h}:{str(m).zfill(2)}:00>"

    def _update_task_list(self):
        self._task_list = self.get_jobs()

    def check_task_list(self):
        temp_task_list = self.load_task_list()
        self._update_task_list()
        for task in temp_task_list:
            task.flag = 'add'
            for old_task in self._task_list:
                if task.name == old_task.name:
                    if task.trigger != old_task.trigger:
                        task.flag = 'changed'
                    elif task.func != old_task.func:
                        task.flag = 'changed'
                    else:
                        task.flag = 'no change'
        return temp_task_list

    def task_manage(self, task_list):
        for task in task_list:
            if task.flag == 'add':
                self.add_job(task.func, trigger=task.trigger, id=task.name)
            elif task.flag == 'changed':
                self.reschedule_job(task.name, trigger=task.trigger)

    def load_task_list(self):
        with open(self.taskfile, 'r') as f:
            jsdata = json.load(f)
        task_list = []
        for task_data in jsdata:
            task = self.task_solver.task_resolve(task_data)
            if task:
                task_list.append(task)
        return task_list

    @decoration_print
    def task_report(self):
        self.print_jobs()


class taskBase(object):
    def __init__(self, task_name, func, trigger):
        self.name = task_name
        self.func = func
        self.trigger = trigger
        self.flag = None

    def __str__(self):
        return f"<{self.name}:{self.func}:{self.trigger}>"


class taskSolver(object):
    def __init__(self, taskfile=None):
        if taskfile:
            self.module_list = [
                venus.api_stock_event,
                taurus.nlp_event,
                mars.database_manager,
                saturn.finance_event,
                saturn.balance_analysis,
                mars.mail_manager,
                mars.log_manager,
                ]
            self.taskfile = taskfile
            self.func_list = {}
            self.load_event()
        else:
            raise FileNotFoundError(taskfile)

    @log_decorator2
    def load_event(self):
        """
        This function will reload modules automatically.
        """
        for mod in self.module_list:
            imp.reload(mod)
            for func in mod.__all__:
                self.func_list[func] = eval(f"{mod.__name__}.{func}")

    @log_decorator
    def task_resolve(self, jsdata: json):
        task = None
        if task_name := jsdata.get('task'):
            if (func := self.func_list.get(task_name)) and (trigger := self._trigger_resolve(jsdata)):
                task = taskBase(task_name, func, trigger)
        return task

    def _trigger_resolve(self, jsdata):
        """
        Resolve the trigger.
        FORMAT:
        day_of_week: 0~6 stands for sun to sat.
        day        : 1~31 stands for day in month.
        hour       : 18 stands for 18:00.
        time       : 18:31 stands for real time.
        work day   : in time format like 18:31, means time on work day.
        sat, sun   : the same like work day
        """
        for k in jsdata.keys():
            if k == 'day of week':
                trigger = CronTrigger(day_of_week=jsdata['day_of_week'])
            elif k == 'day':
                trigger = CronTrigger(day=jsdata['day'])
            elif k == 'hour':
                trigger = CronTrigger(hour=jsdata['hour'])
            elif k == 'time':
                if m := re.match(r'(\d{1,2}):(\d{2})', jsdata['time']):
                    trigger = CronTrigger(
                        hour=int(m.group(1)),
                        minute=int(m.group(2)))
            elif k == 'work day':
                if m := re.match(r'(\d{1,2}):(\d{2})', jsdata['work day']):
                    trigger = CronTrigger(
                        day_of_week='mon,tue,wed,thu,fri',
                        hour=int(m.group(1)),
                        minute=int(m.group(2)))
            elif k == 'sat':
                if m := re.match(r'(\d{1,2}):(\d{2})', jsdata['sat']):
                    trigger = CronTrigger(
                        day_of_week='sat',
                        hour=int(m.group(1)),
                        minute=int(m.group(2)))
            elif k == 'sun':
                if m := re.match(r'(\d{1,2}):(\d{2})', jsdata['sun']):
                    trigger = CronTrigger(
                        day_of_week='sun',
                        hour=int(m.group(1)),
                        minute=int(m.group(2)))
            else:
                trigger = None
        return trigger

    def load_task_file(self):
        with open(self.taskfile, 'r') as f:
            task_json = json.load(f)
        return task_json


def timedelta_convert(dt: datetime.timedelta) -> tuple:
    """
    convert timedelta to hh:mm
    param  : dt  datetime.timedelta
    return : ( hours of int, minute of int )
    """
    h = 24 * dt.days + dt.seconds // 3600
    m = (dt.seconds % 3600) // 60
    return (h, m)


def format_timedelta(dt: datetime.timedelta) -> str:
    h, m = timedelta_convert(dt)
    return f"{h}:{str(m).zfill(2)}"


if __name__ == "__main__":
    import datetime
    event = taskManager2(
        '/home/friederich/Dev/neutrino2/config/task.json', 'Neptune')
    event.task_solver.load_event()
    # print(event.task_solver.func_list)
    result = event.task_solver.load_task_file()
    task_list = []
    for task_data in result:
        task = event.task_solver.task_resolve(task_data)
        task_list.append(task)
    print(task_list)
