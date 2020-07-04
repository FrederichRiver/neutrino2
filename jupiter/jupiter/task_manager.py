#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imp
import json
import os
import re
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from jupiter.utils import ERROR
from jupiter.log_manager import log_decorator
# modules loaded into module list
import venus.api_stock_event
import taurus.nlp_event
import saturn.finance_event
import saturn.balance_analysis
import jupiter.database_manager
import jupiter.log_manager
import jupiter.mail_manager

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


class taskManager(BackgroundScheduler):
    """
    Task manager is a object to manage tasks.
    It will run tasks according to task.json file.
    It will auto load modules without reboot system.
    """
    def __init__(self, taskfile=None, gconfig={}, **options):
        super(BackgroundScheduler, self).__init__(
            gconfig=gconfig, **options)
        # if task file is not found.
        if not taskfile:
            ERROR("Task file is not found.")
        else:
            self.module_list = [
                venus.api_stock_event, taurus.nlp_event,
                jupiter.database_manager, saturn.finance_event,
                saturn.balance_analysis,
                jupiter.mail_manager, jupiter.log_manager]
            self.taskfile = taskfile
            self.func_list = {}
            self.reload_event()

    def reload_event(self):
        """
        This function will reload modules automatically.
        """
        import imp
        try:
            for mod in self.module_list:
                imp.reload(mod)
                for func in mod.__all__:
                    # print(mod.__name__, func)
                    self.func_list[func] = eval(f"{mod.__name__}.{func}")
        except Exception as e:
            ERROR(e)
            ERROR(f"Can not find module {mod.__name__}, {func}.")

    def task_resolve(self, jsdata):
        """
        Resolve the task file into job and trigger.
        File format is json.
        Return: {job: trigger}
        """
        tasklist = {}
        for task in jsdata:
            # job_resolve
            try:
                job = self.func_list[self.job_resolve(task)]
                trigger = self.trigger_resolve(task)
            except KeyError:
                ERROR(f"Job {self.job_resolve(task)} could not be found.")
            except Exception as e:
                ERROR(e)
            # trigger_resolve
            if job and trigger:
                tasklist[job] = trigger
        # print(tasklist)
        # format:
        # { job_function<function> : job_trigger<trigger> }
        return tasklist

    def job_resolve(self, jsdata):
        """
        Filter of task file which can filt the incorrect
        config content.
        """
        jobname = jsdata['task'] if 'task' in jsdata.keys() else None
        return jobname

    def trigger_resolve(self, jsdata):
        """
        Resolve the trigger.
        """
        for k in jsdata.keys():
            if k == 'day_of_week':
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

    def check_task_file(self):
        # if task file not exist, send a warning.
        if os.path.exists(self.taskfile):
            # INFO("Task file checking successd.")
            self.append_task()
        else:
            ERROR("Task plan file is not found.")

    def append_task(self):
        # if exist, resolve the task file.
        try:
            with open(self.taskfile, 'r') as js:
                load_dict = json.load(js)
                result = self.task_resolve(load_dict)
            job_list = self.get_jobs()
            jobexist = False
            for (k, v) in result.items():
                for job in job_list:
                    if k.__name__ == job.id:
                        self.reschedule_job(k.__name__, trigger=v)
                        jobexist = True
                if not jobexist:
                    self.add_job(k, trigger=v, id=k.__name__)
                    print(f'add job {k.__name__}\n')
                jobexist = False
            # self.task_report()
        except Exception as e:
            ERROR("Append task error: ", e)

    def task_report(self):
        print('+-'*15)
        self.print_jobs()
        print('+-'*15)


class taskManager2(BackgroundScheduler):
    """
    Task manager is a object to manage tasks.
    It will run tasks according to task.json file.
    It will auto load modules without reboot system.
    """
    def __init__(self, taskfile=None, task_manager_name=None, gconfig={}, **options):
        super(BackgroundScheduler, self).__init__(gconfig=gconfig, **options)
        # if task file is not found.
        self.task_manager_name = task_manager_name
        self.start_time = datetime.datetime.now()
        if not taskfile:
            ERROR("Task file is not found.")
        else:
            self.module_list = [
                venus.api_stock_event,
                taurus.nlp_event,
                jupiter.database_manager,
                saturn.finance_event,
                saturn.balance_analysis,
                jupiter.mail_manager,
                jupiter.log_manager,
                ]
            self.taskfile = taskfile
            self.func_list = {}
            self.task_solver = taskSolver(taskfile)

    def __str__(self):
        runtime = datetime.datetime.now() - self.start_time
        h, m = timedelta_convert(runtime)
        return f"<Task manager ({self.task_manager_name}) has running for {h}:{str(m).zfill(2)}:00>"

    def task_resolve(self, jsdata):
        """
        Resolve the task file into job and trigger.
        File format is json.
        Return: {job: trigger}
        """
        tasklist = []
        for task_data in jsdata:
            # job_resolve
            task = self.task_solver(task_data)
            tasklist.append(task)
        # format:
        # { job_function<function> : job_trigger<trigger> }
        return tasklist

    def check_task_file(self):
        # if task file not exist, send a warning.
        if os.path.exists(self.taskfile):
            self.append_task()
        else:
            ERROR("Task plan file is not found.")

    def append_task(self):
        # if exist, resolve the task file.
        try:
            with open(self.taskfile, 'r') as js:
                load_dict = json.load(js)
                result = self.task_resolve(load_dict)
            job_list = self.get_jobs()
            jobexist = False
            for (k, v) in result.items():
                for job in job_list:
                    if k.__name__ == job.id:
                        self.reschedule_job(k.__name__, trigger=v)
                        jobexist = True
                if not jobexist:
                    self.add_job(k, trigger=v, id=k.__name__)
                    print(f'add job {k.__name__}\n')
                jobexist = False
            # self.task_report()
        except Exception as e:
            ERROR("Append task error: ", e)

    @decoration_print
    def task_report(self):
        self.print_jobs()


class taskBase(object):
    def __init__(self, task_name, func, trigger):
        self.name = task_name
        self.func = func
        self.trigger = trigger

    def __str__(self):
        return f"<{self.name}:{self.func}:{self.trigger}>"


class taskSolver(object):
    def __init__(self, taskfile=None):
        if taskfile:
            self.module_list = [
                venus.api_stock_event,
                taurus.nlp_event,
                jupiter.database_manager,
                saturn.finance_event,
                saturn.balance_analysis,
                jupiter.mail_manager,
                jupiter.log_manager,
                ]
            self.taskfile = taskfile
            self.func_list = {}
            self.load_event()
        else:
            raise FileNotFoundError(taskfile)

    @log_decorator
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
                m = re.match(r'(\d{1,2}):(\d{2})', jsdata['time'])
                trigger = CronTrigger(
                    hour=int(m.group(1)),
                    minute=int(m.group(2)))
            elif k == 'work day':
                m = re.match(r'(\d{1,2}):(\d{2})', jsdata['work day'])
                trigger = CronTrigger(
                    day_of_week='mon,tue,wed,thu,fri',
                    hour=int(m.group(1)),
                    minute=int(m.group(2)))
            elif k == 'sat':
                m = re.match(r'(\d{1,2}):(\d{2})', jsdata['sat'])
                trigger = CronTrigger(
                    day_of_week='sat',
                    hour=int(m.group(1)),
                    minute=int(m.group(2)))
            elif k == 'sun':
                m = re.match(r'(\d{1,2}):(\d{2})', jsdata['sun'])
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
    return f"{h}:{m}"


if __name__ == "__main__":
    import datetime
    event = taskManager2('/home/friederich/Dev/neutrino2/config/task.json', 'Neptune')
    event.task_solver.load_event()
    # print(event.task_solver.func_list)
    result = event.task_solver.load_task_file()
    task_list = []
    for task_data in result:
        task = event.task_solver.task_resolve(task_data)
        task_list.append(task)
    print(task_list)
