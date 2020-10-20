#!/bin/bash
echo "Start socket server."
python38 run_task_server.py Start
sleep 10
echo "Start service [1]: system update"
python38 system_update.py Start
sleep 5
echo "Start service [2]: database backup"
python38 service_database_backup.py Start
sleep 5