#!/usr/bin/python3
import os
from dev_global.env import PROJ_PATH

root_path = PROJ_PATH
root_path = '/home/friederich/Dev/neutrino2'
lib_list = ['dev_global', 'jupiter', 'polaris', 'venus', 'saturn', 'taurus']
for lib in lib_list:
    # go into library directory
    os.chdir(f"{root_path}/{lib}")
    # run setup script
    os.system("python3 setup.py sdist")
    # remove egg-info file in package
    os.system(f"rm -r {lib}.egg-info")
    # cp package in lib/dist into root path
    os.system(f"cp -r dist/ {root_path}/")
    # remove lib/dist
    os.system("rm -r dist/")
