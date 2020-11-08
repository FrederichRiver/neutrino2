#!/usr/bin/python3.8
import os

PROJ_PATH = '/home/friederich/Dev/neutrino2'
lib_list = ['dev_global', 'mars', 'polaris', 'venus', 'saturn', 'taurus', 'andromeda']
for lib in lib_list:
    # go into library directory
    os.chdir(f"{PROJ_PATH}/{lib}")
    # run setup script
    os.system("python3 setup.py sdist")
    # remove egg-info file in package
    os.system(f"rm -r {lib}.egg-info")
    # cp package in lib/dist into root path
    os.system(f"cp -r dist/ {PROJ_PATH}/")
    # remove lib/dist
    os.system("rm -r dist/")
