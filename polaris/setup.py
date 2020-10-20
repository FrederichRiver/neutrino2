#!/usr/bin/python3
from setuptools import setup, find_packages
from dev_global.env import GITHUB_URL
from polaris import __version__ as v


setup(
        name='polaris',
        version=f"{v[0]}.{v[1]}.{v[2]}",
        packages=find_packages(),
        install_requires=['sqlalchemy>=1.3.16', 'py2neo'],
        author='Fred Monster',
        author_email='hezhiyuan_tju@163.com',
        url=GITHUB_URL,
        license='LICENSE',
        description='None',
        )
