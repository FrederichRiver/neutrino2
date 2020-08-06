#!/usr/bin/python38
from polaris.mysql8 import mysqlHeader

GLOBAL_HEADER = mysqlHeader('stock', 'stock2020', 'stock')
VIEWER_HEADER = mysqlHeader('view', 'view2020', 'stock')
TEST_HEADER = mysqlHeader('stock', 'stock2020', 'test')