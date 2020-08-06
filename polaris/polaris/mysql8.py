#!/usr/bin/python3
"""
This is a library dealing with mysql which is based on sqlalchemy.
Library support MYSQL version 8.
Auther: Friederich River
"""
import json
import datetime
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


__version__ = '4.4.20'


class mysqlBase(object):
    def __init__(self, header):
        """
        :param header: Defines the mysql engine parameters.
        :param engine: is the object returned from create_engine.
        :param session: contains the cursor object.
        """
        self.account = header.account
        self.host = header.host
        self.port = header.port
        self.database = header.database
        mysql_url = (
            f"mysql+pymysql://{header.account}:"
            f"{header.password}"
            f"@{header.host}:{header.port}"
            f"/{header.database}")
        self.engine = create_engine(
            mysql_url,
            encoding='utf8',
            echo=False)
        db_session = sessionmaker(bind=self.engine)
        self.session = db_session()
        self.id_string = (
            f"mysql engine <{header.account}@{header.host}>")

    def __str__(self):
        return self.id_string

    def insert(self, sql):
        self.engine.execute(sql)
        return 1

    def insert2(self, table: str, value: dict):
        if isinstance(value, dict):
            sql = f"INSERT IGNORE INTO {table} "
            for key in value.keys():
                sql += f"{key}={value[key]} "
            self.engine.execute(sql)

    def show_column(self, table: str) -> DataFrame:
        """
        Return a DataFrame like df['col', 'col_type']
        """
        # query for column definition.
        sql = "Show columns from stock_manager"
        select_value = self.engine.execute(sql)
        # translate into dataframe
        df = pd.DataFrame(select_value)
        # dataframe trimming
        col_info = df.loc[:, :1]
        col_info.columns = ['col', 'col_type']
        return col_info

    def select_one(self, table, field, condition):
        """
        Result is a tuple like structure data.
        """
        sql = f"SELECT {field} FROM {table} WHERE {condition}"
        result = self.engine.execute(sql).fetchone()
        return result

    def simple_select(self, table, field):
        """
        Return a tuple like result
        """
        sql = f"SELECT {field} from {table}"
        result = self.engine.execute(sql)
        return result

    def select_values(self, table, field):
        """
        Return a DataFrame type result.
        """
        sql = f"SELECT {field} from {table}"
        select_value = self.engine.execute(sql)
        result = pd.DataFrame(select_value)
        return result

    def update_value(self, table, field, value, condition):
        """
        Single value update.
        """
        sql = (f"UPDATE {table} set {field}={value} WHERE {condition}")
        self.engine.execute(sql)

    def condition_select(self, table, field, condition):
        """
        Return a DataFrame type result.
        """
        sql = f"SELECT {field} from {table} WHERE {condition}"
        select_value = self.engine.execute(sql)
        result = pd.DataFrame(select_value)
        return result

    def query(self, sql):
        result = self.engine.execute(sql).fetchone()
        return result

    def drop_table(self, table_name: str):
        sql = f"DROP TABLE {table_name}"
        self.engine.execute(sql)

    def truncate_table(self, table_name: str):
        sql = f"TRUNCATE TABLE {table_name}"
        self.engine.execute(sql)

    def create_table(self, table):
        """
        : param table: It is form template defined in form module.
        : param engine: It is a sqlalchemy mysql engine.
        """
        table.metadata.create_all(self.engine)

    def create_table_from_table(self, name, table_template):
        """
        Base on a table, create another form which
        is similar with the original table.
        Only name was changed.\n
        Params :\n
        name : which is the target table name.\n
        tableName : which is the original table name.\n
        engine : a database engine base on MySQLBase.\n
        """
        sql = f"CREATE table if not exists {name} like {table_template}"
        self.engine.execute(sql)
        return 1


def _drop_all(base, engine):
    """
    This will drop all tables in database.
    It is a private method only for maintance.
    """
    base.metadata.drop_all(engine)


class mysqlHeader(object):
    """ Here defines the parameters passed into mysql engine.
    """

    def __init__(self, acc, pw, db,
                 host='localhost', port=3306, charset='utf8'):
        if not isinstance(acc, str):
            raise TypeError(f"{acc=} is not correct.")
        if not isinstance(pw, str):
            raise TypeError("Password is not correct.")
        if not isinstance(db, str):
            raise TypeError(f"{db=} is not correct.")
        self.account = acc
        self.password = pw
        self.database = db
        self.host = host
        self.port = port
        self.charset = 'utf8'

    def __str__(self):
        result = (
            str(self.account), str(self.host),
            str(self.port), str(self.database), str(self.charset))
        return result


def create_table(table, engine):
    """
    : param table: It is form template defined in form module.
    : param engine: It is a sqlalchemy mysql engine.
    """
    table.metadata.create_all(engine)


class sqlExecutor(object):
    def __init__(self, executor, sql):
        self.executor = executor

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Json2Sql(mysqlBase):
    """
    Translate Json data into Sql (insert or update)\n
    Working flow :\n
    1. load_table;\n
    2. query and fetch dataframe;\n
    3. translate dataframe to json;\n
    4. to_sql (insert or update)
    """
    def __init__(self, header):
        super(Json2Sql, self).__init__(header)
        self.table_def = dict()
        self.tablename = None

    def load_table(self, tablename):
        """
        Load table infomation like column name and type.
        """
        self.table_def = dict()
        self.tablename = tablename
        # query for column definition.
        sql = f"Show columns from {self.tablename}"
        select_value = self.engine.execute(sql)
        # translate into dataframe
        df = pd.DataFrame(select_value)
        # dataframe trimming
        col_info = df.loc[:, :1]
        col_info.columns = ['col', 'col_type']
        # tranlate dataframe into json
        for index, row in col_info.iterrows():
            self.table_def[row['col']] = row['col_type']
        return self.table_def

    @staticmethod
    def dataframe_to_json(df: pd.DataFrame, keys: list) -> list:
        df.columns = keys
        jsonlist = []
        tmp = {}
        for index, row in df.iterrows():
            tmp = {}
            for k in keys:
                tmp[k] = row[k]
            jsonlist.append(tmp)
        return jsonlist

    def to_sql_insert(self, json_data: json):
        """
        Generate a sql like 'Replace into <table> (<cols>) values (<vals>)'
        """
        # initial 2 list
        col_section = []
        value_section = []
        # iter for each key in json.
        for k, v in json_data.items():
            if k in self.table_def.keys():
                col_section.append(k)
                # values should be tranlate into sql format
                value_section.append(self.trans_value(v))
        # combine into sql and returns.
        col = ','.join(col_section)
        val = ','.join(value_section)
        sql = f"REPLACE into {self.tablename} ({col}) values ({val})"
        return(sql)

    def to_sql_update(self, json_data: json, keys: list):
        val = []
        cond = []
        for k, v in json_data.items():
            if k in self.table_def.keys():
                tmp = f"{k}={self.trans_value(v)}"
                val.append(tmp)
        value = ','.join(val)
        for k in keys:
            tmp = f" ({k}={self.trans_value(json_data[k])}) "
            cond.append(tmp)
        condition = 'AND'.join(cond)
        sql = f"UPDATE {self.tablename} set {value} WHERE {condition}"
        return sql

    @staticmethod
    def trans_value(value):
        from dev_global.env import TIME_FMT
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, int):
            return f"{value}"
        elif isinstance(value, float):
            return f"{value}"
        elif isinstance(value, datetime.date):
            return f"'{value.strftime(TIME_FMT)}'"
        else:
            return 'NULL'


GLOBAL_HEADER = mysqlHeader('stock', 'stock2020', 'stock')
TEST_HEADER = mysqlHeader('stock', 'stock2020', 'test')
VIEWER_HEADER = mysqlHeader('view', 'view2020', 'stock')


if __name__ == '__main__':
    # h = mysqlHeader(1, '1', '1')
    import dev_global.var
    j2q = Json2Sql(GLOBAL_HEADER)
    j2q.load_table('template_stock')
    df = pd.read_csv('/home/friederich/Downloads/002456.csv', names=dev_global.var.stock_table_column, encoding='gb18030')
    tmp = dev_global.var.stock_table_column
    # tmp.remove('stock_code')
    # print(tmp)
    data_list = j2q.dataframe_to_json(df, keys=tmp)
    for data in data_list:
        sql = j2q.to_sql_insert(data)
        print(sql)
