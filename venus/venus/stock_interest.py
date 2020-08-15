#!/usr/bin/python3
from mars.log_manager import log_decorator2
import numpy as np
import pandas as pd
from pandas import DataFrame
from dev_global.var import stock_interest_column
from dev_global.env import CONF_FILE
from polaris.mysql8 import mysqlHeader
from venus.stock_base2 import StockBase


__version__ = '1.2.4'


class EventInterest(StockBase):
    """
    """
    def __init__(self, header):
        super(EventInterest, self).__init__(header)
        self.df = DataFrame()

    def _load_template(self):
        """
        return 1 means success.\n
        """
        self.j2sql.table_def = self.j2sql.load_table('stock_interest')
        return 1

    def create_interest_table(self):
        """
        Initial interest table.
        """
        from venus.form import formInterest
        self.create_table_from_table("stock_interest", formInterest.__tablename__)

    def resolve_interest_table(self, stock_code: str):
        """
        Recognize interest table from html,
        returns a dataframe table.\n
        Used for batch insert.
        """
        url = self.read_url('URL_fh_163', CONF_FILE)
        url = url.format(stock_code[2:])
        # result is a list of DataFrame table.
        result = pd.read_html(url, attrs={'class': 'table_bg001 border_box limit_sale'})
        if result:
            df = result[0]
            df.columns = [
                'report_date', 'int_year', 'float_bonus',
                'float_increase', 'float_dividend',
                'record_date', 'xrdr_date', 'share_date']
            df['char_stock_code'] = stock_code
            df.replace('--', np.nan, inplace=True)
            # change column type according to their pre_fix.
            result = self.dataframe_data_translate(df)
        else:
            result = DataFrame()
        return result

    @log_decorator2
    def record_interest(self, stock_code: str) -> None:
        df = self.resolve_interest_table(stock_code)
        json_data = self.j2sql.dataframe_to_json(df, stock_interest_column)
        for data in json_data:
            # print(data)
            sql = self.j2sql.to_sql_insert(data)
            self.engine.execute(sql)


class EventStockData(StockBase):
    def __init__(self, header: mysqlHeader) -> None:
        super(EventStockData, self).__init__(header)

    def set_ipo_date(self, stock_code):
        """
        Fetch the first date of stock and set it as ipo date.
        """
        query = self.select_values(stock_code, 'trade_date')
        ipo_date = pd.to_datetime(query[0])
        # ipo_date = datetime.date(1990,12,19)
        self.update_value(
            'stock_manager', 'ipo_date',
            f"'{ipo_date[0]}'", f"stock_code='{stock_code}'")
        return ipo_date[0]

    def get_ipo_date(self, stock_code):
        query = self.select_values(stock_code, 'trade_date')
        ipo_date = pd.to_datetime(query[0])
        return ipo_date[0]


if __name__ == "__main__":
    from polaris.mysql8 import GLOBAL_HEADER
    event = EventStockData(GLOBAL_HEADER)
