#!/usr/bin/python3
import pandas as pd
import numpy as np
# from datetime import timedelta
import dev_global.var
from dev_global.env import CONF_FILE
# from dev_global.var import stock_table_column
from mars.utils import read_url, drop_space
from venus.stock_base2 import StockBase
from venus.form import formStockManager
from mars.log_manager import log_decorator, log_decorator2


class EventTradeDataManager(StockBase):
    """
    It is a basic event, which fetch trade data and manage it.
    """
    def __init__(self, header):
        super(EventTradeDataManager, self).__init__(header)
        self.url = read_url('URL_163_MONEY', CONF_FILE)
        self.j2sql.load_table('template_stock')

    @staticmethod
    def net_ease_code(stock_code):
        """
        input: SH600000, return: 0600000\n;
        input: SZ000001, return: 1000001.
        """
        if isinstance(stock_code, str):
            if stock_code[:2] == 'SH':
                stock_code = '0' + stock_code[2:]
            elif stock_code[:2] == 'SZ':
                stock_code = '1' + stock_code[2:]
            else:
                stock_code = None
        else:
            stock_code = None
        return stock_code

    def url_netease(self, stock_code, start_date, end_date):
        query_code = self.net_ease_code(stock_code)
        netease_url = self.url.format(query_code, start_date, end_date)
        return netease_url

    def get_trade_data(self, stock_code, end_date, start_date='19901219') -> pd.DataFrame:
        """
        read csv data and return dataframe type data.
        """
        # config file is a url file.
        url = self.url_netease(stock_code, start_date, end_date)
        print(url)
        df = pd.read_csv(url, names=dev_global.var.stock_table_column, encoding='gb18030')
        return df

    def get_stock_name(self, stock_code):
        """
        Searching stock name from net ease.
        """
        result = self.get_trade_data(stock_code, self.today)
        if not result.empty:
            stock_name = drop_space(result.iloc[1, 2])
        else:
            stock_name = None
        return stock_code, stock_name

    @log_decorator
    def record_stock(self, stock_code):
        """
        Record table <stock_code> into database.
        """
        result = self.check_stock(stock_code)
        # if table exists, result = (stock_code,)
        # else result = (,)
        if not result:
            self.create_stock_table(stock_code)
            self.init_stock_data(stock_code)

    def check_stock(self, stock_code):
        """
        Check whether table <stock_code> exists.
        Used in record_stock()
        """
        result = self.session.query(formStockManager.stock_code).filter_by(stock_code=stock_code).first()
        return result

    def create_stock_table(self, stock_code):
        if self.create_table_from_table(stock_code, 'template_stock'):
            stock = formStockManager(stock_code=stock_code, create_date=self.Today)
            self.session.add(stock)
            self.session.commit()
            return 1
        else:
            return 0

    def _clean(self, df: pd.DataFrame):
        """
        Param: df is a DataFrame like data.
        """
        df.drop(['stock_code'], axis=1, inplace=True)
        df = df[1:]
        df.replace('None', np.nan, inplace=True)
        df = df.dropna(axis=0, how='any')
        return df

    def init_stock_data(self, stock_code):
        """
        used when first time download stock data.
        """
        result = self.get_trade_data(stock_code, self.today)
        result = self._clean(result)
        result.to_sql(
            name=stock_code,
            con=self.engine,
            if_exists='append',
            index=False)
        query = self.session.query(
                formStockManager.stock_code,
                formStockManager.update_date
            ).filter_by(stock_code=stock_code)
        if query:
            query.update({"update_date": self.Today})
        self.session.commit()

    # @log_decorator2
    def download_stock_data(self, stock_code):
        # fetch last update date.
        update = self.session.query(formStockManager.update_date).filter_by(stock_code=stock_code).first()
        if update[0]:
            # tmp = update[0] + timedelta(days=1)
            tmp = update[0]
            update_date = tmp.strftime('%Y%m%d')
        else:
            update_date = '19901219'
        # fetch trade data.
        df = self.get_trade_data(stock_code, self.today, start_date=update_date)
        df = self._clean(df)
        df = df.sort_values(['trade_date'])
        print(df.head(5))
        print(df.columns)
        print(dev_global.var.stock_table_column)
        tmp = dev_global.var.stock_table_column
        tmp.remove('stock_code')
        print(tmp)
        df_json = self.j2sql.dataframe_to_json(df, keys=tmp)
        for data in df_json:
            sql = self.j2sql.to_sql_insert(data)
            self.engine.execute(sql)


if __name__ == "__main__":
    from polaris.mysql8 import GLOBAL_HEADER
    event = EventTradeDataManager(GLOBAL_HEADER)
    stock_code = 'SH600000'
    # event.download_stock_data(stock_code)
    result = event.check_stock(stock_code)
    print(result)
