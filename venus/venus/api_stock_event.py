#!/usr/bin/python3
from venus.company import EventCompany
from mars.network import delay
from venus.finance_report import EventFinanceReport
import pandas
from venus.cninfo import cninfoSpider
from mars.utils import ERROR
from polaris.mysql8 import GLOBAL_HEADER


__version__ = '1.2.15'
__all__ = [
    'event_download_balance_data',
    'event_download_cashflow_data',
    'event_download_finance_report',
    'event_download_income_data',
    'event_download_index_data',
    'event_download_stock_data',
    'event_flag_quit_stock',
    'event_flag_stock',
    'event_finance_info',
    'event_get_hk_list',
    'event_init_interest',
    'event_init_stock',
    'event_rehabilitation',
    'event_record_company_infomation',
    'event_record_company_stock_structure',
    'event_record_new_stock',
    'event_record_orgid',
    'event_update_shibor',
    ]

# Event Trade Data Manager


# new frame
def event_init_stock():
    """
    Init database from a blank stock list.
    """
    from venus.stock_manager2 import EventTradeDataManager
    from venus.stock_base2 import resolve_stock_list
    stock_list = resolve_stock_list('totalstocklist')
    event = EventTradeDataManager(GLOBAL_HEADER)
    for stock_code in stock_list:
        event.record_stock(stock_code)
        delay(5)


# new frame
def event_record_new_stock():
    from venus.stock_manager2 import EventTradeDataManager
    from venus.stock_base2 import resolve_stock_list
    event = EventTradeDataManager(GLOBAL_HEADER)
    stock_list = resolve_stock_list('STOCK')
    for stock_code in stock_list:
        event.record_stock(stock_code)


def event_download_stock_data():
    from venus.stock_manager import EventTradeDataManager
    event = EventTradeDataManager(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock_code in stock_list:
        event.download_stock_data(stock_code)


def event_download_index_data():
    from venus.stock_manager import EventTradeDataManager
    event = EventTradeDataManager(GLOBAL_HEADER)
    stock_list = event.get_all_index_list()
    for stock_code in stock_list:
        event.download_stock_data(stock_code)


# new frame
def event_flag_quit_stock():
    from venus.stock_classify import StockClassify
    from venus.stock_base2 import resolve_stock_list
    event = StockClassify(GLOBAL_HEADER)
    stock_list = resolve_stock_list('STOCK')
    for stock_code in stock_list:
        flag = event.flag_quit_stock(stock_code)
        if flag:
            sql = (
                f"UPDATE stock_manager set flag='q' "
                f"WHERE stock_code='{stock_code}'")
            event.engine.execute(sql)


# event record interest
def event_init_interest():
    from venus.stock_interest import EventInterest
    from venus.stock_base2 import resolve_stock_list
    stock_list = resolve_stock_list('stock')
    event = EventInterest(GLOBAL_HEADER)
    event._load_template()
    for stock_code in stock_list:
        event.record_interest(stock_code)
        delay(10)


# new frame
def event_flag_stock():
    import re
    from venus.stock_classify import StockClassify
    event = StockClassify(GLOBAL_HEADER)
    stock_list = event.get_all_security_list()
    for stock_code in stock_list:
        if re.match(r'^SH60|^SZ00|^SZ300|^SH688', stock_code):
            event.flag_stock_type(stock_code, 't')
        elif re.match(r'^SH900|^SZ200', stock_code):
            event.flag_stock_type(stock_code, 'b')
        elif re.match(r'^SH000|^SH950|^SZ399', stock_code):
            event.flag_stock_type(stock_code, 'i')
        elif re.match(r'^HK', stock_code):
            event.flag_stock_type(stock_code, 'h')


# to be delete
def event_flag_b_stock():
    import re
    from venus.stock_flag import EventStockFlag
    event = EventStockFlag(GLOBAL_HEADER)
    stock_list = event.get_all_security_list()
    for stock_code in stock_list:
        if re.match(r'^SH900|^SZ200', stock_code):
            event.flag_b_stock(stock_code)


# to be delete
def event_flag_index():
    import re
    from venus.stock_flag import EventStockFlag
    event = EventStockFlag(GLOBAL_HEADER)
    stock_list = event.get_all_security_list()
    for stock_code in stock_list:
        if re.match(r'^SH000|^SH950|^SZ399', stock_code):
            event.flag_index(stock_code)


# to be delete
def event_flag_hk_stock():
    import re
    from venus.stock_flag import EventStockFlag
    event = EventStockFlag(GLOBAL_HEADER)
    stock_list = event.get_all_security_list()
    for stock_code in stock_list:
        if re.match(r'^HK', stock_code):
            event.flag_hk_stock(stock_code)


def event_rehabilitation():
    pass


def event_record_company_infomation():
    event = EventCompany(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock_code in stock_list:
        try:
            event.record_company_infomation(stock_code)
        except Exception as e:
            ERROR(e)


def event_finance_info():
    pass


def event_record_company_stock_structure():
    event = EventCompany(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock in stock_list:
        try:
            # print(stock)
            event.record_stock_structure(stock)
            delay(10)
        except Exception as e:
            ERROR(e)


def event_download_finance_report():
    event = EventFinanceReport(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock in stock_list:
        print(f"Download finance report of {stock}.")
        # event.update_balance_sheet(stock)
        event.update_income(stock)
        event.update_balance(stock)
        event.update_cashflow(stock)


# new frame
def event_update_shibor():
    from venus.shibor import EventShibor
    event = EventShibor(GLOBAL_HEADER)
    year_list = range(2006, pandas.Timestamp.today().year + 1)
    for year in year_list:
        url = event.get_shibor_url(year)
        df = event.get_excel_object(url)
        event.get_shibor_data(df)


def event_get_hk_list():
    event = cninfoSpider(GLOBAL_HEADER)
    df = event.get_hk_stock_list()
    event._insert_stock_manager(df)


def event_record_orgid():
    event = cninfoSpider(GLOBAL_HEADER)
    df = event.get_stock_list()
    event._update_stock_manager(df)
    df = event.get_hk_stock_list()
    event._update_stock_manager(df)


def event_download_balance_data():
    event = EventFinanceReport(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock_code in stock_list:
        try:
            event.update_balance(stock_code)
        except Exception:
            ERROR(f"Error occours while recording {stock_code} balance sheet.")


def event_download_cashflow_data():
    event = EventFinanceReport(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock_code in stock_list:
        try:
            event.update_cashflow(stock_code)
        except Exception as e:
            ERROR(f"Error occours while recording {stock_code} cashflow sheet.")
            ERROR(e)


def event_download_income_data():
    event = EventFinanceReport(GLOBAL_HEADER)
    stock_list = event.get_all_stock_list()
    for stock_code in stock_list:
        try:
            event.update_income(stock_code)
        except Exception as e:
            ERROR(f"Error occours while recording {stock_code} income sheet.")
            ERROR(e)


if __name__ == "__main__":
    event_init_interest()
