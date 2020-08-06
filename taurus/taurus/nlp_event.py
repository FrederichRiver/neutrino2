#!/usr/bin/python3

__version__ = 3
__all__ = ['event_download_netease_news', 'event_record_announce_url', ]


def event_download_netease_news():
    from polaris.mysql8 import mysqlHeader
    from dev_global.env import SOFT_PATH
    from taurus.news_downloader import neteaseNewsSpider
    header = mysqlHeader('stock', 'stock2020', 'natural_language')
    event = neteaseNewsSpider(header, SOFT_PATH)
    event.generate_url_list()
    for url in event.url_list:
        event.extract_href(url)
    event.save_process()
    # hfile = SOFT_PATH + 'config/HREF_LIST'
    # event.load_href_file(hfile)
    for url in event.href:
        art = event.extract_article(url)
        event.record_article(art)


def event_record_announce_url():
    from polaris.mysql8 import mysqlHeader, GLOBAL_HEADER
    from venus.stock_base import StockEventBase
    from taurus.announcement import cninfoAnnounce
    event_stock_list = StockEventBase(GLOBAL_HEADER)
    stock_list = event_stock_list.get_all_stock_list()
    mysql_header = mysqlHeader('stock', 'stock2020', 'natural_language')
    event = cninfoAnnounce(mysql_header)
    event._set_param()
    for stock in stock_list:
        event.run(stock)


if __name__ == "__main__":
    event_download_netease_news()
