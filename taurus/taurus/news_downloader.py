#! /usr/bin/env python38
import re
import json
import requests
import lxml
from mars.utils import ERROR
from taurus.model import article, ArticleBase
from polaris.mysql8 import mysqlHeader, mysqlBase
from sqlalchemy import Column, String, Integer, Date, Text
from sqlalchemy.ext.declarative import declarative_base


__version__ = '1.2.4'

article_base = declarative_base()


class NeteaseArticle(ArticleBase):
    def __init__(self, html, url):
        super(NeteaseArticle, self).__init__()
        if not isinstance(html, lxml.etree._Element):
            raise TypeError('html type error')
        self._get_url(url)
        self._get_title(html)
        self._get_author(html)
        self._get_date(html)
        self._get_source(html)
        self._get_content(html)

    def _get_url(self, url):
        self.j_dict['url'] = url

    def _get_date(self, input_content):
        regex_date = r'\d{4}-\d{2}-\d{2}'
        date_string = input_content.xpath("//div[@class='post_time_source']/text()")
        for s in date_string:
            result = re.search(regex_date, s)
            if result:
                self.j_dict['date'] = result[0]

    def _get_title(self, input_content):
        title = input_content.xpath("//div/h1/text()")
        if title:
            self.j_dict['title'] = title[0].strip()

    def _get_source(self, input_content):
        source = input_content.xpath("//div[@class='ep-source cDGray']/span[@class='left']/text()")
        if source:
            result = re.split(r'：', source[0])
            self.j_dict['source'] = result[1].strip()

    def _get_author(self, input_content):
        author = input_content.xpath("//span[@class='ep-editor']/text()")
        if author:
            result = re.split(r'：', author[0])
            self.j_dict['author'] = result[1].strip()

    def _get_content(self, input_content):
        """
        tag: //div[@class=post_text] for netease finance
        """
        html = input_content.xpath("//div[@class='post_text']/p//text()")
        content = ''
        for line in html:
            content += line
        content = content.strip()
        content = content.replace(' ', '')
        content = content.replace('\n', '')
        self.j_dict['content'] = content

    @staticmethod
    def save_content_text(html, filename):
        with open(filename, 'w') as f:
            f.write(html)

    def article_to_json(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.j_dict, ensure_ascii=False, indent=4))

    def article_to_sql(self):
        pass


class formArticle(article_base):
    __tablename__ = 'article'
    idx = Column(Integer)
    title = Column(String(50), primary_key=True)
    url = Column(String(50))
    author = Column(String(20))
    release_date = Column(Date)
    source = Column(String(20))
    content = Column(Text)


class newsSpiderBase(object):
    def __init__(self, header, path):
        self.mysql = mysqlBase(header)
        self.url_list = []
        self.href = []
        self.article_set = []
        self.path = path + 'config/'

    def save_process(self):
        url_file = self.path + 'URL_LIST'
        with open(url_file, 'w') as f:
            for url in self.url_list:
                f.write(str(url) + '\n')
        href_file = self.path + 'HREF_LIST'
        with open(href_file, 'w') as f:
            for url in self.href:
                f.write(str(url) + '\n')

    def load_href_file(self, href_file):
        try:
            with open(href_file, 'r') as f:
                url = f.readline()
                # print(url)
                self.href.append(url)
        except Exception as e:
            print(e)


class neteaseNewsSpider(newsSpiderBase):
    def generate_url_list(self):
        self._chn_list()
        self._hk_list()
        self._us_list()
        self._ipo_list()
        self._fund_list()
        self._future_list()
        self._kcb_list()
        self._forexchange_list()
        self._chairman_list()
        self._bc_list()
        self._business_list()
        self._house_list()

    def _chn_list(self):
        chn_start_url = "https://money.163.com/special/002557S6/newsdata_gp_index.js?callback=data_callback"
        self.url_list.append(chn_start_url)
        chn_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_index_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += chn_url_list

    def _hk_list(self):
        hk_start_url = "https://money.163.com/special/002557S6/newsdata_gp_hkstock.js?callback=data_callback"
        self.url_list.append(hk_start_url)
        hk_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_hkstock_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += hk_url_list

    def _us_list(self):
        us_start_url = "https://money.163.com/special/002557S6/newsdata_gp_usstock.js?callback=data_callback"
        self.url_list.append(us_start_url)
        us_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_usstock_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += us_url_list

    def _ipo_list(self):
        ipo_start_url = "https://money.163.com/special/002557S6/newsdata_gp_ipo.js?callback=data_callback"
        self.url_list.append(ipo_start_url)
        ipo_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_ipo_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += ipo_url_list

    def _future_list(self):
        qhzx_start_url = "https://money.163.com/special/002557S6/newsdata_gp_qhzx.js?callback=data_callback"
        self.url_list.append(qhzx_start_url)
        qhzx_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_qhzx_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += qhzx_url_list

    def _forexchange_list(self):
        forex_start_url = "https://money.163.com/special/002557S6/newsdata_gp_forex.js?callback=data_callback"
        self.url_list.append(forex_start_url)
        forex_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_forex_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += forex_url_list

    def _bc_list(self):
        bitcoin_start_url = "https://money.163.com/special/002557S6/newsdata_gp_bitcoin.js?callback=data_callback"
        self.url_list.append(bitcoin_start_url)
        bitcoin_url_list = [f"https://money.163.com/special/002557S6/newsdata_gp_bitcoin_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += bitcoin_url_list

    def _kcb_list(self):
        kcb_start_url = "http://money.163.com/special/00259D2D/fund_newsflow_hot.js?callback=data_callback"
        self.url_list.append(kcb_start_url)
        kcb_url_list = [f"http://money.163.com/special/00259D2D/fund_newsflow_hot_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += kcb_url_list

    def _fund_list(self):
        fund_start_url = "http://money.163.com/special/00259CPE/data_kechuangban_kechuangban.js?callback=data_callback"
        self.url_list.append(fund_start_url)
        fund_url_list = [f"http://money.163.com/special/00259CPE/data_kechuangban_kechuangban_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += fund_url_list

    def _chairman_list(self):
        chairman_start_url = "http://money.163.com/special/00259CTD/data-yihuiman.js?callback=data_callback"
        self.url_list.append(chairman_start_url)
        chairman_url_list = [f"http://money.163.com/special/00259CTD/data-yihuiman_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += chairman_url_list

    def _business_list(self):
        business_start_url = "http://money.163.com/special/002557RF/data_idx_shangye.js?callback=data_callback"
        self.url_list.append(business_start_url)
        business_url_list = [f"http://money.163.com/special/002557RF/data_idx_shangye_0{i}.js?callback=data_callback" for i in range(2, 10)]
        self.url_list += business_url_list

    def _house_list(self):
        house_start_url = "http://money.163.com/special/002534NU/house2010.html"
        self.url_list.append(house_start_url)
        house_url_list = [f"http://money.163.com/special/002534NU/house2010_{str(i).zfill(2)}.html" for i in range(2, 21)]
        self.url_list += house_url_list

    def extract_href(self, url):
        resp = requests.get(url)
        result = re.findall(
            r'\"docurl\":\"(http.://money.163.com/\d{2}/\d{4}/\d{2}/\w+\.html)\"',
            resp.text)
        self.href += result

    def extract_article(self, url):
        try:
            art = article()
            text = requests.get(url)
            h = etree.HTML(text.text)
            content = h.xpath("//div[@class='post_text']/p/text()")
            art.url = url
            art.title = art._get_title(h)
            art.author = art._get_author(h)
            art.date = art._get_date(h)
            art.source = art._get_source(h)
            content = h.xpath("//div[@class='post_text']/p")
            art.content = art._text_clean(content)
        except Exception as e:
            ERROR("Extract article failed.")
            ERROR(e)
        return art

    def record_article(self, art):
        try:
            insert_data = {
                'title': f"{art.title}",
                'url': f"{art.url}",
                'release_date': f"{art.date}",
                'author': f"{art.author}",
                'source': f"{art.source}",
                'content': f"{art.content}"
                }
            self.mysql.session.execute(
                formArticle.__table__.insert().prefix_with('IGNORE'),
                insert_data)
            self.mysql.session.commit()
        except Exception as e:
            print(e)


class SinaNewsSpider(newsSpiderBase):
    pass


if __name__ == "__main__":
    url = 'https://money.163.com/20/0725/18/FIDBIU90002580S6.html'
    cont = requests.get(url)
    html = lxml.etree.HTML(cont.text)
    art = NeteaseArticle(html, url)
    # art.save_content_text(text_content, '/home/friederich/Documents/test/Test2.html')
    art.article_to_json('/home/friederich/Documents/article.json')
    print('end')

