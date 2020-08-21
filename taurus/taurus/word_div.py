#!/usr/bin/python38
from polaris.mysql8 import NLP_HEADER, mysqlBase
import jieba


class NLPBase(mysqlBase):
    def __init__(self, header) -> None:
        super(NLPBase, self).__init__(header)


class nlp(NLPBase):
    def __init__(self, header):
        super(nlp, self).__init__(header)

    def get_text(self):
        result = self.select_one('article', 'content', 'idx=138100')
        with open('/home/friederich/Downloads/tmp/text', 'w') as f:
            f.write(result[0])

    def read_text(self):
        with open('/home/friederich/Downloads/tmp/text', 'r') as f:
            result = f.read()
        return result

    def process_text(self, content: str):
        result = jieba.cut(content)
        print('|'.join(result))

    def posseg(self, content: str):
        import jieba.posseg as psg
        result = psg.cut(content)
        # print(' '.join([f"{w}/{t}" for w, t in result]))
        print(type(result))

    def get_stopword(self):
        stopword_dict = '/home/friederich/Documents/dict/stopwords.txt'
        swd = [line.strip() for line in open(stopword_dict, 'r').readlines()]
        print(swd)


if __name__ == "__main__":
    event = nlp(NLP_HEADER)
    content = event.read_text()
    event.posseg(content)
    # event.get_stopword()
