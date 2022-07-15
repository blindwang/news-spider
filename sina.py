import requests
import json
import xlwt
from bs4 import BeautifulSoup
import os
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


class Sina:
    def __init__(self, keyword):
        self.kw = keyword
        self.path = os.path.join('data', self.kw)
        self.news = []
        self.sheet_save_path = os.path.join('data', '汇总')

    def getData(self, page):
        headers = {"Host": "interface.sina.cn",
                   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0",
                   "Accept": "*/*", "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
                   "Accept-Encoding": "gzip, deflate, br", "Connection": "keep-alive",
                   "Referer": r"http://www.sina.com.cn/mid/search.shtml?range=all&c=news&q=%E6%97%85%E6%B8%B8&from=home&ie=utf-8",
                   "Cookie": "ustat=__172.16.93.31_1580710312_0.68442000; genTime=1580710312; vt=99; Apache=9855012519393.69.1585552043971; SINAGLOBAL=9855012519393.69.1585552043971; ULV=1585552043972:1:1:1:9855012519393.69.1585552043971:; historyRecord={'href':'https://news.sina.cn/','refer':'https://sina.cn/'}; SMART=0; dfz_loc=gd-default",
                   "TE": "Trailers"}

        params = {"t": "", "q": self.kw, "pf": "0", "ps": "0", "page": page, "stime": "2020-01-01",
                  "etime": "2022-07-13",
                  "sort": "rel", "highlight": "1", "num": "10", "ie": "utf-8"}

        if os.path.exists(self.path) is False:
            os.makedirs(self.path)
        response = requests.get("https://interface.sina.cn/homepage/search.d.json?", params=params, headers=headers)
        # print(response.request.url)
        dic = json.loads(response.text)
        self.news += dic["result"]["list"]

    def writeData(self):
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('MySheet')

        worksheet.write(0, 0, "标题")
        worksheet.write(0, 1, "时间")
        worksheet.write(0, 2, "媒体")
        worksheet.write(0, 3, "网址")
        worksheet.write(0, 4, "关键词")
        url_lst = []
        for i in range(len(self.news)):
            # print(news[i])
            url = self.news[i]["url"]
            # 去重
            if url in url_lst:
                continue
            url_lst.append(url)

            title = self.news[i]["origin_title"]
            worksheet.write(i + 1, 0, title)
            worksheet.write(i + 1, 1, self.news[i]["datetime"])
            worksheet.write(i + 1, 2, self.news[i]["media"])
            worksheet.write(i + 1, 3, url)
            if url.startswith('https://finance.sina.com.cn'):
                self.extract_news(url, title, 'f')
            if url.startswith('https://k.sina.com.cn'):
                self.extract_news(url, title, 'k')
            try:
                content = open(os.path.join(self.path, title + '.txt'), 'rb').read()
                tags = jieba.analyse.extract_tags(content, topK=4)
                if self.kw in tags:
                    tags.remove(self.kw)
                worksheet.write(i + 1, 4, ",".join(tags))
            except:
                pass

        if os.path.exists(self.sheet_save_path) is False:
            os.makedirs(self.sheet_save_path)
        workbook.save(os.path.join(self.sheet_save_path, self.kw + '.xls'))

    def extract_news(self, url, title, type):
        header = {'user-agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=header)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            # print(r.text)
            soup = BeautifulSoup(r.text, features='lxml')
            news = []
            if type == 'f':
                arti = soup.find_all(id="artibody")
                for para in arti[0].find_all('p'):
                    if para.string is not None:
                        news.append(para.string.strip())
            elif type == 'k':
                arti = soup.find_all(id="article")
                news.append(arti[0].text.strip())

            with open(os.path.join(self.path, title + '.txt'), 'w', encoding='utf-8') as f:
                f.write(title + '\n')
                f.writelines(news)
                print("Successfully save 《{}》!".format(title))
        except:
            print("error {}".format(url))

    def labels_to_original(self, labels, forclusterlist):
        assert len(labels) == len(forclusterlist)
        maxlabel = max(labels)
        numberlabel = [i for i in range(0, maxlabel + 1, 1)]
        numberlabel.append(-1)
        result = [[] for i in range(len(numberlabel))]
        for i in range(len(labels)):
            index = numberlabel.index(labels[i])
            result[index].append(forclusterlist[i])
        return result

    def classify_tag(self):
        # 保存目录
        save_path = os.path.join('data', '关键词聚类')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)

        # 分类数
        num = 5
        df = pd.read_excel(os.path.join(self.sheet_save_path,self.kw + '.xls'))
        df = df.fillna('')
        corpus = [tags for tags in df['关键词']]
        # txt = open(self.kw+"关键词.txt", "r", encoding='utf-8').read().split("\n")
        # corpus = []
        # for tags in txt:
        #     corpus.append(tags)
        # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        vectorizer = CountVectorizer(max_features=20000)
        # 该类会统计每个词语的tf-idf权值
        tf_idf_transformer = TfidfTransformer()
        # 将文本转为词频矩阵并计算tf-idf
        tfidf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(corpus))
        # 获取词袋模型中的所有词语
        tfidf_matrix = tfidf.toarray()
        # 获取词袋模型中的所有词语
        word = vectorizer.get_feature_names()
        # print(word)
        # # 统计词频
        # print(tfidf)
        # 聚成num类
        clf = KMeans(n_clusters=num)
        s = clf.fit(tfidf_matrix)

        # 每个样本所属的簇
        label = []
        i = 1
        while i <= len(clf.labels_):
            label.append(clf.labels_[i - 1])
            i = i + 1
        # 获取标签聚类
        y_pred = clf.labels_

        # pca降维，将数据转换成二维
        pca = PCA(n_components=2)  # 输出两维
        newData = pca.fit_transform(tfidf_matrix)  # 载入N维

        xs, ys = newData[:, 0], newData[:, 1]

        df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=corpus))
        groups = df.groupby('label')

        res = self.labels_to_original(y_pred, corpus)
        label = []
        for i in range(len(res)):
            if len(res[i]) > 0:
                label.append(res[i][0])
            else:
                label.append(res[i])
        # 设置类名
        cluster_names = dict(zip(range(len(res)), label))
        # print(cluster_names)
        # 解决中文乱码
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 设置颜色
        cmap = plt.cm.get_cmap('hsv', len(res))

        # 画图
        fig, ax = plt.subplots(figsize=(8, 5))  # set size
        ax.margins(0.02)
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=cluster_names[name],
                    color=cmap(name), mec='none')
            plt.legend()
        plt.title(self.kw + "关键词聚类分布图")
        plt.savefig(os.path.join(save_path, self.kw + "关键词聚类"))
        plt.show()

        # print(res)
        for i in range(len(res)):
            for j in range(min(len(res[i]), 5)):
                print(res[i][j])
            print("=" * 20)

    def run(self):
        for i in range(1, 20):
            self.getData(i)
        self.writeData()


if __name__ == '__main__':
    # kw = ['女权', '家暴', '女性劳动者', '生育', '大女主剧', '离婚']
    kw = ['三孩', '生育友好', '优生优育']
    for k in kw:
        sina = Sina(k)
        sina.run()
        sina.classify_tag()