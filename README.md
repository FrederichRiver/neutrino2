# 关于Neutrino
作者：Friederich River

email：hezhiyuan_tju@163.com

Neutrino是一个量化交易项目，目的在于建立一个监控宏观微观经济的全天候股票交易体系。它采用云计算架构，可以实现以下量化功能：

1. 自动数据采集功能：应用爬虫技术，未来将购买商业化数据接口，实现宏观、微观经济数据采集，金融品价格数据采集，新闻等非结构化数据采集等功能。
2. 自动化分析功能：每日按照任务流水线的模式，顺序执行各种数据分析功能，对交易机会进行分析。监控大数据，实现实时预警。
3. 自动通知功能：一方面是数据分析结果，以日报、周报的方式，向客户发送邮件。二是实时监测的大数据事件，实现实时预警。未来将进一步开发app接收。
4. 回测功能：针对各种量化交易算法，实施回测。
5. AI功能：应用HMM、LSTM、GA、NLP、KG等人工智能技术，对各种基本功能进行增强。

应用以上技术最终实现一个智能的投资顾问，并在云计算上运行。

TO DO:

1. 基于GaussianHMM的隐马尔可夫模型，预测市场类型。
2. 基于NLP的事件提取器。
3. 行业分类机。
    两大行业分类标准：ICB(Industry Classification Benchmark)和GICS(Global Industry Classification Standard)
    在Neo4j上建立GICS行业分类。
    将企业与GICS进行关联，通过NLP进行首次关联。
    对关重企业进行算法和人工调整。
4. 关系数据库 + 知识图谱。
    建立Neo4j数据库及驱动引擎
    Neo4j应用
    Install：    从neo4j的官方网站下载neo4j-server的安装包，解压缩后直接cp到/opt目录
    Run： /opt/neo4j/bin/neo4j start 运行server
    Visit： http://localhost:7474 可访问neo4j server
    Operation: 使用Cypher语言进行查询。
