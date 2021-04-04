# elasticesearch_python
simple python search engine for csv-movie (every row is a movie with severa details) using elasticsearch

data download: https://www.kaggle.com/jrobischon/wikipedia-movieplots?select=wiki_movie_plots_deduped.csv .

tools/packages: Elasticsearch - ES, Kibana for visualization.
language: python

use ES to index files, to search with a query processed by TF-IDF, and then to parse the retrieved recalls.
用ES去索引文件，然后用语句去搜索，语句经过TF-IDF简单处理，然后对搜索到结果进行解析

data is files from kaggle, movie files, every row is a movie than includes several features such as title, plot, release year, and so on.
数据是电影文件，每一行都是一个电影，电影有几个属性：标题，内容，发布时间等等
