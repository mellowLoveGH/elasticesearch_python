from elasticsearch import helpers, Elasticsearch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


path = "wiki_movie_plots_deduped.csv"
df = pd.read_csv(path)
#print( df.head() )
print( "whole data: shape", df.shape )

# dataframe cols/features
def DFcols(df):
    cols = []
    for col in df:
        cols.append(col)
    #print( cols )
    return cols

cols = DFcols(df)
print( "columns: ", cols )

# sample of 1000 articles, randomly
num = 1000
sample = df.sample(num, random_state=6)
#print( sample.head() )
print( "sample data: shape", sample.shape )

def basicInfo(df, verbose=True):
    RY_range = list( set( df['Release Year'] ) )
    origin_range = list( set( df['Origin/Ethnicity'] ) )
    genre_range = list( set( df['Genre'] ) )
    gen = set()
    for i in genre_range:
        if '/' in i:
            tmp = i.split('/')
            for w in tmp:
                gen.add(w.strip())
        elif ',' in i:
            tmp = i.split(',')
            for w in tmp:
                gen.add(w.strip())
        else:
            gen.add(i)
    genre_range = list( gen )
    if verbose:
        print("Release Year: \t", RY_range)
        print("Origin/Ethnicity: \t", origin_range)
        print("Genre: \t", genre_range)
    return RY_range, origin_range, genre_range

print("the basic info about those movie collection: \n")
basicInfo(sample)
print()


from datetime import datetime

# connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


# 
# indexing
# 
print( "indexing documents ..." )


# index name, doc type
my_index = 'ir_hw'
my_doc_type = 'movie'
print( "index name: ", my_index, "\t doc type: ", my_doc_type)

# the document is mapping as the following structure
mapping = {
    "properties": {
        "id":{ "type":"long" },
        "Release Year": { "type": "keyword" },
        'Title':{
            "type": "text", "fields": { "field": { "type": "keyword" }                
            }
        },
        'Origin/Ethnicity':{
            "type": "text", "fields": { "field": { "type": "keyword" }                
            }
        },
        'Director': {
            "type": "text", "fields": { "field": { "type": "keyword" }                
            }
        },
        'Cast': {
            "type": "text", "fields": { "field": { "type": "keyword" }                
            }
        },
        'Genre': {
            "type": "text", "fields": { "field": { "type": "keyword" }                
            }
        },
        'Wiki Page': {
            "type": "keyword"
        },
        'Plot': {
            "type": "text", "fields": { "field": { "type": "keyword" }                
            }
        }
    }
}

# create the index with doc type, put the mapping
es.indices.delete(index=my_index, ignore=[400, 404])
es.indices.create(index=my_index, ignore=400)
es.indices.put_mapping(index=my_index, doc_type=my_doc_type, body=mapping, include_type_name = True)

# there may be some NaN values in the sample data
sample = sample.replace(np.nan, '', regex=True) # deal with NaN
# upload a sample of 1000 articles with full text to Elasticsearch
index_list = []
for ind, row in sample[:num].iterrows():
    my_doc = row.to_dict()
    es.index(index=my_index, doc_type=my_doc_type, id=ind, body=my_doc) #, ignore=400)
    index_list.append( ind )
#print( index_list[:5] )
print( "upload a sample of 1000 articles with full text to Elasticsearch" )
print()

### used for test
"""
print( index_list[:10] )
idx = index_list[9]
print( idx )
res = es.get(index=my_index, id=idx)
test_plot = res['_source']['Plot']
print(test_plot) 
print()
"""

#
# Sentence Splitting, Tokenization and Normalization
#
print( "Sentence Splitting, Tokenization and Normalization ..." )

# remove stopwords and Tokenization 
def tokenization(es, inputText, analyzer="english"):
    #analyzer = ['english'] # stop
    res = es.indices.analyze(body={"analyzer" : analyzer,"text" : inputText})
    tokens = []
    for i in res['tokens']:
        #print(i['token'])
        tokens.append( i['token'] )
    return tokens

# Sentence Splitting
def senSplit(es, inputText, analyzer="english"):
    sen_dic = {}
    s_counter = 1
    sentence_delimiter = '. '
    sentences = inputText.split(sentence_delimiter)
    for sentence in sentences:
        sentence = tokenization(es, sentence, analyzer)
        if len(sentence) > 0:
            sen_dic[s_counter] = sentence
            s_counter += 1
    return sen_dic

# Normalization 


### used for test

"""
inputText = input("please input your query: \n")
print()
print(inputText)
"""

#
# Selecting Keywords
#
print("Selecting Keywords ...")

# form word-set from your input text
def termSet(sen_dic):
    ws = set()
    for i in sen_dic:
        ws = ws.union( set(sen_dic[i]) )
    return ws

# calculate the term frequency for every sentence
def termFre(ws, sen):
    tf = dict.fromkeys(ws, 0)
    for i in sen:
        tf[i] = tf[i] + 1
    doc_len = len(sen)
    for i in tf:
        tf[i] = tf[i] / doc_len
    return tf

import math

# IDF, calculate the idf for every word/token
def termIDF(ws, sen_dic):
    N = len( sen_dic )
    idf = dict.fromkeys(ws, 0)
    for i in idf:
        c = 0
        for j in sen_dic:
            if i in sen_dic[j]:
                c = c + 1
                #rint(i, sen_dic[j])
        idf[i] = math.log( N/c )
        #print( i )
    return idf

# calculate the weight for every word in every document/sentence
# sen_dic, dict that includes many sentences split by inputText
def calWeight(sen_dic):
    ws = termSet(sen_dic)
    idf = termIDF(ws, sen_dic)
    weights = {}
    for i in sen_dic:
        sen = sen_dic[i]
        tf = termFre(ws, sen)
        wgt = {}
        for j in tf:
            w = tf[j] * idf[j]
            if w > 0: # only reserve the terms whose weight > 0
                wgt[j] = w
        # order by weight:
        #wgt = sorted(wgt.items(),key=lambda x:x[1],reverse=True)
        wgt = dict(sorted(wgt.items(), key=lambda item: item[1],reverse=True))
        #print(wgt)
        weights[i] = wgt        
    return weights

from collections import Counter
# select keywords
def selectKeys(weights, top=10):
    keys = set()
    for i in weights:
        c = Counter( weights[i] )
        L = len(weights[i])
        if L > top:
            L = top
        most_common = c.most_common(L)
        tmp = [key for key, val in most_common]
        keys = keys.union( set(tmp) )
    keys = list( keys )
    return keys


### used for test
"""
print( "processed your input text and select keys from it as follows: " )
sen_dic = senSplit(es, inputText, "stop")
count = 0
for i in sen_dic:
    print( i, sen_dic[i] )
    count = count + len( sen_dic[i] )
print(count)
weights = calWeight(sen_dic)
keys = selectKeys(weights)
keys = " ".join(keys)
print( keys )
print()
"""

#
# Stemming or Morphological Analysis
#
print( "Stemming or Morphological Analysis" )
setting2 = {
  "settings": {
    "analysis": {
      "filter": {
        "english_stop": {
          "type":       "stop",
          "stopwords":  "_english_"
        },
        "light_english_stemmer": {
          "type":       "stemmer",
          "language":   "light_english" 
        },
        "english_possessive_stemmer": {
          "type":       "stemmer",
          "language":   "possessive_english"
        }
      },
      "analyzer": {
        "english": {
          "tokenizer":  "standard",
          "filter": [
            "english_possessive_stemmer",
            "lowercase",
            "english_stop",
            "light_english_stemmer", 
            "asciifolding" 
          ]
        }
      }
    }
  }
}


# closr first, then add settings, then open
es.indices.close(index=my_index)
es.indices.put_settings(index=my_index, body=setting2 )
es.indices.open(index=my_index)

print( "here is the used analysis filter & analyzer for following search: \t", setting2 )
print()


#
# Searching
#
print( "Searching ..." )

#queryText = keys #used for test
searchContent = [ 'Title', 'Plot', 'Director', 'Cast', 'Wiki Page']

# generate query to search, gievn input Text
def generateQuery(queryText, origin="", genre="", yearFrom=1900, yearTo=2022, searchContent=[ 'Title', 'Plot', 'Director', 'Cast', 'Wiki Page']):
    if len(queryText) == 0: # return all
        query_body = { "query":{ "match_all":{} } }
        return query_body
    
    # basic query
    query_body = { "query": {  "bool": {  "must": [ { "multi_match": { "query": queryText, "fields" : searchContent } } ],
          "filter": [ { "range": { "Release Year":{"gt":yearFrom, "lt":yearTo} }}  ] } } }
    
    # when user decide certain fields such as Origin/Ethnicity, Genre
    if len(origin) > 0:
        query_body["query"]["bool"]["filter"].append( { "match": { 'Origin/Ethnicity':  origin }} )
    if len(genre) > 0:
        query_body["query"]["bool"]["filter"].append( { "match": { 'Genre':  genre }} )
    return query_body

# search, print the top 10 recall results
def searching(es, my_index, querybody):
    result = es.search(index=my_index, body=querybody)
    recallNum = result['took']
    recallContent = result['hits']['hits']
    top = 10
    if top>recallNum:
        top = recallNum
    print( "find the top ", top, " most relevant results: \n" )
    for it in recallContent[:top]:
        print( "index: ", it['_id'], "\t relevant score: ", it['_score'] )
        content = it['_source']
        print( "Release Year: ", content['Release Year'], "\t Origin/Ethnicity: ", content['Origin/Ethnicity'] )
        print( "Genre: ",  content['Genre'] )
        print( "Title: ", content['Title'] )
        print( "Plot: ", content['Plot'][:100], "..." )
        print(  )
    return result

### used for test
"""
print( queryText )
querybody = generateQuery(queryText)
searching(es, my_index, querybody)
print()
querybody
"""

# 
# Engineering a Complete System
#

RY, orgin, genre = basicInfo(df, verbose=False)
yearFrom = RY[0]
yearTo = RY[-1]
origin = " ".join(orgin)

def analyzeToStr(es, text, analyzer="english"):
    re = es.indices.analyze(body={"analyzer" : analyzer, "text" : text })
    re = re['tokens']
    tmp = set()
    for it in re:
        tmp.add(it['token'])
    tmp = list(tmp)
    return " ".join(tmp)

genre = " ".join(genre)
"""

"""
genre = analyzeToStr(es, genre)
tmp = genre.split(' ')
gs = set()
for it in tmp:
    gs.add( it.strip() )
genre = " ".join(list( gs ))
#print(genre)

def process(text):
    tmp = text.split(',')
    return " ".join(tmp)

def showDetail(res):
    print("here is the detail about this article: ")
    print()
    detail = res['_source']
    for i in detail:
        print(detail[i])
    return

def retrieve(es, query, origin, genre, yearFrom, yearTo, genreBool=False):
    keys = ""
    if len(query) > 200:
        sen_dic = senSplit(es, query, "stop")
        weights = calWeight(sen_dic)
        keys = selectKeys(weights)
        keys = " ".join(keys)
    else:
        keys = analyzeToStr(es, query, analyzer="standard")
    #print( keys )
    if genreBool:
        querybody = generateQuery(keys, origin, genre, yearFrom, yearTo)
    else:
        querybody = generateQuery(keys, origin=origin, yearFrom=yearFrom, yearTo=yearTo)
    
    result = searching(es, my_index, querybody)
    return 

def launch(es, origin, genre, yearFrom, yearTo):
    # year range
    yearFiled = input( "Do you want to set the range of Release Year? Y/N: " )
    if 'Y' in yearFiled:
        yearFrom = int( input("please enter from which year, for example: 2000. ") )
        yearTo = int( input("please enter to which year, for example: 2020. ") )
    # Origin/Ethnicity
    originBool = False
    originFiled = input( "Do you want to set certain Origin/Ethnicity? Y/N: " )
    if 'Y' in originFiled:
        origin = input("please enter which origin(s), for example: Hong Kong, American, British...")
        origin = process(origin)
        originBool = True    
    # Genre
    genreBool = False
    genreFiled = input( "Do you want to set certain Genre? Y/N: " )
    if 'Y' in genreFiled:
        genre = input("please enter which genre(s), for example: drama, action, comedy, war, romantic...")
        genre = process(genre)
        genreBool = True
    #query / inputText
    query = input( "please input your query text: " )
    print()
    print()
    #
    retrieve(es, query, origin, genre, yearFrom, yearTo, genreBool)
    
    # see detail of certain article
    detail = input( "Do you want to see the detail about the movie? Y/N: " )
    res = ""
    if 'Y' in detail:
        id_num = int( input("please enter the index of the movie(, for example: 621, 11030...): ") )
        res = es.get(index=my_index, id=id_num)
        showDetail(res)
    return


# launch the program
stop = False

while not stop:
    print("================welcome to my IR: ================\n\n\n")
    launch(es, origin, genre, yearFrom, yearTo)
    it = input("Do you want to continue searching? Y/N: ")
    if 'Y' not in it:
        stop = True
 
print()
print()
print("thanks you!")






