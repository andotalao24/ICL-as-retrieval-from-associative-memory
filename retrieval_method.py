import socket
import pandas as pd
import os
import socket
import math
import json
import numpy as np
from rank_bm25 import BM25Okapi


from concurrent.futures import ThreadPoolExecutor
local_ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))

from sentence_transformers import SentenceTransformer


def cos_dis(em1,em2):
    #larger, the better
    return np.dot(em1,em2)/(np.linalg.norm(em1)*np.linalg.norm(em2))

def dot_dis(em1,em2):
    #larger, the better
    return np.dot(em1,em2)

def elucid_dis(em1,em2):
    #smaller, the better
    return np.sqrt(np.sum((em1-em2)**2))



def getTopSent(eval_dt,train_dt,k):
    corpus=[d['que'] for d in train_dt]
    query=[d['que'] for d in eval_dt]
    model_sent = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_sent.encode(corpus)
    query_embeddings = model_sent.encode(query)
    #Print the embeddings
    ret=[]
    for query_embedding in query_embeddings:
        sortSent = dict()
        for i,sent in enumerate(embeddings):
            d=cos_dis(query_embedding,sent)
            sortSent[i]=d
        sortSent=sorted(sortSent.items(), key=lambda x: x[1],reverse=True)
        ret.append(train_dt[i[0]] for i in sortSent[:k])
    return ret



def getTopBM25(eval_dt,train_dt,k):
    corpus=[d['que'] for d in train_dt]
    querys=[d['que'] for d in eval_dt]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    ret=[]
    for query in querys:
        tokenized_query = query.split(" ")
        sents=bm25.get_top_n(tokenized_query, corpus, n=k)
        ret.append(train_dt[corpus.index(sent)] for sent in sents)
    return ret