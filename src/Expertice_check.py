#-*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples

from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import *
from sklearn.cluster import *

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from operator import itemgetter
from operator import attrgetter

from pyjarowinkler import distance
from collections import Counter

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import nltk

import math
import time

import csv
import sys

import re
import io
import os

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.cluster import AgglomerativeClustering
import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")

def professionalism(path,extractive_keyword_result,reviewee_index,top_limit):
    
    reviewee,index,top=extractive_keyword_result,reviewee_index,top_limit
    temp_id,temp_doi = 0,''
    
    temp_title = reviewee.loc[index]['submitter_title']
    temp_attribure = reviewee.loc[index]['submitter_attribute']
    
    reviewer_attr = pd.read_csv(path, encoding='latin1')
    
    reviewer_attr.loc[-1] = [str(temp_id),temp_doi,temp_title,temp_attribure]
    reviewer_attr.index += 1
    reviewer_attr.sort_index(inplace=True)
    reviewer = reviewer_attr['reviewer_paper_attribure']
    
    jac_token,jac,cos,avg=[],[],[],[]

    for t in range(len(reviewer)):        
        jac_token.append(set(nltk.ngrams((nltk.word_tokenize(reviewer[t])), n=1)))  
        
    for j in range(len(reviewer)):
        jac.append(1-(nltk.jaccard_distance(jac_token[0], jac_token[j])))

    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(reviewer)
    doc_term_matrix = sparse_matrix.todense()
    
    df = pd.DataFrame(doc_term_matrix, 
                      columns=count_vectorizer.get_feature_names(), 
                      index=[i for i in reviewer])
    cos=cosine_similarity(df, df)[0].tolist()

    for i in range(len(jac)):
        avg.append((jac[i] + cos[i])/2)
        
    reviewer_attr['sim']=avg
    
    reviewer_attr2 = reviewer_attr.sort_values(by=['sim'], axis=0, ascending=False)
    reviewer_attr3=reviewer_attr2[~reviewer_attr2.duplicated(['reviewer_orcid'],keep=False) == True]
    reviewer_attr4 = reviewer_attr2[reviewer_attr2.duplicated(['reviewer_orcid'],keep='last') == True]
    reviewer_attr5=pd.concat([reviewer_attr3,reviewer_attr4])
    reviewer_attr5.rename(columns = {'reviewer_paper_attribure' : 'reviewer_paper_feature'}, inplace = True)
    reviewer2 = list(reviewer_attr5['reviewer_paper_feature'])
    
    reviewer_attribute=[]
    for i in range(len(reviewer)):
        a=((reviewer[i]).split(' '))
        b=a[:20]
        temp=[]
        for j in range(20):
            temp.append(b[j])
            temp += (str(b[j]) + ',')
        reviewer_attribute.append(temp[:-1])
    
    common_texts_and_tags = [
        (text, [f"str_{i}",]) for i, text in enumerate(reviewer_attribute)
    ]
    
    TRAIN_documents = [TaggedDocument(words=text, tags=tags) for text, tags in common_texts_and_tags]
    
    model = Doc2Vec(TRAIN_documents)
    
    for text, tags in common_texts_and_tags:
        trained_doc_vec = model.docvecs[tags[0]]
        inferred_doc_vec = model.infer_vector(text)

    dtm_df=[]
    for text, tags in common_texts_and_tags:

        inferred_doc_vec = model.infer_vector(text)

        dtm_df_temp=[]
        for text2, tag2s in common_texts_and_tags:

            inferred_doc_vec = model.infer_vector(text2)

            sim = word_vectors.wmdistance(text, text2)

            dtm_df_temp.append(sim)
        dtm_df.append(dtm_df_temp)

    dissim_df = pd.DataFrame(data=dtm_df)
    dissimilarity=dissim_df.values

    model1 = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
    model1.fit(dissimilarity)
    labels1 = model1.labels_
    dissim_df['cluster']=labels1
    
    cluster_reviewer = reviewer_attr5[reviewer_attr5['cluster'] == reviewer_attr5.loc[0]['cluster']]

    cluster_reviewer2 = cluster_reviewer.sort_values(by=['sim'], axis=0, ascending=False)
    
    professionalism=cluster_reviewer2.iloc[0:top+1]
    
    return dissim_df

professionalism(path='./reviewer_pool/reviewer_attribute.csv',
                extractive_keyword_result=reviewee,
                reviewee_index=0,
                top_limit=20,
                silhouette_range=25)