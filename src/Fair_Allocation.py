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

def multisort(xs, specs):
    
    for key, reverse in reversed(specs):
        
        xs.sort(key=attrgetter(key), reverse=reverse)
        
    return xs

def save_csv(output_path,extractive_keyword_result,professionalism_result,reviewee_index,top_limit):
    
    path,ee_num,top=output_path,reviewee_index,top_limit
    reviewee,reviewer_rank_name=extractive_keyword_result,professionalism_result
    
    export_data=[]
    for i in range((top*2)):

        temp=[]
        temp.append(reviewee.iloc[(1//top*2)+ee_num]['submitter_title'])
        temp.append(reviewee.iloc[(1//top*2)+ee_num]['date'])
        temp.append(reviewee.iloc[(1//top*2)+ee_num]['submitter_name'])

        temp.append(reviewer_rank_name.iloc[i]['reviewer_name'])
        temp.append(reviewer_rank_name.iloc[i]['reviewer_orcid'])
        temp.append(reviewer_rank_name.iloc[i]['title'])
        temp.append(reviewer_rank_name.iloc[i]['sim'])
        temp.append(reviewer_rank_name.iloc[i]['count'])

        export_data.append(temp)
        
    try :
        export_csv = pd.read_csv(path,index_col=0)
    except FileNotFoundError :
        export_csv = pd.DataFrame([],columns=['submitter_title','date','submitter_name','reviewer_name','reviewer_orcid','title','sim','count'])
    
    for i in range(len(export_data)):
        export_csv.loc[len(export_csv)] = export_data[i]
    
    export_csv2 = export_csv.sort_values(by=['sim'], axis=0, ascending=False)
    
    export_csv2.to_csv(path)
    
def equl_distribution(input_csv_path, output_csv_path):
    
    export_csv2 = pd.read_csv(input_csv_path,index_col=0)
    
    class Paper:
        
        def __init__(self, title, date, submitter, reviwer_name, reviwer_orcid, title2, sim, count):
            self.title = title
            self.date = date
            self.submitter = submitter
            self.reviwer_name = reviwer_name
            self.reviwer_orcid = reviwer_orcid
            self.reviwer_title = title2
            self.sim = sim
            self.count = count

        def __repr__(self):
            return repr((self.title, self.date, self.submitter, self.reviwer_name, self.reviwer_orcid, self.reviwer_title, self.sim, self.count))

    papers,objs=[export_csv2.iloc[i].tolist() for i in range(len(export_csv2))],[]

    for paper in papers:
        objs.append(Paper(*paper))
    
    o = (multisort(list(objs), (('date', False), ('sim', True))))
    
    final_list=[]
    for i in range(0,len(export_csv2),6) :
        temp_list=[]
        for t in range(6):
            if len(temp_list) == 3:break
            else :
                temp = i + t
                if (o[temp].count) < 3 :
                    o[temp].count += 1
                    for j in range(0+temp, len(export_csv2)) :
                        if (o[temp].reviwer_name == o[j].reviwer_name) :
                            o[j].count += 1
                    o[temp].count -= 1
                    class_1=(str(o[temp]))[1:-1]
                    class_2=class_1.replace('\'','')
                    class_3=class_2.split(', ')
                    temp_list.append(class_3)
        final_list.extend(temp_list)
        
    final=pd.DataFrame(final_list,columns=['submitter_title','date','submitter_name','reviewer_name','reviewer_orcid','title','sim','count'])
    final.to_csv(output_csv_path)
    
    
# save_csv(output_path='./algorithm_output/export_csv.csv',
#          extractive_keyword_result=reviewee,
#          professionalism_result=reviewer_rank,
#          reviewee_index=0,
#          top_limit=3)

# equl_distribution(input_csv_path='./algorithm_output/export_csv.csv',
#                  output_csv_path='./algorithm_output/final_csv.csv')