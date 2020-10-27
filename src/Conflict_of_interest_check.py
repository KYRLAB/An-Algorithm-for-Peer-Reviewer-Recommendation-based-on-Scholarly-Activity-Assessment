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

def interest(co_author_path, reviewer_information_path, co_author_network_path, professionalism_result, extractive_keyword_result, reviewee_index,matrix_multifly_count):
    
    path1,path2=co_author_path,reviewer_information_path
    network_path=co_author_network_path
    temp,reviewee = professionalism_result,extractive_keyword_result
    index,multifly=reviewee_index,matrix_multifly_count

    co_author_csv = pd.read_csv(path1, encoding='latin1')
    
    co_author_df = co_author_csv.merge(temp, on=['reviewer_orcid'])  
    co_author_df2 = co_author_df.iloc[:]['reviewer_name'].tolist()
    
    try :

        network_csv = pd.read_csv(network_path, encoding='latin1',index_col=0)
        
    except FileNotFoundError :
        
        df1=pd.read_csv('./network/network0704.csv')
        df2=pd.read_csv('./network/network0704_2.csv')

        tmp=[1]
        tmp.extend(i*2 for i in range(1,11))

        for k in range(1,len(df1.columns)):
            a=df1.columns[k]
            a1=df2.loc[df2['reviewer_coauthor_title']==a]
            a_list=[]
            for i in range(len(tmp)):
                a_list.append(a1.iloc[0][tmp[i]])
            for i in range(len(df1)):
                if (df1.iloc[i,0] in a_list) is True :df1.iloc[i,k]=1
                else :df1.iloc[i,k]=0

        mat=(df1.values)[:,1:]
        mat2=mat.T
        mat3=np.dot(mat,mat2)
        for i in range(len(mat3)):
            mat3[i,i]=0
        fof=np.dot(mat3,mat3)
        
        df1_index=(df1.iloc[:,0]).tolist()
        df1_col=(df1.columns)[1:]
        network_csv=pd.DataFrame(data=mat3,index=df1_index,columns=df1_index)
        network_csv.to_csv(network_path)
    
    reviewer_list=(network_csv.index).tolist()
    
    reviewee_list=[]
    reviewee.fillna(0, inplace=True)
    for i in range(1,11):
        col_index = (i*3)+5
        if reviewee.loc[index][col_index] != 0:
            reviewee_list.append(reviewee.loc[index][col_index])
    
    co_rel_df = pd.DataFrame(
        columns=[i for i in reviewer_list],
        index=[j for j in reviewee_list])
    
    for i in range(len(reviewee_list)):
        indet_temp=reviewer_list.index(reviewee_list[i])
        co_rel_df.iloc[i,indet_temp]=1
    co_rel_df.fillna(0, inplace=True)
    
    network_csv_v=network_csv.values
    for i in range(multifly):
        network_csv_v = network_csv_v.dot(network_csv_v)

    co_rel_df_v=co_rel_df.values
    fof = co_rel_df_v.dot(network_csv_v)
    
    df_fof=pd.DataFrame(data=fof,
                 index=(co_rel_df.index).tolist(),
                 columns=(network_csv.index).tolist())
    
    df_fof_series =df_fof.loc[:, (df_fof != 0).any(axis=0)]
    df_fof_series_list=(df_fof_series.columns).tolist()
    
    reviewer_list1 = list(set(co_author_df2).difference(df_fof_series_list))
    
    co_inst_csv = pd.read_csv(path2, encoding='latin1')

    co_inst_df = co_inst_csv.merge(temp, on=['reviewer_orcid'])

    reviewee_list2=[]
    for i in range(1,11):
        col_index = (i*3)+6
        if reviewee.loc[index][col_index] != 0:
            reviewee_list2.append(reviewee.loc[index][col_index])
    
    reviewer_list2,reviewer_inst_list=[],[]
    for j in range(len(co_inst_df)):
        inst_list_temp=[]
        reviewer_list2.append(co_inst_df['reviewer_name'][j])
        reviewer_inst_list.append(co_inst_df['reviewer_institution'][j])

    inst_rel_df = pd.DataFrame(
        columns=[i for i in reviewer_list2],
        index=[j for j in reviewee_list])

    for i in range(len(reviewee_list2)):
        for j in range(len(reviewer_inst_list)):
            if (reviewee_list2[i] == reviewer_inst_list[j]) : inst_rel_df.iloc[i, j] = 1
            else : inst_rel_df.iloc[i, j] = 0
    
    inst_rel_df_series = inst_rel_df.loc[:, (inst_rel_df != 0).any(axis=0)]
    inst_rel_df_series_list=(inst_rel_df_series.columns).tolist()
    
    reviewer_list2 = list(set(reviewer_list2).difference(inst_rel_df_series_list))
    
    reviewer_rank_list = list(set(reviewer_list1).intersection(reviewer_list2))

    id_index,sim_index,count_index,title_index=[],[],[],[]
    reviewer_rank = pd.DataFrame({'reviewer_name': reviewer_rank_list})
    
    for i in range(len(reviewer_rank)):
        for j in range(len(co_author_df)):
            if reviewer_rank.loc[i]['reviewer_name'] == co_author_df.loc[j]['reviewer_name'] :
                id_index.append(int(co_author_df.iloc[j]['reviewer_orcid']))
                sim_index.append(co_author_df.iloc[j]['sim'])
                title_index.append(co_author_df.iloc[j]['reviewer_title'])
                break
            if reviewer_rank.loc[i]['reviewer_name'] == co_inst_df.loc[j]['reviewer_name'] :
                count_index.append(co_inst_df.iloc[j]['count'])
    
    last_count=[]
    for i in range(len(reviewer_rank)):
        last_count.append(0)
    
    reviewer_rank['reviewer_orcid']=id_index
    reviewer_rank['title']=title_index
    reviewer_rank['sim']=sim_index
    reviewer_rank['count']=last_count
    
    reviewer_rank2 = reviewer_rank.sort_values(by=['sim'], axis=0, ascending=False)
                
    return reviewer_rank2

# reviewer_rank = interest(
#     co_author_path='./reviewer_pool/reviewer_coauthor.csv',
#     reviewer_information_path='./reviewer_pool/reviewer_information.csv',
#     co_author_network_path='./co_author_network/network.csv',
#     professionalism_result=reviewer,
#     extractive_keyword_result=reviewee,
#     reviewee_index=0,
#     matrix_multifly_count=2)