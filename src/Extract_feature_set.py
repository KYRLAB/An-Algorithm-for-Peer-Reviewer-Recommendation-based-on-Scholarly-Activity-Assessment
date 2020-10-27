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

def extractive_keyword(path,database_update_path,extract_word_num=20):
    
    reviewee = pd.read_csv(path, encoding='latin1')
    count,temp = len(reviewee),[]

    for i in range(count):
        
        temp_intro = reviewee['submitter_intro'][i]

        textrank_text = ''

        for c in (keywords(temp_intro, words=extract_word_num, lemmatize=True).split('\n')):
            
            textrank_text += (c+ " ")

        temp.append(textrank_text)

    reviewee['submitter_attribute']=temp
    
    #return type : pandas.dataframe
    return reviewee

# extractive_keyword(path='./submitter/submitter.csv',
#                     database_update_path='./submitter/submitter.update.csv',
#                     extract_word_num=20)