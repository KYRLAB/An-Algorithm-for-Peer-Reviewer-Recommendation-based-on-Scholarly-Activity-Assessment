#-*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.datasets import *
from sklearn.cluster import *
from gensim.summarization import keywords

import pandas as pd
import numpy as np

import csv, requests, json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import warnings
warnings.filterwarnings(action='ignore')

class peer_reviewer:
  def __init__(self, manuscript: dict):
    self.manuscript = manuscript
    self.submitted()

  def submitted(self):
    submitter_list=['date', 'submitter_orcid', 'submitter_name', 'submitter_institution',
        'submitter_email', 'submitter_title_doi', 'submitter_title', 'submitter_intro', 
        'submitter_author1_name','submitter_author1_institution', 'submitter_author1_email',
        'submitter_author2_name', 'submitter_author2_institution', 'submitter_author2_email',
        'submitter_author3_name','submitter_author3_institution', 'submitter_author3_email',
        'submitter_author4_name', 'submitter_author4_institution', 'submitter_author4_email',
        'submitter_author5_name', 'submitter_author5_institution', 'submitter_author5_email',
        'submitter_author6_name', 'submitter_author6_institution', 'submitter_author6_email', 
        'submitter_author7_name', 'submitter_author7_institution', 'submitter_author7_email',
        'submitter_author8_name', 'submitter_author8_institution', 'submitter_author8_email',
        'submitter_author9_name', 'submitter_author9_institution', 'submitter_author9_email',
        'submitter_author10_name', 'submitter_author10_institution', 'submitter_author10_email']
    submitter_df = pd.DataFrame(columns=[i for i in submitter_list])
    self.submitter_df = submitter_df.append(self.manuscript, ignore_index = True)

  def jane(self, jane_headers):
    jane_data = {
      'text': self.submitter_df["submitter_intro"],
      'languageCount': '7',
      'typeCount': '19',
      'openaccess': 'no preference',
      'pubmedcentral': 'no preference',
      'findAuthors': 'Find authors'}
    
    url = "https://jane.biosemantics.org/suggestions.php"
    res = requests.get(url)

    if res.status_code == requests.codes.ok:
        
        respond = requests.post(url, headers=jane_headers, data=jane_data)
        soup = BeautifulSoup(respond.text, 'html.parser')

        table_sub = soup.find_all('td', {'width':['100%']})

        jane_submitter=[]
        for i in table_sub:
            tmp=i.text.replace('\n','')
            jane_submitter.append(tmp)
        
        table_co = soup.find_all('table')
        df_jane_co = pd.DataFrame(pd.read_html(str(table_co)))
        self.jane_submitter = jane_submitter[:10]

        jane_submitter_co=[]
        for i in range(2,len(df_jane_co)):
            tmp=(((df_jane_co.iloc[i]).iloc[0]).loc[0,2])
            tmp=tmp.replace(', ',',')
            tmp=tmp.split(',')
            jane_submitter_co.append(tmp)
        
        self.jane_submitter_co = jane_submitter_co[:10]
        jane_submitter_url=[]
        for i in range(len(jane_submitter)):
          table_abst = soup.find_all('div', {'id':['info'+str(i)]})
          for j in table_abst[0].find_all('a', href=True):
              jane_submitter_url.append(j['href'])
        self.jane_submitter_url = jane_submitter_url[:10]

    else :
        print(res.status_code)

  def pubmed(self, pubmed_headers):
      url = "https://pubmed.ncbi.nlm.nih.gov"
      res = requests.get(url)

      jane_submitter_abst_org=[]
      if res.status_code == requests.codes.ok:

          for i in range(len(self.jane_submitter_url)) :

            response = requests.get(self.jane_submitter_url[i], headers=pubmed_headers)

            soup_abst = BeautifulSoup(response.text, 'html.parser')
            table_abst = soup_abst.find('div', {'class':['abstract-content selected']})
            tmp = (table_abst.find('p').getText())
            tmp = ' '.join((tmp).split('\n'))
            jane_submitter_abst_org.append(tmp.strip())

          self.jane_submitter_abst_org = jane_submitter_abst_org

      else :
          print(res.status_code)

  def orcid(self, orcid_headers):
    orcid_params = [
    ('q', '{!edismax qf="given-and-family-names^50.0 family-name^10.0 given-names^10.0 credit-name^10.0 other-names^5.0 text^1.0" pf="given-and-family-names^50.0" bq="current-institution-affiliation-name:[* TO *]^100.0 past-institution-affiliation-name:[* TO *]^70" mm=1}aakash goel'),
    ('start', '0'),
    ('rows', '50'),]

    url = 'https://pub.orcid.org/v3.0/expanded-search/'
    orcid_list=[]
    for i in range(len(self.jane_submitter)) :

      tmp = ['q']
      tmp2 = '{!edismax qf="given-and-family-names^50.0 family-name^10.0 given-names^10.0 credit-name^10.0 other-names^5.0 text^1.0" pf="given-and-family-names^50.0" bq="current-institution-affiliation-name:[* TO *]^100.0 past-institution-affiliation-name:[* TO *]^70" mm=1}'
      tmp.append(tmp2 + str(self.jane_submitter[i].lower()))

      orcid_params[0] = tuple(tmp)
      res = requests.get(url, headers=orcid_headers, params=tuple(orcid_params))

      if res.status_code == requests.codes.ok:
        soup_abst = BeautifulSoup(res.text, 'html.parser')
        soup_abst = json.loads(str(soup_abst))
        try :
          orcid_list.append(soup_abst['expanded-result'][0]['orcid-id'])
        except :
          orcid_list.append('0000-0000-0000-0000')
      else :
        print(res.status_code)
    self.orcid_list = orcid_list

  def feature_set(self):
    jane_submitter_abst, jane_except = [], []

    for i in range(len(self.jane_submitter_abst_org)) :
      try : jane_submitter_abst.append(' '.join(keywords(self.jane_submitter_abst_org[i], words=20, lemmatize=True).split('\n')))
      except : jane_except.append(i)

    for i in range(len(jane_except)-1,-1,-1):
      del self.jane_submitter[jane_except[i]]
      del self.jane_submitter_co[jane_except[i]]
      del self.orcid_list[jane_except[i]]

    textrank_text=' '.join(keywords((self.submitter_df['submitter_intro'][0]), words=20, lemmatize=True).split('\n'))
    self.submitter_df['submitter_attribute']=[textrank_text]
    self.jane_submitter_abst = jane_submitter_abst

  def reviewer_pool(self, path):
    network_csv = pd.read_csv(path, encoding='utf8', index_col=0)
    network_df = pd.DataFrame(index=self.jane_submitter,columns=[i for i in (network_csv.index)])
    
    for i in range(len(network_df.index)):
      tmp=[q for q, item in enumerate(network_df.columns) if item in set(self.jane_submitter_co[i])]
      for j in tmp:
        network_df.iloc[i,j] = 1
    network_df.fillna(0, inplace=True)

    network_csv=network_csv.append(network_df)
    network_csv=network_csv.join(network_df.T)
    network_csv.fillna(0, inplace=True)
    self.network_csv = network_csv

  def affinity_check(self, path, friends_of_friends: int):
    matrix_multifly = friends_of_friends - 1
    network_csv_v=(self.network_csv.values)
    for i in range(matrix_multifly):
        network_csv_v = network_csv_v.dot(network_csv_v)
    
    submitter_index = 0
    submitter_list, reviewer_list = [], (self.network_csv.index).tolist()

    self.submitter_df.fillna(0, inplace=True)
    for i in range(1,11):
        col_index = (i*3)+5
        if self.submitter_df.loc[submitter_index][col_index] != 0:
            submitter_list.append(self.submitter_df.loc[submitter_index][col_index])

    co_rel_df = pd.DataFrame(
        columns=[i for i in reviewer_list],
        index=[j for j in submitter_list])

    for i in range(len(submitter_list)):
        indet_temp=reviewer_list.index(submitter_list[i])
        co_rel_df.iloc[i,indet_temp]=1

    co_rel_df.fillna(0, inplace=True)

    network_csv_v = (co_rel_df.values).dot(network_csv_v)
    network_csv_v=pd.DataFrame(data=network_csv_v,
                  index=(co_rel_df.index).tolist(),
                  columns=(self.network_csv.index).tolist())
    
    network_csv_v = network_csv_v.loc[:, (network_csv_v != 0).any(axis=0)]
    network_csv_v = (network_csv_v.columns).tolist()

    co_inst_csv = pd.read_csv('./db/reviewer_information.csv', encoding='utf8')

    submitter_inst_list=[]
    for i in range(1,11):
        col_index = (i*3)+6
        if self.submitter_df.loc[submitter_index][col_index] != 0:
            submitter_inst_list.append(self.submitter_df.loc[submitter_index][col_index])

    reviewer_name_list,reviewer_inst_list=[],[]
    for i in range(len(co_inst_csv)):
        inst_list_temp=[]
        reviewer_name_list.append(co_inst_csv['reviewer_name'][i])
        reviewer_inst_list.append(co_inst_csv['reviewer_institution'][i])

    co_inst_csv = pd.DataFrame(
        columns=[i for i in reviewer_name_list],
        index=[j for j in submitter_list])

    for i in range(len(submitter_inst_list)):
      tmp=[j for j,x in enumerate(reviewer_inst_list) if x==submitter_inst_list[i]]
      for k in tmp:
        co_inst_csv.iloc[i,k] = 1

    co_inst_csv.fillna(0, inplace=True)
    co_inst_csv = co_inst_csv.loc[:, (co_inst_csv != 0).any(axis=0)]
    self.co_inst_csv = (co_inst_csv.columns).tolist()
    
    network_csv_v.extend(co_inst_csv)
    self.affinity = list(set(network_csv_v))
    
  def expertise_check(self,path):

    def create_dataframe(matrix, tokens):
      doc_names = [f'doc_{i}' for i, _ in enumerate(matrix)]
      df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
      return(df)

    reviewer_coauthor=(pd.read_csv(path, encoding='utf8'))
    tmp = pd.DataFrame(columns=list(reviewer_coauthor.columns), index=[x+937 for x in range(len(self.jane_submitter))])
    tmp['reviewer_name'], tmp["reviewer_orcid"] = self.jane_submitter, self.orcid

    reviewer_coauthor=reviewer_coauthor.append(tmp)

    reviewer_attr=pd.read_csv('./db/reviewer_attribute.csv', encoding='utf8')
    tmp = pd.DataFrame(columns=list(reviewer_attr.columns), index=[x+937 for x in range(len(self.jane_submitter_abst))])
    tmp['reviewer_paper_attribure'], tmp["reviewer_orcid"] = self.jane_submitter_abst, self.orcid
    reviewer_attr = reviewer_attr.append(tmp)

    submitter_index = 0
    name_sum, id_sum, find_id=[],[],[]

    for i in range(2,21,2):
        name_sum+=list(reviewer_coauthor.iloc[:,i])
        id_sum+=list(reviewer_coauthor.iloc[:,i+1])
        
    name_sum += reviewer_coauthor['reviewer_name'].tolist()
    id_sum += reviewer_coauthor['reviewer_orcid'].tolist()

    for i in range(len(self.affinity)):
        temp = [ x for x, y in enumerate(name_sum) if y == self.affinity[i] ]
        if temp != []: find_id.append(id_sum[temp[0]])

    for i in range(len(find_id)):
        reviewer_attr = reviewer_attr[reviewer_attr.reviewer_orcid != find_id[i]]

    reviewer_attr = reviewer_attr.reset_index(drop=True)
    reviewer_attr.loc[-1] = [
        self.submitter_df.loc[submitter_index]['submitter_orcid'],
        self.submitter_df.loc[submitter_index]['submitter_title_doi'],
        self.submitter_df.loc[submitter_index]['submitter_title'],
        self.submitter_df.loc[submitter_index]['submitter_attribute']]

    reviewer_attr.index += 1
    reviewer_attr.sort_index(inplace=True)
    reviewer_attr_list = reviewer_attr['reviewer_paper_attribure']

    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(reviewer_attr_list)

    tokens = count_vectorizer.get_feature_names()
    tokens_matrix=create_dataframe(vector_matrix.toarray(),tokens)

    cosine_matrix = cosine_similarity(vector_matrix)
    cosine_df = create_dataframe(cosine_matrix,tokens_matrix.index)
    cos = list(cosine_df['doc_0'])

    submit_jac, jac = (set(nltk.ngrams((nltk.word_tokenize(reviewer_attr_list[0])), n=1))), []

    for t in range(len(reviewer_attr_list)):   
        jac_temp = set(nltk.ngrams((nltk.word_tokenize(reviewer_attr_list[t])), n=1))
        jac.append(1-(nltk.jaccard_distance(submit_jac, jac_temp)))

    reviewer_attr['sim'] = [(x + y)/2 for x, y in zip(jac, cos)]

    reviewer_attr = reviewer_attr.sort_values(by=['sim'], axis=0, ascending=False)
    reviewer_attr = pd.concat([
        (reviewer_attr[~reviewer_attr.duplicated(['reviewer_orcid'],keep=False) == True]),
        (reviewer_attr[reviewer_attr.duplicated(['reviewer_orcid'],keep='last') == True])])

    reviewer_attr = reviewer_attr.sort_values(by=['sim'], axis=0, ascending=False)
    reviewer_attr = reviewer_attr[reviewer_attr['sim'] != 0]

    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(reviewer_attr['reviewer_paper_attribure'])

    tokens = count_vectorizer.get_feature_names()
    tokens_matrix=create_dataframe(vector_matrix.toarray(),tokens)

    Agglo_clust_model = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
    Agglo_clust_model.fit(tokens_matrix)

    count_list = [0 for i in range(len(Agglo_clust_model.labels_))]
    reviewer_attr['cluster'] = Agglo_clust_model.labels_

    reviewer_attr = (reviewer_attr[reviewer_attr['cluster'] == reviewer_attr.loc[0]['cluster']]).iloc[1:15]
    reviewer_attr = reviewer_attr.join(reviewer_coauthor.set_index('reviewer_orcid')[['reviewer_name']],on='reviewer_orcid')
    reviewer_attr = pd.concat([
        (reviewer_attr[~reviewer_attr.duplicated(['reviewer_orcid'],keep=False) == True]),
        (reviewer_attr[reviewer_attr.duplicated(['reviewer_orcid'],keep='last') == True])])
    self.reviewer_attr = reviewer_attr.sort_values(by=['sim'], axis=0, ascending=False)

    self.cluster_reviewer_name = list(reviewer_attr['reviewer_name'])
    self.expertise = list(reviewer_attr['reviewer_name'])

  def fair_allocation(self, scholar_google_headers, h_index_headers):
    scholar_google_params = [
      ('hl', 'en'),
      ('view_op', 'search_authors'),
      ('mauthors', 'Olyaee MH'),
      ('btnG', '')]
    
    url = 'https://scholar.google.com/citations'

    scholar_url_list=[]
    for i in range(len(self.cluster_reviewer_name)) :
      tmp = ['mauthors']
      tmp.append(str(self.cluster_reviewer_name[i].lower()))

      scholar_google_params[2] = tuple(tmp)
      res = requests.get(url, headers=scholar_google_headers, params=tuple(scholar_google_params))

      if res.status_code == requests.codes.ok:
        soup_abst = BeautifulSoup(res.text, 'html.parser')
        gsc_1usr = soup_abst.find_all('div', {'class':['gsc_1usr']})
        if gsc_1usr != [] :
          gsc_1usr = (gsc_1usr[0].find('a', {'class':['gs_ai_pho']}))["href"]
          scholar_url_list.append((gsc_1usr.split('user='))[1])
        else : 
          scholar_url_list.append(0)
      else :
        print(res.status_code)

    h_index_params = [
        ('hl', 'en'),
        ('user', 'pHROIAEAAAAJ'),]
    
    url = 'https://scholar.google.com/citations'
    h_index_list=[]
    for i in range(len(scholar_url_list)) :

      tmp = ['user']
      tmp.append(str(scholar_url_list[i]))

      h_index_params[1] = tuple(tmp)
      res = requests.get(url, headers=h_index_headers, params=tuple(h_index_params))

      if res.status_code == requests.codes.ok:
        soup_abst = BeautifulSoup(res.text, 'html.parser')
        gsc_1usr = soup_abst.find_all('td', {'class':['gsc_rsb_std']})
        if gsc_1usr == [] :
          h_index_list.append('0')
        else :
          h_index_list.append(gsc_1usr[2].text)
      else :
        h_index_list.append('0')

    h_index_list.insert(0,0)
    self.reviewer_attr['h_index'] = h_index_list[1:]

    tmp=[q for q, item in enumerate(self.reviewer_attr['h_index']) if item in '0']
    self.reviewer_attr=(self.reviewer_attr.iloc[:]).drop((self.reviewer_attr.iloc[:]).index[tmp])

    self.fair_allocation_recommendation = self.reviewer_attr.iloc[:,[0,2,4,6,7]]
