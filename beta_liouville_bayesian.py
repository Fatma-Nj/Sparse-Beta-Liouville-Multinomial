# -*- coding: utf-8 -*-
"""

@author: Fatma Najar

Beta-liouville mutlinomial - Bayes classifier -
"""
###########################################################
                  #Imports libraries
###########################################################
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import math
from coclust.evaluation.external import accuracy #accuracy metric for clustering
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
###########################################################


###########################################################
                   # Functions
###########################################################
            
def scaling_factor(a, k_0, L, N):
    
    scale = 0
    lamda = 1.
    sum_m = 1e-10
    for k in range(k_0,L):
        try:
            fact_k = (np.math.gamma(k+1)/np.math.gamma(np.abs(k-k_0)+1)) \
              * (np.math.gamma(a * k)) / (np.math.gamma(a *k + N))
        except OverflowError:
            fact_k = 1e20
        prior = lamda ** k  # exponential prior
        #prior = k ** (-lamda) #polynomial prior
        
        m_k = fact_k * prior 
        
        
        sum_m = sum_m + m_k
        scale =  scale +  ((k_0 * a + N)/(k * a + N)) *  m_k 
        
        m_k = 0
    scale /= sum_m
    
    return scale

def predictive_Beta_Liouville(words, a, alpha, beta, L):
    "dimension of words"
   
    _, D = words.shape
    
    #predictive_D = 0
    N = np.sum(words[:,0:D-2])
    M = N + np.sum(words[:,D-1])
    
    N_d = np.sum(words, axis = 0)
    "count the number of observed words in each class j"
    k_0 = np.count_nonzero(N_d)
    
    predictive_D = np.zeros((D,))
    
    scale = scaling_factor(a, k_0, L, N)   
    for d in range(D): 
        if (N_d[d] == 0).any(): #if the words are unseen    
            predictive_D [d] = 1/(L- k_0) * (1 - scale)         
        else:                                         # if the words are observed            
            predictive_D [d] = (a + N_d[d])/(k_0 * a + N)  * ( alpha + N) /(alpha + beta + M)* scale
            
    return predictive_D


      
def emotion_prediction(words, theta, pis):

    N, D = words.shape
    posterior = np.ones((N,))
    

    for i in range(N):
        for d in range(D):
            posterior [i] = posterior [i] + (words[i,d] * math.log10(theta[d]+1e-10))
              
        posterior[i] = math.log10(pis) +  posterior [i]
        
        
    return posterior
############################################################               

"""
###############################################
              Data preprocessing
###############################################"""
data = pd.read_csv("Data\emotion_data.txt",sep="\t")
data_samples = data['tweet'] 
label = data['emotion']

#emotion label
labell = np.zeros((len(label),))  

for i in range(len(label)):
    if (label[i]=="fear"):
        labell[i]=0
    if (label[i]=="joy"):
        labell[i]=1
    if (label[i]=="sadness"):
        labell[i]=2
    if (label[i]=="anger"):
        labell[i]=3

"split the data in 70% training and 30% testing"
data_train, data_test, label_train, label_test = train_test_split(data_samples, labell, test_size=0.3)


# First, we construct the vocabulary,

vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=800) #V=size of vocabulary
x_fits = vectorizer.fit_transform(data_train)

x_train = vectorizer.transform(data_train).toarray()
x_test = vectorizer.transform(data_test).toarray()



"Multinomial Naive Bayes"
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

clf.fit(x_train, label_train)

predi_M = clf.predict(x_test)
accuracy_M = accuracy_score(label_test, predi_M)
pr = pearsonr(label_test, predi_M)[0]

"""###############################################
             Words prediction
###############################################
"""
"parameters"
a = 1.
alpha = .2
beta = .2

N, D = x_train.shape
L = D - 1 

T, D = x_test.shape

nb_words = T * D

K = 4  # number of class/topic
new_param = np.zeros((K,D))

for j in range(K):

    param_j = predictive_Beta_Liouville(x_train[label_train==j,:], a, alpha, beta, L)
    #param_j = predictive_Dirichlet(x_train[label_train==j,:], alpha, L)
        
    new_param[j,:] = param_j
    
        
   
"""
##################################################
             Emotion prediction
#################################################
"""
#Normalized new param

for j in range(K):
    new_param[j,:] = abs(new_param[j,:]/sum(new_param[j,:]))


posterior_cluster = np.zeros((T,K))


for j in range(K):
    "determining the prior"
    list1 = label_test.tolist()
    prior =  list1.count(j)
    posterior_cluster[:,j] = emotion_prediction(x_test, new_param[j,:], prior)
    
    


"Evaluation metrics"
label_predicted = np.zeros((T,))
for i in range(T):
    (m,label_predicted[i]) = max((v,index) for index,v in enumerate(posterior_cluster[i,:]))



    
accuracy_test = accuracy(label_test, label_predicted)


F_1_mic = f1_score(label_test, label_predicted, average='micro')


# pearson correlation
pr_correlation = pearsonr(label_test, label_predicted)[0]


