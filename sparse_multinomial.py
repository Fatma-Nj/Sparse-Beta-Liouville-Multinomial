# -*- coding: utf-8 -*-
"""



@author: Fatma Najar

Sparse mutlinomial - prediction distribution
"""
###########################################################
                  #Imports libraries
###########################################################
import numpy as np
import pandas as pd
import math
from scipy.special import gammaln, logsumexp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from coclust.evaluation.external import accuracy #accuracy metric for clustering
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
###########################################################


###########################################################
                   # Functions
###########################################################
            
def scaling_factor(alpha, k_0, L, N):
    
    scale = 0
    lamda = .9
    sum_m = 1e-10
    for k in range(k_0,L-1):
        try:
            fact_k = (np.math.gamma(k+1)/np.math.gamma(np.abs(k-k_0)+1)) \
              * (np.math.gamma(alpha * k)) / (np.math.gamma(alpha *k + N))
        except OverflowError:
            fact_k = 1e20
        prior = lamda ** k  # exponential prior
        #prior = k ** (-lamda) #polynomial prior
        
        m_k = fact_k * prior 
        
        
        sum_m = sum_m + m_k
        scale =  scale +  (k_0 * alpha + N)/(k * alpha + N) *  m_k 
        
        m_k = 0
    scale /= sum_m
    
    return scale

def predictive_Dirichlet(words, alpha, L):
    _, D = words.shape
    
    #predictive_D = 0
    N = np.sum(words)
 
    
    N_d = np.sum(words, axis = 0)
    "count the number of observed words in each class j"
    k_0 = np.count_nonzero(N_d)
    
    predictive_D = np.zeros((D,))
    
    scale = scaling_factor(alpha, k_0, L, N)
    
   
    for d in range(D):  
        
        "Dirichlet smoothing "
        # predictive_D [d]=  (alpha + N_d[d])/(D * alpha + N)   #dirichlet smoothing
        " Dirichlet smoothing+ Vocabulary knowledge"
        if (N_d[d] == 0).any():  #if the words are unseen
            predictive_D [d]= 1/(L- k_0) * (1 - scale)   
           
        else:                                         # if the words are observed            
            predictive_D [d]= (alpha + N_d[d])/(k_0 * alpha + N)  * scale
                
      
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
# First, we construct the vocabulary,

vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=800) #V=size of vocabulary
x_fits = vectorizer.fit_transform(data_samples)

x_counts = vectorizer.transform(data_samples).toarray()

"split the data in 70% training and 30% prediction/testing"
data_train, data_test, label_train, label_test = train_test_split(x_counts, labell, test_size=0.1)
"""
###############################################
             Words prediction
###############################################
"""

"parameters"


a = 1.
alpha = .2
beta = .2

N, D = data_train.shape
L = D - 1 

T, D = data_test.shape

nb_words = T * D

K = 4  # number of class/topic
new_param = np.zeros((K,D))


for j in range(K):

    param_j = predictive_Dirichlet(data_train[label_train==j,:], alpha, L)
        
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
    posterior_cluster[:,j] = emotion_prediction(data_test, new_param[j,:], prior)
    
    


"Evaluation metrics"
label_predicted = np.zeros((T,))
for i in range(T):
    (m,label_predicted[i]) = max((v,index) for index,v in enumerate(posterior_cluster[i,:]))



    
accuracy_d = accuracy(label_test, label_predicted)


F_1_mic = f1_score(label_test, label_predicted, average='micro')


# pearson correlation
pr_correlation = pearsonr(label_test, label_predicted)[0]


