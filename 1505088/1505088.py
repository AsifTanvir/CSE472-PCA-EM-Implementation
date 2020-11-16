# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:33:28 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_spd_matrix


origin_data = pd.read_csv("data.txt", sep=" ", header=None)
standardized_data = StandardScaler().fit_transform(origin_data)
mean_vec = np.mean(standardized_data, axis=0)
cov_mat = (standardized_data - mean_vec).T.dot((standardized_data - mean_vec)) / (standardized_data.shape[0]-1)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

    
matrix_w = np.hstack((eig_pairs[0][1].reshape(-1,1), eig_pairs[1][1].reshape(-1,1)))


convertedInput = np.dot(standardized_data,matrix_w)
plt.title('PCA') 
plt.xlabel('Input Dimension 1') 
plt.ylabel('Input Dimension 2') 
plt.xticks() 
plt.yticks() 
plt.scatter(convertedInput[:,0],convertedInput[:,1],color='blue')
#plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, which='both')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

# =============================================================================
# EM starts here..
# Initializing Params
# =============================================================================
K = 3
W_k = np.ones(K)/K
means_mu = np.random.choice(convertedInput.flatten(), (K,convertedInput.shape[1]))
covariance_sigma = []
for j in range(K):
  covariance_sigma.append(make_spd_matrix(convertedInput.shape[1]))
covariance_sigma = np.array(covariance_sigma)

N = convertedInput.shape[0]
D = convertedInput.shape[1]
likelihood_array = np.zeros((N,K))

small_val = 1e-10
def calc_logLieklihood(convertedInput, means_mu, covariance_sigma, W_k):
    max_like = 0
    for n in range(N):
        likelihood = 0
        for k in range(K):
            denom = (1 / np.sqrt(((2*math.pi)**D) * np.linalg.det(covariance_sigma[k])))
            exponent = -0.5 * np.linalg.multi_dot([(convertedInput[n] - means_mu[k]).T, np.linalg.inv(covariance_sigma[k]),(convertedInput[n] - means_mu[k])])
            
            N_k = denom * (np.exp(exponent))
            likelihood += (W_k[k] * N_k)
            likelihood_array[n][k] = (W_k[k] * N_k)
            
        max_like += np.log(likelihood + small_val)
    return max_like




init_likelihood = calc_logLieklihood(convertedInput, means_mu, covariance_sigma,W_k)
print(init_likelihood, covariance_sigma)
current_likelihood = 0.0  
for itr in range(60):
    
    P_k = np.zeros((N,K))
    for n in range(N):
        pk = []
        for k in range(K):
            denom = (1 / np.sqrt(((2*math.pi)**D) * np.linalg.det(covariance_sigma[k])))
            exponent = -0.5 * np.linalg.multi_dot([(convertedInput[n] - means_mu[k]).T, np.linalg.inv(covariance_sigma[k]),(convertedInput[n] - means_mu[k])])
            N_k = denom * (np.exp(exponent))
            #pik = N_k*W_k[k]
            pk.append(N_k*W_k[k])
        pk = pk / np.sum(pk)
        P_k[n] = pk
        
    #M step
    for kk in range(K):
        sum_Pik = 0
        sum_means = np.zeros_like(means_mu[kk,:]).reshape(1,-1)
        sum_covariance = np.zeros_like(covariance_sigma[kk,:])
        for nn in range(N):
            input_x = convertedInput[nn,:].reshape(1,-1)
            sum_Pik += P_k[nn][kk]
            sum_means += (P_k[nn][kk] * input_x)
            #sum_covariance += (np.dot( (convertedInput[n] - means_mu[k]), (convertedInput[n] - means_mu[k]))) * P_k[n][k]
            #sum_covariance += np.linalg.multi_dot([P_k[n][k], (convertedInput[n] - means_mu[k]) , (convertedInput[n] - means_mu[k]).T])
        means_mu[kk] = sum_means / sum_Pik
        
        for nm in range(N):
            input_x = convertedInput[nm,:].reshape(1,-1)
            sum_covariance += (np.dot( (input_x - means_mu[kk]).T, (input_x - means_mu[kk]))) * P_k[nm][kk]
            
        covariance_sigma[kk] = sum_covariance / sum_Pik
        W_k[kk] = sum_Pik / N
        #print(":\n",covariance_sigma)
    
    current_likelihood = calc_logLieklihood(convertedInput, means_mu, covariance_sigma, W_k)
    diff = abs(init_likelihood - current_likelihood)
    print("vals \n", current_likelihood, diff)
    if diff < small_val:
        print("\n Mean: ", means_mu, "\n Covariance: ", covariance_sigma, "\n Mixing Coefficient: ", W_k)
        break
    else:
        init_likelihood = current_likelihood
        #print(n)



#x = convertedInput - means_mu[0]
#print(W_k , means_mu, covariance_sigma.shape, convertedInput.shape, max_like)