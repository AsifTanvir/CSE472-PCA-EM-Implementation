# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:33:28 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


origin_data = pd.read_csv("data.txt", sep=" ", header=None)
standardized_data = StandardScaler().fit_transform(origin_data)
mean_vec = np.mean(standardized_data, axis=0)
cov_mat = (standardized_data - mean_vec).T.dot((standardized_data - mean_vec)) / (standardized_data.shape[0]-1)
print('Covariance matrix \n%s',cov_mat , cov_mat.shape)
#cov_mat = np.cov(standardized_data.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n',eig_vecs)
print('\nEigenvalues \n',eig_vals)
#print(origin_data.head())

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[1])
    
matrix_w = np.hstack((eig_pairs[0][1].reshape(-1,1), eig_pairs[1][1].reshape(-1,1)))
#Alternative matrix
#matrix_w = np.array([[0.447,0.895],[0.895,-0.447]])
print('Matrix W:\n', matrix_w)