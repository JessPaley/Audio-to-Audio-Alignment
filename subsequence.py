import numpy as np
from numpy.lib import row_stack
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist

# To do:
# DTW determines multiple possible path candidates representing the potentially overlapping between pair of recording

# Needs to consider all possible subsequences of Y to find the optimal one
# - Determining the cost of an optimal warping path

### Calculate Euclidean Distance ###
def Distance_matrix(sig1, sig2):
    # # Old way
    # dis_matrix = np.zeros((sig1.size, sig2.size))
    # column = dis_matrix.shape[0]
    # row = dis_matrix.shape[1]
    # for i in range(column):
    #     for j in range(row):
    #         dis_matrix[i,j] = np.abs(sig1[i]-sig2[j])
    
    # Using cdist
    X,Y = np.atleast_2d(sig1, sig2)
    dis_matrix = cdist(X.T, Y.T,'euclidean')
    return dis_matrix

### Cost Matrix ###
def Cost_matrix(sig1, sig2):
    d_matrix = Distance_matrix(sig1, sig2)
    c_matrix = np.zeros((sig1.shape[0], sig2.shape[0]))
    column = c_matrix.shape[0]
    row = c_matrix.shape[1]

    # c_matrix[0, 1:] = Calculate First Row
    c_matrix[0,:] = d_matrix[0,:]
    
    # c_matrix[1:, 0] = Calculate First Column
    n = 0
    for i in range(0, sig1.shape[0]):
        n = d_matrix[i, 0] + n
        c_matrix[i, 0] = n
    
    # Calculate Rest Matrix
    for i in range(1, column):
        for j in range(1, row):
            minvalue = min(c_matrix[i-1,j-1], c_matrix[i,j-1], c_matrix[i-1,j])
            c_matrix[i,j] = d_matrix[i, j] + minvalue
    return c_matrix

### Calculate DTW Path ###
def DTW(matrix):
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    # path = [[i + 1,j + 1]]
    path = [[i, j]]
    while i > 0 or j > 0:
        a_list = [matrix[i-1,j-1], matrix[i,j-1], matrix[i-1,j]]
        minvalue = min(a_list)
        min_index = a_list.index(minvalue)
        if min_index == 0:
            i = i - 1
            j = j - 1
        elif min_index == 1:
            i = i
            j = j - 1
        elif min_index == 2:
            i = i - 1
            j = j
        # path.append([i+1,j+1])
        path.append([i,j])
    path_np = np.flip(np.array(path))
    # print(matrix)
    # print(path)
    return path_np


X = np.array([3, 0, 6])
Y = np.array([2, 4, 0, 4, 0, 0, 5, 2])

d_matrix = Distance_matrix(X,Y)
c_matrix = Cost_matrix(X,Y) 
path = DTW(c_matrix)
# print(d_matrix)
print(c_matrix)
print(c_matrix[-1,:])
print(path)