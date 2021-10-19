import numpy as np

# test_signal = np.array([0,4,4,0,-4,-4,0]) # y-axis
# test_signal2 = np.array([1,3,4,3,1,-1,-2,-1,0]) # x-axis

test_signal = np.array([1,6,8,2]) # y-axis
test_signal2 = np.array([4,2,5,7]) # x-axis

### Distance Matrix ###
def Distance_matrix(sig1, sig2):
    dis_matrix = np.zeros((sig1.size, sig2.size))
    column = dis_matrix.shape[0]
    row = dis_matrix.shape[1]
    for i in range(column):
        for j in range(row):
            dis_matrix[i,j] = np.abs(sig1[i]-sig2[j])
    return dis_matrix

### Cost Matrix ###
def Cost_matrix(sig1, sig2):
    d_matrix = Distance_matrix(sig1, sig2)
    c_matrix = np.zeros((sig1.size, sig2.size))
    column = c_matrix.shape[0]
    row = c_matrix.shape[1]

    # c_matrix[0, 1:] = Calculate First Row
    n = 0
    for i in range(0, sig2.size):
        n = d_matrix[0,i] + n
        c_matrix[0, i] = n
    
    # c_matrix[1:, 0] = Calculate First Column
    n = 0
    for i in range(0, sig1.size):
        n = d_matrix[i, 0] + n
        c_matrix[i, 0] = n
    
    for i in range(1, column):
        for j in range(1, row):
            minvalue = min(c_matrix[i-1,j-1], c_matrix[i,j-1], c_matrix[i-1,j])
            c_matrix[i,j] = d_matrix[i, j] + minvalue
    # print(d_matrix)
    # print(c_matrix)
    return d_matrix, c_matrix
Cost_matrix(test_signal, test_signal2)

### DTW Path ###
def DTW(matrix):
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    path = [(i + 1,j + 1)]
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
        path.append((i+1,j+1))
    print(matrix)
    print(path)
    return path
c_matrix = Cost_matrix(test_signal, test_signal2)[1]
DTW(c_matrix) # Return list of tuples, (y-axis, x-axis)
