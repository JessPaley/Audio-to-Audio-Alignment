import numpy as np

test_signal = np.array([1,4,5,10,9,3,2,6,8,4])
test_signal2 = np.array([1,7,3,4,1,10,5,4,7,4])

def DTW_matrix(sig1, sig2):
    matrix = np.zeros((sig1.size, sig2.size))
    column = matrix.shape[0]
    row = matrix.shape[1]

    # sig1[i] index through column, sig2[j] index through row in each column
    for i in range(column): 
        for j in range(row):
            value = np.abs(sig1[i]-sig2[j])
            if i == 0 and j == 0:
                # print("i=", i, "& j=", j, "t1")
                minvalue = 0
                matrix[i,j] = value + minvalue
            elif i == 0:
                # print("i=", i, "& j=", j, "t2")
                minvalue = matrix[i, j-1]
                matrix[i,j] = value + minvalue
            elif j == 0:
                # print("i=", i, "& j=", j, 't3')
                minvalue = matrix[i-1, j]
                matrix[i,j] = value + minvalue
            else:
                # print("i=", i, "& j=", j, 't4')
                minvalue = min(matrix[i-1,j-1], matrix[i,j-1], matrix[i-1,j])
                matrix[i,j] = value + minvalue
    print(matrix)
    return matrix

DTW_matrix(test_signal, test_signal2)