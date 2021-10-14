import sys
import numpy as np
import heapq

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

# based on Dijkstra's Algorithm
# starts at the first element in the matrix and ends at the last element
# add visited nodes to new list and return list
# # how to make it more efficient?
def shortestPath(matrix):
    edgeSet = makeEdges(matrix)
    pq = []
    output = []
    heapq.heappush(pq, (0, (-1, -1), (0, 0)))  # start at first matrix element
    while len(pq) > 0:
        current = heapq.heappop(pq)
        # to do:
        # add connecting nodes to priority queue
        # update distances in edgeSet
        # add visited nodes and distances to output list

    return output

# helper method for shortestPath()
# creates a tuple with distance initialized to the maximum integer, the
# starting node (row, column), and the ending node (row, column)
def makeEdges(matrix):
    col = matrix.shape[0]
    row = matrix.shape[1]
    i = 0
    j = 0
    myEdges = []
    while i < col:
        while j < row:
            if i < col - 1 and j < row - 1:
                myEdges.append((sys.maxsize, (i, j), (i + 1, j + 1)))
            if i < col - 1:
                myEdges.append((sys.maxsize, (i, j), (i + 1, j)))
            if j < row - 1:
                myEdges.append((sys.maxsize, (i, j), (i, j + 1)))
            j += 1
        j = 0
        i += 1
    return myEdges


test1 = np.array([1, 4, 2])
test2 = np.array([1, 7, 3])

shortestPath(DTW_matrix(test1, test2))