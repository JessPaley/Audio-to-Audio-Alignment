from librosa.core import pitch
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist
import librosa, librosa.display
import math
import matplotlib.pyplot as plt

### Block Audio ###
def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

### Read Audio ###
def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)
    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32
        audio = x / float(2**(nbits - 1))
    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.
    return(samplerate, audio)

### RMS ###
def extract_rms(xb):
    x_square = 0
    length = xb.size
    window = np.hanning(length)
    xb = xb * window
    for i in range(length):
        x_square += xb[i]**2
    rms = math.sqrt(x_square/length)
    # truncate at -100 db
    if(rms < 1e-5):
        rms = 1e-5
    rms_db = 20 * math.log10(rms)
    return rms_db

### ZCR ###
def extract_zerocrossingrate(xb):
    length = xb.size
    window = np.hanning(length)
    xb = xb * window
    cross_boolean = np.diff(xb > 0)
    cross_ind = np.where(cross_boolean == True)[0]
    mean = cross_ind.size / (length-1)
    ZC = mean
    return ZC

### Get Audio Block ###
def getBlock(fileName, blockSize=1024, hopSize=512):
    fs, audio = ToolReadAudio(fileName)
    xb, t = block_audio(audio, blockSize, hopSize, fs)
    numBlock = math.ceil(audio.size / 512)
    return xb, numBlock, audio, fs

### Generate the Feature Matrix ###
def Feature_Matrix(fileName):
    xb, numBlock, audio, fs = getBlock(fileName)
    feature_matrix = np.zeros([numBlock, 2])
    max_rms = 0
    max_zcr = 0

    for i in range(0, numBlock):
        rms = extract_rms(xb[i])
        if abs(rms) > max_rms:
            max_rms = abs(rms)

        zcr = extract_zerocrossingrate(xb[i])
        if abs(zcr) > max_zcr:
            max_zcr = abs(zcr)
        feature_vec = np.array([rms, zcr])
        feature_matrix[i] = feature_vec

    ### Normalize ###
    for i in range(0, numBlock):
        feature_matrix[i][0] = feature_matrix[i][0] / max_rms
        feature_matrix[i][1] = feature_matrix[i][1] / max_zcr
    print('feature:\n', feature_matrix)
    return feature_matrix

### Calculate Euclidean Distance ###
def Distance_matrix(sig1, sig2):
    if sig1.shape[1] > 1:
        dis_matrix = cdist(sig1, sig2,'euclidean')
    else:
        dis_matrix = np.zeros((sig1.shape[0], sig2.shape[0]))
        column = dis_matrix.shape[0]
        row = dis_matrix.shape[1]
        for i in range(column):
            for j in range(row):
                dis_matrix[i,j] = np.linalg.norm(sig1[i]-sig2[j])
    return dis_matrix
     
### Cost Matrix ###
def Cost_matrix(sig1, sig2):
    d_matrix = Distance_matrix(sig1, sig2)
    c_matrix = np.zeros((sig1.shape[0], sig2.shape[0]))
    column = c_matrix.shape[0]
    row = c_matrix.shape[1]

    # c_matrix[0, 1:] = Calculate First Row
    n = 0
    for i in range(0, sig2.shape[0]):
        n = d_matrix[0,i] + n
        c_matrix[0, i] = n
    
    # c_matrix[1:, 0] = Calculate First Column
    n = 0
    for i in range(0, sig1.shape[0]):
        n = d_matrix[i, 0] + n
        c_matrix[i, 0] = n
    
    for i in range(1, column):
        for j in range(1, row):
            minvalue = min(c_matrix[i-1,j-1], c_matrix[i,j-1], c_matrix[i-1,j])
            c_matrix[i,j] = d_matrix[i, j] + minvalue
    return d_matrix, c_matrix

### DTW Path ###
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
    path_np = np.array(path)
    # print(matrix)
    # print(path)
    return path_np


# test_matrix = np.array([[1,2],[3,4],[5,6]])
# test_matrix2 = np.array([[3,7],[3,8],[10,5],[3,7]])
test_matrix = np.array([1,3,9,2,1])
test_matrix2 = np.array([2,0,0,8,7,2])

file = '7100 Research (Local File)/Part_1.wav'
file2 = '7100 Research (Local File)/Full_1.wav'
FeatureMatrix = Feature_Matrix(file)
FeatureMatrix2 = Feature_Matrix(file2)
print("feature shape: \n", FeatureMatrix.shape)

# FeatureMatrix, FeatureMatrix2
# test_matrix, test_matrix2
c_matrix = Cost_matrix(FeatureMatrix, FeatureMatrix2)[1]
d_matrix = Cost_matrix(FeatureMatrix, FeatureMatrix2)[0]
dtw_calculation = DTW(c_matrix) # Return numpy array, [y-axis, x-axis]
# print('distance matrix\n', d_matrix)
# print('cost matrix\n', c_matrix)
# print('path\n', dtw_calculation)


# ### Plotting ###
# plt.figure(figsize=(9, 3))
# plt.subplot(1, 2, 1)
# plt.imshow(c_matrix, cmap='gray_r', origin='lower', aspect='equal')
# plt.plot(dtw_calculation[:, 1], dtw_calculation[:, 0], color='r')
# plt.clim([0, np.max(c_matrix)])
# plt.colorbar()
# plt.title('Cost Matrix With Path')
# plt.xlabel('Sequence Y')
# plt.ylabel('Sequence X')

# plt.subplot(1, 2, 2)
# plt.imshow(d_matrix, cmap='gray_r', origin='lower', aspect='equal')
# plt.plot(dtw_calculation[:, 1], dtw_calculation[:, 0], marker='o', color='r')
# plt.clim([0, np.max(d_matrix)])
# plt.colorbar()
# plt.title('Distance Matrix With Path')
# plt.xlabel('Sequence Y')
# plt.ylabel('Sequence X')

# plt.tight_layout()
# # plt.show()


### To do: ###
# Make similarity graph of some audio from the dataset
#   - Pitch Chroma graph
#       - block the audio with 12 dimension vector each time
#       - normalize the data by each block at a time
#       - calculate the distance matrix using euclidian distance
#
#   - RMS, Zero Crossing graph (switch to librosa)
#       - normalize by using z-score, each feature row at a time
# 
# Optimization
#   - Search about pdist2
