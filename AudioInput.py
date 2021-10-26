import numpy as np
from scipy.io.wavfile import read as wavread
import math

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

### Generate the Feature Matrix ###
def Feature_Matrix(fileName, blockSize=1024, hopSize=512):
    fs, audio = ToolReadAudio(fileName)
    xb, t = block_audio(audio, blockSize, hopSize, fs)
    numBlock = math.ceil(audio.size / 512)

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
    return feature_matrix

### Calculate Euclidean Distance ###
def Distance_matrix(sig1, sig2):
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
    # print('distance matrix', d_matrix)
    # print('cost matrix', c_matrix)
    return d_matrix, c_matrix

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


# test_matrix = np.array([[1,2],[3,4],[5,6]])
# test_matrix2 = np.array([[3,7],[3,8],[10,5],[3,7]])

file = '7100 Research/SO_RE_80_piano_melody_lime_Gmaj(Mono_1).wav'
file2 = '7100 Research/SO_RE_80_piano_melody_lime_Gmaj(Mono_2).wav'
FeatureMatrix = Feature_Matrix(file)
FeatureMatrix2 = Feature_Matrix(file2)

c_matrix = Cost_matrix(FeatureMatrix, FeatureMatrix2)[1]
d_matrix = Cost_matrix(FeatureMatrix, FeatureMatrix2)[0]
DTW(c_matrix) # Return list of tuples, (y-axis, x-axis)