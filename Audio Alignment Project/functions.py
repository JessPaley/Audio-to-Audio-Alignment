import librosa, librosa.display
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
    
    # x, sr = librosa.load(cAudioFilePath)
    # samplerate = sr
    # audio = x
    return(samplerate, audio)

### Pitch Chroma ###
def chroma(audio, sr=44100, blockSize=2048, hopSize=512):
    S = np.abs(librosa.stft(y=audio, n_fft=blockSize, hop_length=hopSize))
    chromagram = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=blockSize, hop_length=hopSize, n_chroma=12)
    t_chromagram = chromagram.T
    norm_chromagram = t_chromagram

    # # normalize the chromagram, sum up to 1
    # norm_chromagram = np.zeros([t_chromagram.shape[0],t_chromagram.shape[1]])
    # for i in range(t_chromagram.shape[0]):
    #     norm_chromagram[i] = t_chromagram[i] / np.sum(t_chromagram[i])
    return norm_chromagram

### Calculate Euclidean Distance ###
def Distance_matrix(sig1, sig2):
    # # Old way
    # dis_matrix = np.zeros((sig1.size, sig2.size))
    # column = dis_matrix.shape[0]
    # row = dis_matrix.shape[1]
    # for i in range(column):
    #     for j in range(row):
    #         dis_matrix[i,j] = np.abs(sig1[i]-sig2[j])
    
    # # Using cdist
    # X,Y = np.atleast_2d(sig1, sig2)
    # dis_matrix = cdist(X.T, Y.T,'euclidean')

    dis_matrix = cdist(sig1, sig2,'euclidean')
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
def modified_DTW(matrix, runAll=True):
    N = matrix.shape[0] # Row
    M = matrix.shape[1] # Column
    if runAll==True:
        for a in range(0, M):
            n = N - 1
            m = a
            path = [[n, m]]
            while n > 0:
                if m == 0:
                    n = n-1
                    m = 0
                else:
                    a_list = [matrix[n-1,m-1], matrix[n,m-1], matrix[n-1,m]]
                    minvalue = min(a_list)
                    min_index = a_list.index(minvalue)
                    if min_index == 0:
                        n = n - 1
                        m = m - 1
                    elif min_index == 1:
                        n = n
                        m = m - 1
                    elif min_index == 2:
                        n = n - 1
                        m = m
                path.append([n,m])
                path.reverse()
            # path_np = np.flip(np.array(path))
            path_np = np.array(path)
            X = np.zeros(path_np.shape[0])
            Y = np.zeros(path_np.shape[0])
            for i in range(0, path_np.shape[0]):
                X[i] = path_np[i][0]
                Y[i] = path_np[i][1]      
            result = linregress(X,Y)

            if abs(result.slope - 1) < 0.5:
                print(result.slope)
                print(np.amax(path_np, axis=0)[1])
                max_vec = np.amax(path_np, axis=0)
                start_ind = m
                end_ind = max_vec[1]

    elif runAll==False:
        n = N - 1
        m = matrix[-1, :].argmin() # Locate the lowest cost index from distance matrix
        path = [[n, m]]
        while n > 0:
            if m == 0:
                # n = n-1
                # m = 0
                continue
            else:
                a_list = [matrix[n-1,m-1], matrix[n,m-1], matrix[n-1,m]]
                minvalue = min(a_list)
                min_index = a_list.index(minvalue)
                if min_index == 0:
                    n = n - 1
                    m = m - 1
                elif min_index == 1:
                    n = n
                    m = m - 1
                elif min_index == 2:
                    n = n - 1
                    m = m
            path.append([n,m])
            # path.reverse()
        path.reverse()
        # path_np = np.flip(np.array(path))
        path_np = np.array(path)

        X = np.zeros(path_np.shape[0])
        Y = np.zeros(path_np.shape[0])
        for i in range(0, path_np.shape[0]):
            X[i] = path_np[i][0]
            Y[i] = path_np[i][1]      
        result = linregress(X,Y)
        # print('Path Slope: ', result.slope)

        max_vec = np.amax(path_np, axis=0)
        start_ind = m
        end_ind = max_vec[1]

    return path_np, start_ind, end_ind

### Convert path index to samples ###
def pathInd2Time(path, hop_len=512, fs=44100):
    time4ref = []
    time4other = []
    refPath = []
    otherPath = []
    for i in range(0, len(path)):
        sample_ref = librosa.frames_to_samples(path[i][1], hop_length=hop_len)
        sample_other = librosa.frames_to_samples(path[i][0], hop_length=hop_len)
        time4ref.append(sample_ref/fs)
        time4other.append(sample_other/fs)
        refPath.append(path[i][1])
        otherPath.append(path[i][0])
    return time4ref, time4other, refPath, otherPath

### Moving Average Slope ###
def averageSlope(otherPath, refPath, windowSize):
    filtered_x = []
    filtered_y = []
    all_slope = []
    for i in range(0, len(refPath)-windowSize+1):
        win_a = refPath[i:i+windowSize]
        win_b = otherPath[i:i+windowSize]
        slope, intercept, r, p, se = linregress(win_a, win_b)
        # slope, intercept = np.polyfit(win_a,win_b,1)
        # c = win_b[0]

        x = win_a[int(windowSize/2)]
        y = slope * x + intercept
        # filtered_x.append(x)
        filtered_x.append(librosa.frames_to_samples(x, hop_length=windowSize)/44100)
        # filtered_y.append(slope * x + intercept)
        filtered_y.append(librosa.frames_to_samples(y, hop_length=windowSize)/44100)
        all_slope.append(slope)
    return filtered_x, filtered_y

### Moving Average Filter ###
def MAfilter(signal, windowSize):
    filtered_signal = np.convolve(signal, np.ones(windowSize), 'valid') / windowSize
    
    filtered_signal_test = []
    for i in range(0, len(signal)-windowSize+1):
        ind = np.sum(signal[i:i+windowSize])/windowSize
        filtered_signal_test.append(round(ind,2))
    return filtered_signal, filtered_signal_test

### CSV Writer ###
def writeCSV(audioPath_ref, audioPath_test, filtered_ref, filtered_test):
    ref_Name = audioPath_ref.split("/")[-1]
    test_Name = audioPath_test.split("/")[-1]
    
    import csv

    fields = [ref_Name, test_Name]
    rows_ref = np.array(filtered_ref).reshape((len(filtered_ref),1))
    rows_test = np.array(filtered_test).reshape((len(filtered_test),1))
    rows = np.concatenate((rows_ref,rows_test), axis=1)
    
    filename = "timestamps.csv"
    with open(filename,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

