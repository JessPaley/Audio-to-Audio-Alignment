import librosa, librosa.display
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt
import math

# To do:
# DTW determines multiple possible path candidates representing the potentially overlapping between pair of recording
# Start Running DTW column by column? From the Cost Matrix?

# Needs to consider all possible subsequences of Y to find the optimal one
# - Determining the cost of an optimal warping path

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
    # norm_chromagram = t_chromagram

    # normalize the chromagram, sum up to 1
    norm_chromagram = np.zeros([t_chromagram.shape[0],t_chromagram.shape[1]])
    for i in range(t_chromagram.shape[0]):
        norm_chromagram[i] = t_chromagram[i] / np.sum(t_chromagram[i])
    return norm_chromagram

### Feature Vector ###
def featVector(path):
    fs, audio = ToolReadAudio(path)
    chromagram = chroma(audio, sr=fs)
    return chromagram

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
            print(path_np)

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
            path.reverse()
        # path_np = np.flip(np.array(path))
        path_np = np.array(path)
        print("lowest cost index starting at:", m)

    return path_np

# # Number Test:
# X = np.array([3, 0, 6])
# Y = np.array([2, 4, 0, 4, 0, 0, 5, 2])
# # X = np.array([1, 3, 9, 2, 1])
# # Y = np.array([2, 0, 0, 8, 7, 2])
# d_matrix = Distance_matrix(X,Y)
# c_matrix = Cost_matrix(X,Y) 
# path = modified_DTW(c_matrix, runAll=False)


# Audio Test:
file = 'Assignments/7100 Research (Local File)/Subsequence(pid9048-01_bip).wav'
file2 = "Assignments/7100 Research (Local File)/Full(pid1263-01_bip).wav" #ref
# chromaVec = featVector(file)
# chromaVec2 = featVector(file2) #ref

# d_matrix = Distance_matrix(chromaVec,chromaVec2)
# c_matrix = Cost_matrix(chromaVec,chromaVec2) 
# path = modified_DTW(c_matrix, runAll=False)
# # print(path)

# plt.subplot(2,1,1)
# plt.imshow(d_matrix, origin='lower', aspect='auto')
# # plt.plot(path[:, 1], path[:, 0], color='r')
# plt.clim([0, np.max(d_matrix)])
# plt.colorbar()
# plt.title('Subsequence - Lowest Cost Path (Distance Matrix)')
# plt.xlabel('Subsequence')
# plt.ylabel('Full')

# plt.subplot(2,1,2)
# plt.imshow(d_matrix, origin='lower', aspect='auto')
# plt.plot(path[:, 1], path[:, 0], color='r')
# plt.clim([0, np.max(d_matrix)])
# plt.colorbar()
# plt.title('Subsequence - Lowest Cost Path (Distance Matrix)')
# plt.xlabel('Subsequence')
# plt.ylabel('Full')

# plt.tight_layout()
# plt.show()



# To Do:
# Evaluating the algorithm
# Compare self similarity, and check the onset time on dataset

# Pick a frame from the reference track (pid9072-01): measure #(start) -> measure #(end)
# Not Unique Frame Index: 25 - 49
# Unique Frame Index: 244 - 288

# Save the time(sec) value for the frame
# Use the subsequence algorithm to find the frame in other tracks (path index)
# Convert the subsequence path index to time(sec) and 

def readCSV(filepath, ref_track, start_ind, end_ind):
    import csv
    import os

    fields = []
    rows = []
    with open(filepath, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)    
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    
    ind_ref = fields.index(ref_track)
    start_t = float(rows[start_ind][ind_ref])
    end_t = float(rows[end_ind+1][ind_ref])
    print(start_t)
    print(end_t)

    # print(ind_ref)
    # print(fields)
    # print(rows[0])
    return start_t, end_t

def self_evaluation(audioPath, start_t, end_t):
    fs, audio_ref = ToolReadAudio(audioPath)
    audio_frame = audio_ref[math.ceil(start_t*fs): math.ceil(end_t*fs)]
    # print(audio_ref.size)
    # print(math.ceil(start_t * fs))

    chromagram_ref = chroma(audio_ref, sr=fs)
    chromagram_frame = chroma(audio_frame, sr=fs)

    d_matrix = Distance_matrix(chromagram_frame,chromagram_ref)
    c_matrix = Cost_matrix(chromagram_frame,chromagram_ref) 
    path = modified_DTW(c_matrix, runAll=False)
    print(path)

    # Plotting
    plt.subplot(2,1,1)
    plt.imshow(d_matrix, origin='lower', aspect='auto')
    # plt.plot(path[:, 1], path[:, 0], color='r')
    plt.clim([0, np.max(d_matrix)])
    plt.colorbar()
    plt.title('Subsequence - Lowest Cost Path (Distance Matrix)')
    plt.xlabel('Full')
    plt.ylabel('Subsequence')

    plt.subplot(2,1,2)
    plt.imshow(d_matrix, origin='lower', aspect='auto')
    plt.plot(path[:, 1], path[:, 0], color='r')
    plt.clim([0, np.max(d_matrix)])
    plt.colorbar()
    plt.title('Subsequence - Lowest Cost Path (Distance Matrix)')
    plt.xlabel('Full')
    plt.ylabel('Subsequence')

    plt.tight_layout()
    plt.show()

csv_filepath = "Assignments/7100 Research (Local File)/M06-1beat_time.csv"
ref_track = "pid9072-01"
audioPath = "Assignments/7100 Research (Local File)/mazurka06-1/pid9072-01.wav"
start_t, end_t = readCSV(csv_filepath, ref_track, 244, 288)
self_evaluation(audioPath, start_t, end_t)