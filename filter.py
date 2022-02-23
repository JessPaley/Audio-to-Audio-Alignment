import librosa, librosa.display
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt
import math
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
    for i in range(0, len(path)):
        sample_ref = librosa.frames_to_samples(path[i][1], hop_length=hop_len)
        sample_other = librosa.frames_to_samples(path[i][0], hop_length=hop_len)
        time4ref.append(sample_ref/fs)
        time4other.append(sample_other/fs)
    return time4ref, time4other

### Create Plot ###
def plot(d_matrix, path):
    # Plotting
    plt.subplot(2,1,1)
    plt.imshow(d_matrix, origin='lower', aspect='auto')
    plt.clim([0, np.max(d_matrix)])
    plt.colorbar()
    plt.title('Subsequence - DTW (Pitch Chroma)')
    plt.xlabel('Full')
    plt.ylabel('Subsequence')

    # Plot with path
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

def readCSV(filepath, trackName, start_ind, end_ind, trackName_ref="pid9072-01"):
    import csv

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
    
    ind_ref = fields.index(trackName_ref)
    start_GT = float(rows[start_ind][ind_ref])
    end_GT = float(rows[end_ind+1][ind_ref])
    print("Ground Truth starting time: ", start_GT)
    print("Ground Truth ending: ", end_GT)

    ind_test = fields.index(trackName)
    start_t = float(rows[start_ind][ind_test])
    end_t = float(rows[end_ind+1][ind_test])
    print("%s starting time: %f" %(trackName, start_t))
    print("%s ending time: %f" %(trackName, end_t))

    return start_t, end_t, start_GT, end_GT

### Moving Average Filter ###
def MAfilter(signal, windowSize):
    filtered_signal = np.convolve(signal, np.ones(windowSize), 'valid') / windowSize
    
    filtered_signal_test = []
    for i in range(0, len(signal)-windowSize+1):
        ind = np.sum(signal[i:i+windowSize])/windowSize
        filtered_signal_test.append(round(ind,2))
    return filtered_signal, filtered_signal_test

def output(audioPath_ref, audioPath_test, start_t, end_t):
    fs, audio_ref = ToolReadAudio(audioPath_ref)
    fs, audio_tracks = ToolReadAudio(audioPath_test)
    audio_snippet = audio_ref[math.ceil(start_t*fs): math.ceil(end_t*fs)] # cuts out the snippet

    chromagram_tracks = chroma(audio_tracks, sr=fs)
    chromagram_snippet = chroma(audio_snippet, sr=fs)

    d_matrix = Distance_matrix(chromagram_snippet,chromagram_tracks)
    c_matrix = Cost_matrix(chromagram_snippet,chromagram_tracks)
    path, start_ind, end_ind = modified_DTW(c_matrix, runAll=False)
    # plot(d_matrix, path)
    time4ref, time4other = pathInd2Time(path, hop_len=512, fs=44100)

    # test = [1,2,3,4,5,6,7,8,9,10]
    filtered_signal, filtered_signal_test = MAfilter(time4ref, 128)
    filtered_signal_2, filtered_signal_test_2 = MAfilter(time4other, 128)

    print(filtered_signal_test)
    print(filtered_signal_test_2)
    return time4ref, time4other, filtered_signal_test, filtered_signal_test_2

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



audioPath_ref = "7100 Research (Local File)/mazurka06-1/pid9072-01.wav"
audioPath_test = "7100 Research (Local File)/mazurka06-1/pid9063-01.wav"

# Select time to cut the clip from the audioPath_test audio
start_t = 133.2
end_t = 152.6
time4ref,time4other,filtered_ref,filtered_test = output(audioPath_ref, audioPath_test, start_t, end_t)
writeCSV(audioPath_ref, audioPath_test, filtered_ref, filtered_test)

# # Plot
# plt.subplot(1,2,1)
# plt.plot(time4ref, time4other)
# plt.title("Original")
# plt.xlabel("Reference Track Time")
# plt.ylabel("Subsequence Time")
# plt.subplot(1,2,2)
# plt.plot(filtered_ref, filtered_test)
# plt.title("Filtered")
# plt.xlabel("Reference Track Time")
# plt.ylabel("Subsequence Time")
# plt.show()
