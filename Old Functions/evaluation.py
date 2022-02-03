import csv
import librosa, librosa.display
from librosa.util.utils import frame
import numpy as np
from numpy.lib import row_stack
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt

#____________________________________________________________________________________________________#
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
def featVector(path, analyzeDuration):
    fs, audio = ToolReadAudio(path)
    audio = audio[0 : analyzeDuration*fs] # analyze first x seconds
    chromagram = chroma(audio, sr=fs)
    return chromagram

### Calculate Euclidean Distance ###
def Distance_matrix(sig1, sig2):
    dis_matrix = cdist(sig1, sig2,'euclidean')
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
        n = d_matrix[0, i] + n
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
#____________________________________________________________________________________________________#


### TO DO ###
# Pick a reference recording (average duration)
# * pid9072-01

# Use the beat markings from the dataset (time) to find corresponding frames on DTW path calculation (frame)
# * dataset - convert time to sample: time * fs
# * DTW path - convert path index to sample: librosa.frames_to_samples

# Pick the frame: closest to the sample value of the dataset (after converting: time -> sample)

# Do a DTW calculation of the reference track with other recordings
# Find both the path tuple value (ref, other) of picked frames

# Compare the result:
# The frame (other) with the corresponding dataset onset (time)

# Find overall deviation

### Read in csv dataset ###
def readCSV(filepath, referenceTrack, analyzeDuration, otherTrack=None):
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

    if otherTrack == None:
        ind_ref = fields.index(referenceTrack)
        fields = np.array(fields)
        rows = np.array(rows)

        # csv_data = np.zeros([rows.shape[0], fields.shape[0]])
        # csv_data = np.vstack((fields, rows))

        # Return the column of choosen reference track
        beatOnsetTime_ref = rows[:,ind_ref]
        beatOnsetTime_ref = beatOnsetTime_ref.astype(np.float64)
        beatOnsetTime_ref = beatOnsetTime_ref[:np.argmin(np.abs(beatOnsetTime_ref - analyzeDuration))] # change length to analyze
        return beatOnsetTime_ref

    else:
        ind_other = fields.index(otherTrack)
        ind_ref = fields.index(referenceTrack)
        fields = np.array(fields)
        rows = np.array(rows)

        # Return the column of choosen reference track
        beatOnsetTime_ref = rows[:,ind_ref]
        beatOnsetTime_ref = beatOnsetTime_ref.astype(np.float64)
        beatOnsetTime_ref = beatOnsetTime_ref[:np.argmin(np.abs(beatOnsetTime_ref - analyzeDuration))] # change length to analyze
        
        # Return the column of other track
        beatOnsetTime_other = rows[:,ind_other]
        beatOnsetTime_other = beatOnsetTime_other.astype(np.float64)
        # beatOnsetTime_other = beatOnsetTime_other[:np.argmin(np.abs(beatOnsetTime_other - analyzeDuration))] # change length to analyze
        return beatOnsetTime_ref, beatOnsetTime_other

### Convert time oneset from dataset to sample ###
def time2Sample(fs, beatOnset, audio):
    convert = np.zeros(beatOnset.shape[0])
    for i in range(convert.shape[0]):
        convert[i] = beatOnset[i] * fs
    return convert

### Convert path indicies to sample ###
def pathInd2Sample(dtw_path, hop_length=512):
    convertInd1 = np.zeros(dtw_path.shape[0])
    convertInd2 = np.zeros(dtw_path.shape[0])
    for i in range(dtw_path.shape[0]):
        sample4Audio1, sample4Audio2 = librosa.frames_to_samples(dtw_path[i], hop_length=hop_length)
        convertInd1[i] = sample4Audio1
        convertInd2[i] = sample4Audio2
    return convertInd1, convertInd2

### Find the corresponding frame index on DTW of the reference track ###
def Evaluate_frame(csv_filepath, TrackName_ref, AudioPath_ref, AudioFolder, analyzeDuration):
    samplerate, audio = ToolReadAudio(AudioPath_ref)
    beatOnset_data = readCSV(csv_filepath, TrackName_ref, analyzeDuration)
    time2sample_data = time2Sample(samplerate, beatOnset_data, audio)

    import os
    for file in os.listdir(AudioFolder):
        other_record = AudioFolder + '/' + file
        reference_record = AudioFolder + '/' + TrackName_ref + '.wav' # Put Reference Track here
        audioName = file.split('.')[0]
        print(audioName,':')

        # Get the Onset data from CSV
        beatOnsetTime_ref, beatOnsetTime_other = readCSV(csv_filepath, TrackName_ref, analyzeDuration, otherTrack=audioName)
        beatOnsetTime_other = beatOnsetTime_other[0:beatOnsetTime_ref.shape[0]]
        # print(beatOnsetTime_ref.shape)
        # print(beatOnsetTime_other.shape)

        # Calculate Chroma
        chromaVec = featVector(other_record, analyzeDuration)
        chromaVec2 = featVector(reference_record, analyzeDuration) # Reference Trackm Chroma Calculation

        # Calculate DTW
        c_matrix = Cost_matrix(chromaVec, chromaVec2)
        dtw_path = DTW(c_matrix)

        # Convert Index to Sample
        ind1, ind2 = pathInd2Sample(dtw_path)

        # Find the corresponding frame index on DTW 
        frameInd = np.zeros(time2sample_data.shape[0])
        for i in range(time2sample_data.shape[0]):
            index = np.argmin(np.abs(ind1 - time2sample_data[i]))
            frameInd[i] = index
        print("framInd Size: ", frameInd.shape)

        # Separate the frameInd tuple into two 
        dtw_path_frame = []
        for i in range(frameInd.size):
            dtw_path_frame.append(dtw_path[ int(frameInd[i]) ].tolist())
        dtw_path_frame = np.array(dtw_path_frame)
        frameInd_ref = dtw_path_frame[:,0]
        frameInd_other = dtw_path_frame[:,1]
        # print(frameInd_ref.shape)
        # print(frameInd_other)

        # Convert Frame Ind back to Time 
        frameInd2Time_other = frameInd_other * 512 / 44100
        print("Onset Time Calculated From DTW: \n", frameInd2Time_other)

        # Sum of absolute differences (SAD), the sum of squared differences (SSD), Standard Deviation (STD)
        SAD_cal = np.sum(np.abs(frameInd2Time_other - beatOnsetTime_other))
        print("Sum of Absolute Differences:", SAD_cal)

        SSD_cal = np.sum(np.square(frameInd2Time_other - beatOnsetTime_other))
        print("Sum of Squared Differences:", SSD_cal)

        STD_cal = np.std(frameInd2Time_other-beatOnsetTime_other)
        print("Standard Deviation", STD_cal)
        # print("correlation:", np.corrcoef(np.array((frameInd2Time_other - beatOnsetTime_other)))[0, 1])

        print('______________________________________________________________________')
    # return SAD_cal, SSD_cal, STD_cal

csv_filepath = "7100 Research (Local File)/M06-1beat_time.csv"
TrackName_ref = "pid9072-01"
AudioPath_ref = "7100 Research (Local File)/mazurka06-1/pid9072-01.wav"
AudioFolder = "7100 Research (Local File)/mazurka06-1"
analyzeDuration = 30 #seconds

SAD_cal, SSD_cal, STD_cal = Evaluate_frame(csv_filepath, TrackName_ref, AudioPath_ref, AudioFolder, analyzeDuration)






# plt.imshow(d_matrix, origin='lower', aspect='equal')
# plt.scatter(dtw_path_ref_frame[:, 1], dtw_path_ref_frame[:, 0], s = 25, color='r')
# plt.plot(dtw_path[:, 1], dtw_path[:, 0], color='b')
# plt.clim([0, np.max(d_matrix)])
# plt.colorbar()
# plt.title('Distance Matrix - Pitch Chroma')
# plt.xlabel('Sequence Y')
# plt.ylabel('Sequence X')

# plt.tight_layout()
# plt.show()