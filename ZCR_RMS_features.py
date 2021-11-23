import librosa
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt

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

### Z-Score Normalize ###
def norm_zscore(array):
    mean = np.mean(array)
    std = np.std(array)
    normalized = np.zeros(len(array))
    for i in range(len(array)):
        normalized[i] = (array[i] - mean)/std
    return normalized

### ZCR ###
def ZCR(audio, blockSize=1024, hopSize=512):
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=blockSize, hop_length=hopSize)
    zcr_norm = norm_zscore(zcr[0])
    return zcr_norm

### RMS ###
def RMS(audio, blockSize=1024, hopSize=512, fs=None):
    rms = librosa.feature.rms(y=audio, S=fs, frame_length=blockSize, hop_length=hopSize, center=True, pad_mode='reflect')
    rms_norm = norm_zscore(rms[0])
    return rms_norm

### Feature Vector ###
def featVector(path):
    fs, audio = ToolReadAudio(path)
    zcr = ZCR(audio)
    rms = RMS(audio)

    feature_vec = np.zeros([zcr.shape[0], 2])
    for i in range(len(zcr)):
        vec = np.array([zcr[i], rms[i]])
        feature_vec[i] = vec
    return feature_vec

### Calculate Euclidean Distance ###
def Distance_matrix(sig1, sig2):
    dis_matrix = cdist(sig1, sig2,'euclidean')
    # if sig1.shape[1] > 1:
    #     dis_matrix = cdist(sig1, sig2,'euclidean')
    # else:
    #     dis_matrix = np.zeros((sig1.shape[0], sig2.shape[0]))
    #     column = dis_matrix.shape[0]
    #     row = dis_matrix.shape[1]
    #     for i in range(column):
    #         for j in range(row):
    #             dis_matrix[i,j] = np.linalg.norm(sig1[i]-sig2[j])
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


file = '7100 Research (Local File)/pid1263-01.wav'
file2 = '7100 Research (Local File)/pid9048-01.wav'
FeatureVec = featVector(file)
FeatureVec2 = featVector(file2)

# c_matrix = Cost_matrix(FeatureVec, FeatureVec2)[1]
d_matrix = Distance_matrix(FeatureVec, FeatureVec2)

### Plotting ###
# plt.figure(figsize=(9, 3))
# plt.subplot(1, 2, 1)
# plt.imshow(c_matrix, origin='lower', aspect='equal')
# # plt.plot(dtw_calculation[:, 1], dtw_calculation[:, 0], color='r')
# plt.clim([0, np.max(c_matrix)])
# plt.colorbar()
# plt.title('Cost Matrix With Path')
# plt.xlabel('Sequence Y')
# plt.ylabel('Sequence X')

# plt.subplot(1, 2, 2)
plt.imshow(d_matrix, origin='lower', aspect='equal')
# plt.plot(dtw_calculation[:, 1], dtw_calculation[:, 0], marker='o', color='r')
plt.clim([0, np.max(d_matrix)])
plt.colorbar()
plt.title('Distance Matrix - RMS/ZCR')
plt.xlabel('Sequence Y')
plt.ylabel('Sequence X')

plt.tight_layout()
plt.show()