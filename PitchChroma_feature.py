import librosa, librosa.display
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
    
    # x, sr = librosa.load(cAudioFilePath)
    # samplerate = sr
    # audio = x
    return(samplerate, audio)

### Pitch Chroma ###
def chroma(audio, sr=44100, blockSize=2048, hopSize=512):
    S = np.abs(librosa.stft(y=audio, n_fft=blockSize, hop_length=hopSize))
    chromagram = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=blockSize, hop_length=hopSize, n_chroma=12)
    t_chromagram = chromagram.T

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
    dis_matrix = cdist(sig1, sig2,'euclidean')
    return dis_matrix


file = '7100 Research (Local File)/pid1263-01.wav'
file2 = '7100 Research (Local File)/pid9048-01.wav'
chromaVec = featVector(file)
chromaVec2 = featVector(file2)

d_matrix = Distance_matrix(chromaVec, chromaVec2)
# plt.subplot(1, 2, 2)
plt.imshow(d_matrix, origin='lower', aspect='equal')
# plt.plot(dtw_calculation[:, 1], dtw_calculation[:, 0], marker='o', color='r')
plt.clim([0, np.max(d_matrix)])
plt.colorbar()
plt.title('Distance Matrix - Pitch Chroma')
plt.xlabel('Sequence Y')
plt.ylabel('Sequence X')

plt.tight_layout()
plt.show()
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
# plt.show()


