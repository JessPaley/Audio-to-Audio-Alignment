import reathon.nodes as reaper
from scipy.io.wavfile import read as wavread
import os

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

project = reaper.Project()
AudioFolder = 'Audio Snippet Folder' # Put audio folder directory here

for file in os.listdir(AudioFolder):
    audioPath_snippet = AudioFolder + '/' + file
    audioName = file.split('.')[0] # Audio name without '.wav'

    if audioPath_snippet.split('.')[-1] != 'wav': # skip file that is not .wav
        continue

    fs, audioFile = ToolReadAudio(audioPath_snippet)
    audio_length = audioFile.size / fs # Calculate the time of the audio file
    print(audio_length)

    Audio = reaper.Source(file = audioPath_snippet)
    item = reaper.Item(Audio, length=audio_length, postion=0)
    project.add(reaper.Track(item, name=audioName))

project.write('test.rpp')


### Test for writing one track ###
# Audio = reaper.Source(file = "/Users/Owen/Desktop/MUSI-6201/Assignments/Audio Snippet Folder/pid9063 snippet3.wav")
# # track = reaper.Track() # Create blank track()
# project = reaper.Project() # Create Project

# item = reaper.Item(Audio, length=15, position=4)
# project.add(reaper.Track(item))

# # for x in range(5):
# #     project.add(reaper.Track())

# # project.write("basic.rpp")