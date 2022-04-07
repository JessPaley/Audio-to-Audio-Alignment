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

def rppWriter(AudioFolder,refAudio_dir):
    project = reaper.Project()

    # Importing Ref Audio
    fs, audioFile_ref = ToolReadAudio(refAudio_dir)
    audioFile_ref_len = audioFile_ref.size / fs
    Audio_ref = reaper.Source(file = refAudio_dir)
    item_ref = reaper.Item(Audio_ref, length=audioFile_ref_len, position=0)
    audioName = refAudio_dir.split('/')[-1]
    project.add(reaper.Track(item_ref, name=audioName))

    # Importing Snippets Folder
    for file in os.listdir(AudioFolder):
        audioPath_snippet = AudioFolder + '/' + file
        audioName = file.split('.')[0] # Audio name without '.wav'

        if audioPath_snippet.split('.')[-1] != 'wav': # skip file that is not .wav
            continue

        fs, audioFile = ToolReadAudio(audioPath_snippet)
        audio_length = audioFile.size / fs # Calculate the time of the audio file
        # print(audio_length)

        Audio = reaper.Source(file = audioPath_snippet)
        item = reaper.Item(Audio, length=audio_length, position=0) # Able to change track position here
        project.add(reaper.Track(item, name=audioName))
    
    project.write('test.rpp')
    return

AudioFolder = 'Audio Snippet Folder' # Put audio folder directory here
refAudio_dir = '7100 Research (Local File)/mazurka06-2/pid9090-01.wav'
rppWriter(AudioFolder,refAudio_dir)


### Test for writing one track ###
# Audio = reaper.Source(file = "/Users/Owen/Desktop/MUSI-6201/Assignments/Audio Snippet Folder/pid9063 snippet3.wav")
# # track = reaper.Track() # Create blank track()
# project = reaper.Project() # Create Project

# item = reaper.Item(Audio, length=15, position=4)
# project.add(reaper.Track(item))

# # for x in range(5):
# #     project.add(reaper.Track())

# # project.write("basic.rpp")