import functions as f
import FileDirectory as c

# Trigger the project

def run():
    import os
    import numpy as np
    for file in os.listdir(c.AudioFolder):
        audioPath_snippet = c.AudioFolder + '/' + file
        audioPath_ref = c.ReferenceAudio # Put Reference Track here
        audioName = file.split('.')[0] # Audio name without '.wav'
        if audioPath_snippet.split('.')[-1] == 'DS_Store':
            continue

        fs, audio_ref = f.ToolReadAudio(c.ReferenceAudio)
        fs, audio_snippet = f.ToolReadAudio(audioPath_snippet)

        chromagram_ref = f.chroma(audio_ref, sr=fs)
        chromagram_snippet = f.chroma(audio_snippet, sr=fs)

        # d_matrix = f.Distance_matrix(chromagram_snippet,chromagram_ref)
        c_matrix = f.Cost_matrix(chromagram_snippet,chromagram_ref)
        path, start_ind, end_ind = f.modified_DTW(c_matrix, runAll=False)

        time4ref, time4other, refPath, otherPath = f.pathInd2Time(path, hop_len=512, fs=44100)

        # # Moving Average Filter
        # filtered_signal, filtered_signal_test = f.MAfilter(time4ref, 128)
        # filtered_signal_2, filtered_signal_test_2 = f.MAfilter(time4other, 128)

        # Average Slope Filter
        filtered_x, filtered_y = f.averageSlope(otherPath, refPath, 512)
        print(audioName)
        print("Start: {}; End: {}" .format(filtered_x[0],filtered_x[-1]))
        print("Evenly Spaced: ", np.linspace(filtered_x[0],filtered_x[-1],50))
        
        # # Output csv File
        # f.writeCSV(c.ReferenceAudio, c.AudioSnippet, filtered_x, filtered_y)

# if __name__ == '__main__':
run()
