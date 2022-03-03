import functions as f
import FileDirectory as c

# Trigger the project

def run():
    fs, audio_ref = f.ToolReadAudio(c.ReferenceAudio)
    fs, audio_snippet = f.ToolReadAudio(c.AudioSnippet)

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
    
    # Output csv File
    f.writeCSV(c.ReferenceAudio, c.AudioSnippet, filtered_x, filtered_y)

# if __name__ == '__main__':
run()
