import functions as f

def runCalculation(AudioFolder, ReferenceAudio):
    import os
    audioName_list = []
    reference_time = []
    snippet_time = []
    for file in os.listdir(AudioFolder):
        audioPath_snippet = AudioFolder + '/' + file
        audioPath_ref = ReferenceAudio # Put Reference Track here
        audioName = file.split('.')[0] # Audio name without '.wav'
        print(audioPath_snippet)
        if audioPath_snippet.split('.')[-1] != 'wav':
            continue

        fs, audio_ref = f.ToolReadAudio(ReferenceAudio)
        fs, audio_snippet = f.ToolReadAudio(audioPath_snippet)

        chromagram_ref = f.chroma(audio_ref, sr=fs)
        chromagram_snippet = f.chroma(audio_snippet, sr=fs)

        # d_matrix = f.Distance_matrix(chromagram_snippet,chromagram_ref)
        c_matrix = f.Cost_matrix_step(chromagram_snippet,chromagram_ref)
        # path, start_ind, end_ind = f.modified_DTW(c_matrix, runAll=False)
        path, start_ind, end_ind = f.modified_DTW_step(c_matrix)

        time4ref, time4other, refPath, otherPath = f.pathInd2Time(path, hop_len=512, fs=44100)

        # Average Slope Filter, x = timestamps for reference, y = timestamps for snippet
        filtered_x, filtered_y = f.averageSlope(otherPath, refPath, 512)

        audioName_list.append(audioName)
        reference_time.append(filtered_x)
        snippet_time.append(filtered_y)
    return audioName_list, reference_time