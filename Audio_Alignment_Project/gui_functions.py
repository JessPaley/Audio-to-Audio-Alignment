import functions as f

### .rpp Project Writer ###
def rppWriter(AudioFolder,refAudio_dir,position_t):
    import reathon.nodes as reaper
    import os
    project = reaper.Project()

    # Importing Ref Audio
    fs, audioFile_ref = f.ToolReadAudio(refAudio_dir)
    audioFile_ref_len = audioFile_ref.size / fs
    Audio_ref = reaper.Source(file = refAudio_dir)
    item_ref = reaper.Item(Audio_ref, length=audioFile_ref_len, position=0)
    audioName_ref = refAudio_dir.split('/')[-1]
    project.add(reaper.Track(item_ref, name=audioName_ref))

    # Importing Snippets Folder
    for file in os.listdir(AudioFolder):
        audioPath_snippet = AudioFolder + '/' + file
        audioName = file.split('.')[0] # Audio name without '.wav'

        if audioPath_snippet.split('.')[-1] != 'wav': # skip file that is not .wav
            continue

        fs, audioFile = f.ToolReadAudio(audioPath_snippet)
        audio_length = audioFile.size / fs # Calculate the time of the audio file
        # print(audio_length)

        Audio = reaper.Source(file = audioPath_snippet)
        item = reaper.Item(Audio, length=audio_length, position=position_t[audioName]) # Able to change track position here
        project.add(reaper.Track(item, name=audioName))
    
    project.write('aligned_project.rpp')
    return

