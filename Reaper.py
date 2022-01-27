from reaper_python import *

def main():
    items = RPR_CountMediaItems(0)
    track1 = RPR_GetTrack(0, 0)
    track2 = RPR_GetTrack(0, 1)
    clip1 = RPR_GetMediaItem(0, 0)
    clip2 = RPR_GetMediaItem(0, 1)
    # use DTW
    testArray = [1, 2, 3] # beat alignment from DTW
    for i in testArray:
      RPR_SetTakeStretchMarker(clip2, i, 0, 0)
    

   

main()

# move start of subsequence clip to correct location
# get item by take?
# use stretch markers for each beat onset datum
# align to first track
# handle multiple items on a track?
