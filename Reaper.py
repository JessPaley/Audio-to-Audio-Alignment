from reaper_python import *

def main():
    items = RPR_CountMediaItems(0)
    track1 = RPR_GetTrack(0, 0)
    track2 = RPR_GetTrack(0, 1)
    track3 = RPR_GetTrack(0, 2)
    clip1 = RPR_GetMediaItem(0, 0)
    clip2 = RPR_GetMediaItem(0, 1)
    clip3 = RPR_GetMediaItem(0, 2)
    myTake = RPR_GetTake(clip2, 0)
    mySubsequence = RPR_GetTake(clip3, 0)
    # use DTW

    referenceTimes = [4.435011299999999, 5.13161, 5.893083900000001, 6.3661451, 7.0102268]
    secondaryTrackTimes = [3.8777324, 4.7600907, 5.553083900000001, 6.066145099999999, 6.6702268]

    RPR_SetMediaItemPosition(clip3, 5, True) # position in seconds


    for i in range(0, len(secondaryTrackTimes)):
      output = RPR_SetTakeStretchMarker(myTake, -1, secondaryTrackTimes[i], referenceTimes[i])
     
        

   

main()

# move start of subsequence clip to correct location
# currently assumes there is one media item on each track
# use stretch markers for each beat onset datum
# align to first track
# handle multiple items on a track?
