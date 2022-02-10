from reaper_python import *

def main():
    items = RPR_CountMediaItems(0)
    track1 = RPR_GetTrack(0, 0)
    track2 = RPR_GetTrack(0, 1)
    clip1 = RPR_GetMediaItem(0, 0)
    clip2 = RPR_GetMediaItem(0, 1)
    myTake = RPR_GetTake(clip2, 0)
    # use DTW

    # file1 = open('./MazurkaCSVs/M06-1beat_time.csv', 'r')
    # lines = file1.readLines()
    referenceTimes = [4.435011299999999, 5.13161, 5.893083900000001, 6.3661451, 7.0102268]
    secondaryTrackTimes = [3.8777324, 4.7600907, 5.553083900000001, 6.066145099999999, 6.6702268]


    # for i in lines:
    #     beatData = i.split()
    #     referenceTimes.append(beatData[3])
    #     secondaryTrackTimes.append(beatData[5])

    for i in range(0, len(secondaryTrackTimes)):
      output = RPR_SetTakeStretchMarker(myTake, -1, secondaryTrackTimes[i], referenceTimes[i])
     
        

   

main()

# move start of subsequence clip to correct location
# get item by take?
# use stretch markers for each beat onset datum
# align to first track
# handle multiple items on a track?
