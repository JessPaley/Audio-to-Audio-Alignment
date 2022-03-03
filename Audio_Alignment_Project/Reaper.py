from reaper_python import *
import csv
import run

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

    funcitonOutput = run()

    referenceTimes = funcitonOutput[0]
    secondaryTrackTimes = fucntionOutput[1]
    # myFile = open("timestamps.csv")
    # csvreader = csv.reader(myFile)
    # for row in csvreader:
    #     referenceTimes.append(row[0])
    #     secondaryTrackTimes.append(row[1])


    subsequenceStartPosition = secondaryTrackTimes[0]  # position in seconds
    RPR_SetMediaItemPosition(clip3, subsequenceStartPosition, True)


    for i in range(0, len(secondaryTrackTimes)):
      output = RPR_SetTakeStretchMarker(myTake, -1, secondaryTrackTimes[i] + subsequenceStartPosition, referenceTimes[i])
      output = RPR_SetTakeStretchMarker(mySubsequence, -1, secondaryTrackTimes[i], referenceTimes[i])


main()

# move start of subsequence clip to correct location
# currently assumes there is one media item on each track
# align to first track

