from reaper_python import *
# from reapy import reascript_api as RPR
import csv


def main():
    # items = RPR_CountMediaItems(0)
    # track1 = RPR_GetTrack(0, 0)
    # track2 = RPR_GetTrack(0, 1)
    # track3 = RPR_GetTrack(0, 2)
    # clip1 = RPR_GetMediaItem(0, 0)
    # clip2 = RPR_GetMediaItem(0, 1)
    # clip3 = RPR_GetMediaItem(0, 2)
    # myTake = RPR_GetTake(clip2, 0)
    # mySubsequence = RPR_GetTake(clip3, 0)

    # # use DTW
    # funcitonOutput = run()
    # referenceTimes = funcitonOutput[0]
    # secondaryTrackTimes = fucntionOutput[1]

    # Printing Message Function For Debugging
    def mb(message):
        RPR_MB(str(message), 'Print', 0)

    # myFile = open("../timestamps.csv")
    # csvreader = csv.reader(myFile)
    # csv_header = next(csvreader)
    # num_Snippet = len(csv_header)
    # rows = []
    # for row in csvreader:
    #     rows.append(row)
    # print(rows[0][0])

    for i in range(1, RPR_CountTracks(0)):
        track = RPR_GetTrack(0, i)
        trackName = RPR_GetTrackName(track, 'Track', 100)[2]

        # Get Index For Corresponding Track Name
        index = csv_header.index(trackName)
        clip = RPR_GetMediaItem(0, i)
        for j in range(1, RPR_CountTakes(clip)):
            take = RPR_GetTake(clip, j)
            RPR_SetTakeMarker(clip, float(rows[index][0]), True)

    # subsequenceStartPosition = secondaryTrackTimes[0]  # position in seconds
    # RPR_SetMediaItemPosition(clip3, subsequenceStartPosition, True)

    # for i in range(0, len(secondaryTrackTimes)):
    #   output = RPR_SetTakeStretchMarker(myTake, -1, secondaryTrackTimes[i] + subsequenceStartPosition, referenceTimes[i])
    #   output = RPR_SetTakeStretchMarker(mySubsequence, -1, secondaryTrackTimes[i], referenceTimes[i])


main()

# move start of subsequence clip to correct location
# currently assumes there is one media item on each track
# align to first track

# Extract Data from csv file, but user need to input audio files to Reaper themselves
