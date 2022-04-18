local py = require "python"
local run = py.import "run"

function main()
    items = reaper.CountMediaItems(0)
    track1 = reaper.GetTrack(0, 0)
    track2 = reaper.GetTrack(0, 1)
    track3 = reaper.GetTrack(0, 2)
    clip1 = reaper.GetMediaItem(0, 0)
    clip2 = reaper.GetMediaItem(0, 1)
    clip3 = reaper.GetMediaItem(0, 2)
    myTake = reaper.GetTake(clip2, 0)
    mySubsequence = reaper.GetTake(clip3, 0)

    funcitonOutput = run()

    referenceTimes = funcitonOutput[0]
    secondaryTrackTimes = fucntionOutput[1]



    subsequenceStartPosition = secondaryTrackTimes[0]
    reaper.SetMediaItemPosition(clip3, subsequenceStartPosition, True)


    for i=0, len(secondaryTrackTimes) do
      output = reaper.SetTakeStretchMarker(myTake, -1, secondaryTrackTimes[i] + subsequenceStartPosition, referenceTimes[i])
      output = reaper.SetTakeStretchMarker(mySubsequence, -1, secondaryTrackTimes[i], referenceTimes[i])
    end

end


main()
