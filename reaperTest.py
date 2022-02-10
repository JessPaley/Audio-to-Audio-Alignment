referenceTimes = [4.435011299999999, 5.13161, 5.893083900000001, 6.3661451, 7.0102268]
secondaryTrackTimes = [3.8777324, 4.7600907, 5.553083900000001, 6.066145099999999, 6.6702268]

expected = []

for i in range(0, len(referenceTimes) - 1):
     time1 = referenceTimes[i + 1] - referenceTimes[i]
     time2 = secondaryTrackTimes[i + 1] - secondaryTrackTimes[i]
     expected.append(time1/ time2)

print(expected)