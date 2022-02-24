import csv

def main():
    referenceTimes = []
    secondaryTrackTimes = []

    myFile = open("timestamps_averageSlope.csv")
    csvreader = csv.reader(myFile)
    for row in csvreader:
        referenceTimes.append(float(row[0]))
        secondaryTrackTimes.append(float(row[1]))
    print(referenceTimes)


main()