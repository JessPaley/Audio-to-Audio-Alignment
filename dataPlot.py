import matplotlib.pyplot as plt
import numpy as np

### For finding repetition location ###
def readCSV(filepath, trackName, start_ind, end_ind, repetition_ind):
    import csv

    fields = []
    rows = []
    with open(filepath, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)    
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

    diff = end_ind-start_ind
    start_list = []
    end_list = []
    for i in range(0,len(repetition_ind)):
        ind4track = fields.index(trackName)
        start_t = float(rows[repetition_ind[i]][ind4track])
        end_t = float(rows[repetition_ind[i]+diff+1][ind4track])
        start_list.append(start_t)
        end_list.append(end_t)
    return start_list, end_list

### Starting and ending position plot ###
def positionPlot(txtPath):
    with open(txtPath) as f:
        contents = f.readlines()
        f.close

    start_GT = float(contents[0].split()[1])
    end_GT = float(contents[0].split()[2])

    y = []
    y2 = []
    x = []
    label = []
    start_GT_list = []
    end_GT_list = []
    for i in range(1, len(contents)):
        # print(contents[i].split())
        # tup = (float(contents[i].split()[1]), float(contents[i].split()[2]))
        y.append(float(contents[i].split()[1]))
        y2.append(float(contents[i].split()[2]))
        label.append(contents[i].split()[0])
        x.append(i-1)
        start_GT_list.append(start_GT)
        end_GT_list.append(end_GT)
    
    all_start = np.array([y])
    accuracy = (all_start < start_GT + 3) & (all_start > start_GT - 3)
    start_accuracy = np.sum(accuracy)/accuracy.size * 100

    all_end = np.array([y2])
    accuracy2 = (all_end < end_GT + 3) & (all_end > end_GT - 3)
    end_accuracy = np.sum(accuracy2)/accuracy2.size * 100
    
    fig, ax = plt.subplots()
    ax.plot(label, start_GT_list, label="Ground Truth Start Time")
    ax.plot(label, end_GT_list, label="Ground Truth End Time")
    ax.plot(label, y, 'o', color='green', label='start')
    ax.plot(label, y2, 'o', color='red', label='end')
    # plt.text(10,start_GT-8,"start accuracy: %d" %(start_accuracy), fontsize=10)
    # plt.text(10,end_GT+8,"end accuracy: %d" %(end_accuracy), fontsize=10)
    ax.set_ylim([0, max(y2)+20])
    plt.title("Index 25 to 49")
    plt.xticks(x,label,rotation=90, fontsize=7)
    plt.ylabel("Time (sec)")
    plt.legend()
    plt.show()

### Deviation Plot ###
def deviationPlot(txtPath, csvPath, repetition=False):
    with open(txtPath) as f:
        contents = f.readlines()
        f.close

    label = []
    start_deviation = []
    end_deviation = []
    if repetition == True:
        for i in range(1, len(contents)):
            trackName = contents[i].split()[0]
            label.append(trackName)
            start_list, end_list = readCSV(csvPath, trackName, start_ind, end_ind, repetition_ind)

            cal_dev_s = abs(np.array(start_list) - float(contents[i].split()[3]))
            cal_dev_e = abs(np.array(end_list) - float(contents[i].split()[4]))
            start_deviation.append(np.min(cal_dev_s))
            end_deviation.append(np.min(cal_dev_e))
    else:
        for i in range(1, len(contents)):
            trackName = contents[i].split()[0]
            label.append(trackName)
            cal_dev_s = abs(float(contents[i].split()[1]) - float(contents[i].split()[3]))
            cal_dev_e = abs(float(contents[i].split()[2]) - float(contents[i].split()[4]))
            start_deviation.append(cal_dev_s)
            end_deviation.append(cal_dev_e)
    
    X = np.arange(len(start_deviation))
    fig, ax = plt.subplots()
    ax.bar(X + 0.00, start_deviation, color = 'b', width = 0.35, edgecolor ='grey', label ='start time')
    ax.bar(X + 0.35, end_deviation, color = 'g', width = 0.35, edgecolor ='grey', label ='end time')
    plt.xticks([r + 0.35 for r in range(len(start_deviation))],label,rotation=90, fontsize=7)
    plt.ylabel("Time Deviation (sec)")
    plt.title("Mazurka06-1 Time Deviation (Snippet index 169 to 195)")
    plt.legend()
    plt.show()

start_ind = 169 
end_ind = 195
repetition_ind = [97,169] # Give all the location where the snippet is repeated in the piece
txtPath = "index 169 to 195"
csvPath = "Assignments/7100 Research (Local File)/M06-1beat_time.csv"
# dataPlot(txtPath)
deviationPlot(txtPath, csvPath, repetition=True)