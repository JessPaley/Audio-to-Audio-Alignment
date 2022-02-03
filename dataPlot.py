import matplotlib.pyplot as plt

def dataPlot(txtPath):
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
    
    fig, ax = plt.subplots()
    ax.plot(label, start_GT_list, label="Ground Truth Start Time")
    ax.plot(label, end_GT_list, label="Ground Truth End Time")
    ax.plot(label, y, 'o', color='green')
    ax.plot(label, y2, 'o', color='red')
    plt.title("Index 244 to 288")
    plt.xticks(x,label,rotation=90, fontsize=7)
    plt.ylabel("Time (sec)")
    plt.legend()
    plt.show()


txtPath = "Assignments/7100 Research (Local File)/index_244_to_288 copy.txt"
dataPlot(txtPath)