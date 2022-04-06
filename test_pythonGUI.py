from tkinter import *
from tkinter import filedialog

root = Tk()
root.title('File Directory Input Helper')
# root.filename = filedialog.askopenfilename(initialdir='/', title='Select Folder')

# # Input box
# e = Entry(root) 
# e.pack()

# Function used in myButton command
def ButtonClick1():
    # label = Label(root, text='Choose Audio Snippet Folder')
    folder4snippet = filedialog.askdirectory()
    return folder4snippet
def ButtonClick2():
    # label = Label(root, text='Choose Audio Snippet Folder')
    refAudio_directory = filedialog.askdirectory()
    return refAudio_directory

# Creating a label widget
mylabel = Label(root, text='Hello World!')
mylabel2 = Label(root, text='Audio Snippet Folder')
mylabel3 = Label(root, text='Reference Audio Track')
mylabel.grid(row=0, column=0)
mylabel2.grid(row=1, column=0)
mylabel3.grid(row=2, column=0)

# Create Buttons
myButton = Button(root, text='Choose', padx=25, command=ButtonClick1)
myButton2 = Button(root, text='Choose', padx=25, command=ButtonClick2)
myButton.grid(row=1, column=50)
myButton2.grid(row=2, column=50)


root.mainloop()
