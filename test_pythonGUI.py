from tkinter import *
from tkinter import filedialog
from test_rppWriter import rppWriter

root = Tk()
root.title('File Directory Input Helper')
root.geometry('550x250')
# root.filename = filedialog.askopenfilename(initialdir='/', title='Select Folder')

# Function used in myButton command
SnippetPath = StringVar()
refAudioPath = StringVar()

def ButtonClick1():
    # label = Label(root, text='Choose Audio Snippet Folder')
    folder4snippet = filedialog.askdirectory()
    SnippetPath.set(folder4snippet)

def ButtonClick2():
    # label = Label(root, text='Choose Audio Snippet Folder')
    # refAudio_directory = filedialog.askdirectory()
    refAudio_directory =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("wav files","*.wav"),("all files","*.*")))
    refAudioPath.set(refAudio_directory)

def ButtonRun():
    folder_dir = SnippetPath.get()
    refAudio_dir = refAudioPath.get()
    # print(folder_dir)
    # print(refAudio_dir)
    rppWriter(folder_dir,refAudio_dir)
    return 

# Creating a label widget
mylabel = Label(root, text='File Directory Input Helper')
mylabel2 = Label(root, text='Audio Snippet Folder')
mylabel3 = Label(root, text='Reference Audio Track')
mylabel.grid(row=0, column=0)
mylabel2.grid(row=1, column=0)
mylabel3.grid(row=2, column=0)

# Create Buttons
myButton = Button(root, text='Choose', padx=25, command=ButtonClick1)
myButton2 = Button(root, text='Choose', padx=25, command=ButtonClick2)
myButton3 = Button(root, text='Run', padx=50, command=ButtonRun)
Snippet_dir_label = Entry(root,textvariable=SnippetPath)
refAudio_dir_label = Entry(root,textvariable=refAudioPath)
myButton.grid(row=1, column=50)
myButton2.grid(row=2, column=50)
myButton3.grid(row=3, column=25)
Snippet_dir_label.grid(row=1, column=25)
refAudio_dir_label.grid(row=2, column=25)


root.mainloop()
