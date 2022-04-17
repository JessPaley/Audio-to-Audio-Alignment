import functions as f
from tkinter import Tk,StringVar,filedialog,Label,Button,Entry,ttk
import gui_functions as gui
import FileDirectory as c
import runCalculation as cal

import time
import threading
import os

# Trigger the project
def run():
    ### GUI Labels and Buttons###
    root = Tk()
    root.title('File Directory Input Helper')
    root.geometry('800x250')
    # Create path and text variables
    SnippetPath = StringVar()
    refAudioPath = StringVar()
    outputPath = StringVar()
    text = StringVar() # Text that display progress

    # Function for folder choosing buttons
    def ButtonClick1():
        folder4snippet = filedialog.askdirectory()
        SnippetPath.set(folder4snippet)
        count = 0
        for file in os.listdir(folder4snippet):
            audioPath_snippet = folder4snippet + '/' + file
            if audioPath_snippet.split('.')[-1] != 'wav':
                continue
            count += 1
        text.set("There are {} audio files found".format(count))
        myButton3["state"] = "normal"
    def ButtonClick2():
        refAudio_directory =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("wav files","*.wav"),("all files","*.*")))
        refAudioPath.set(refAudio_directory)
        if refAudio_directory == "":
            myButton3["state"] = "disable"
        else:
            myButton3["state"] = "normal"
    def ButtonClick3():
        folder4output = filedialog.askdirectory()
        outputPath.set(folder4output)
        text.set("Output location is set to\n{}".format(folder4output))
    def ButtonClick4():
        output_dir = outputPath.get()
        ReaperFilePath = '{}/aligned_project.rpp'.format(output_dir)
        gui.file_open(ReaperFilePath)

    # The run button that runs everything  
    def ButtonRun():
        folder_dir = SnippetPath.get()
        refAudio_dir = refAudioPath.get()
        output_dir = outputPath.get()

        # Read File, Run Alignment Calculation
        audioName_list, reference_time = cal.runCalculation(folder_dir, refAudio_dir)
        
        # # Output csv File
        # f.csv_writer_row(audioName_list, reference_time)

        # Output Position Dictionary
        pos_t = f.position_dict(audioName_list, reference_time)

        # Create .rpp Project with alignment information
        gui.rppWriter(folder_dir,refAudio_dir,output_dir,pos_t) # Write Reaper File
        
        # time.sleep(1) # Simulate doing something
        text.set("Reaper Project Created \n (click Open Project button to open Reaper)")
        myButton5["state"] = "normal"
        return

    # Creating a label widget
    mylabel = Label(root, text='File Directory Input Helper')
    mylabel2 = Label(root, text='Audio Snippet Folder \n (these files will be aligned to the main track)')
    mylabel3 = Label(root, text='Main Audio Track \n (the other files will be aligned to this one)')
    mylabel4 = Label(root, textvariable = text,
                 bg='#fff', fg='#f00', pady=10, padx=10, font=10)
    mylabel5 = Label(root, text='Output Reaper File \n (choose output location & open the generated project)')
    mylabel.grid(row=0, column=0)
    mylabel2.grid(row=1, column=0)
    mylabel3.grid(row=2, column=0)
    mylabel4.grid(row=5, column=25) # Output Status Label
    mylabel5.grid(row=3, column=0) # Output Path Label

    # Create Buttons and Entry Box
    myButton = Button(root, text='Select Folder', padx=25, command=ButtonClick1)
    myButton2 = Button(root, text='Select Main Track', padx=25, command=ButtonClick2)
    myButton3 = Button(root, text='Run', padx=50, command=ButtonRun, state='disable') 
    myButton4 = Button(root, text='Output Location', padx=25, command=ButtonClick3) # Open or choose Reaper file directory
    myButton5 = Button(root, text='Open Project', padx=25, command=ButtonClick4, state='disable') # Open or choose Reaper file directory
    Snippet_dir_label = Entry(root,textvariable=SnippetPath)
    refAudio_dir_label = Entry(root,textvariable=refAudioPath)

    myButton.grid(row=1, column=50)
    myButton2.grid(row=2, column=50)
    myButton3.grid(row=4, column=25) # Run Button
    myButton4.grid(row=3, column=25)
    myButton5.grid(row=3, column=50)
    Snippet_dir_label.grid(row=1, column=25)
    refAudio_dir_label.grid(row=2, column=25)

    # # Progress Bar
    # my_progress = ttk.Progressbar(root, orient='horizontal', 
    #     length=300, mode='indeterminate')
    # my_progress.grid(row=4, column=25)

    root.mainloop()
    return

if __name__ == '__main__':
    run()
