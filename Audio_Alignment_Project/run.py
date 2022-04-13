import functions as f
from tkinter import Tk,StringVar,filedialog,Label,Button,Entry
import gui_functions as gui
import FileDirectory as c
import runCalculation as cal

# Trigger the project

def run():
    ### GUI Labels and Buttons###
    root = Tk()
    root.title('File Directory Input Helper')
    root.geometry('550x250')
    # Create path variables
    SnippetPath = StringVar()
    refAudioPath = StringVar()

    # Function for buttons
    def ButtonClick1():
        folder4snippet = filedialog.askdirectory()
        SnippetPath.set(folder4snippet)
    def ButtonClick2():
        refAudio_directory =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("wav files","*.wav"),("all files","*.*")))
        refAudioPath.set(refAudio_directory)
    def ButtonRun():
        folder_dir = SnippetPath.get()
        refAudio_dir = refAudioPath.get()
        # Read File, Run Alignment Calculation
        audioName_list, reference_time = cal.runCalculation(folder_dir, refAudio_dir)
        
        # # Output csv File
        # f.csv_writer_row(audioName_list, reference_time)

        # Output Position Dictionary
        pos_t = f.position_dict(audioName_list, reference_time)

        # Create .rpp Project with alignment information
        gui.rppWriter(folder_dir,refAudio_dir,pos_t) # Write Reaper File
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
    return

if __name__ == '__main__':
    run()
