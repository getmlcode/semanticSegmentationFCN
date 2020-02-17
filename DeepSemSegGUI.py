# from ImageSemanticSegmentor.FCN.FCN import FullyConvNet
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import os

class deepSemSeg_GUI:
    TestImage = []
    SegmentedImage = []

    def __init__(self, deepSemSeg_GUI_root):
        deepSemSeg_GUI_root.title("Deep Semantic Segmentation")
        deepSemSeg_GUI_root.minsize(500, 460)
        # deepSemSeg_GUI_root.maxsize(500, 460)
        deepSemSeg_GUI_root.geometry("500x460+500+150")

        # Read and resize open folder image
        folderImg = PhotoImage(file=os.getcwd() + "\\gui_imgs\\open.png").subsample(
            25, 25
        )

        # Create Tab Control
        tabControl = ttk.Notebook(deepSemSeg_GUI_root)

        # Train Tab Design

        # Parameter input frame design
        trainTab = ttk.Frame(tabControl)
        tabControl.add(trainTab, text="     Train     ")

        trainParamFrame = Frame(
            trainTab,
            highlightbackground="blue",
            width=50,
            height=50,
            highlightthickness=2,
        )
        trainParamFrame.pack(side=LEFT, fill=BOTH, expand=1, padx=2, pady=4)


        vggModelDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="VGG          ",
            compound=LEFT,
            command=lambda : self.setThisDirectory(0),
        )
        vggModelDirButton.image = folderImg
        vggModelDirButton.grid(row=0, pady=2, padx=2, sticky=NSEW)

        self.vggModelPath = Entry(trainParamFrame, bd=4, width=30)
        self.vggModelPath.insert(0, "Vgg Model Directory")
        self.vggModelPath.grid(
            row=0, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )


        trainDataDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Train         ",
            compound=LEFT,
            command=lambda : self.setThisDirectory(1),
        )
        trainDataDirButton.image = folderImg
        trainDataDirButton.grid(row=1, pady=1, padx=2, sticky=NSEW)

        self.trainDataPath = Entry(trainParamFrame, bd=4, width=30)
        self.trainDataPath.insert(0, "Train Data Directory")
        self.trainDataPath.grid(
            row=1, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        trainLabelDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Label         ", 
            compound=LEFT,
            command=lambda : self.setThisDirectory(2),
        )
        trainLabelDirButton.image = folderImg
        trainLabelDirButton.grid(row=2, pady=1, padx=2, sticky=NSEW)

        self.trainLabelPath = Entry(trainParamFrame, bd=4, width=30)
        self.trainLabelPath.insert(0, "Train Label Directory")
        self.trainLabelPath.grid(
            row=2, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        validationDataDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Validation",
            compound=LEFT,
            command=lambda : self.setThisDirectory(3),
        )
        validationDataDirButton.image = folderImg
        validationDataDirButton.grid(row=3, pady=1, padx=2, sticky=NSEW)

        self.validationDataPath = Entry(trainParamFrame, bd=4, width=30)
        self.validationDataPath.insert(0, "Validation Data Directory")
        self.validationDataPath.grid(
            row=3, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        testDataDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Test          ",
            compound=LEFT,
            command=lambda : self.setThisDirectory(4),
        )
        testDataDirButton.image = folderImg
        testDataDirButton.grid(row=4, pady=1, padx=2, sticky=NSEW)

        self.testDataPath = Entry(trainParamFrame, bd=4, width=30)
        self.testDataPath.insert(0, "Test Data Directory")
        self.testDataPath.grid(
            row=4, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        testResDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Result       ",
            compound=LEFT,
            command=lambda : self.setThisDirectory(5),
        )
        testResDirButton.image = folderImg
        testResDirButton.grid(row=5, pady=1, padx=2, sticky=NSEW)

        self.testResPath = Entry(trainParamFrame, bd=4, width=30)
        self.testResPath.insert(0, "Test Result Directory")
        self.testResPath.grid(
            row=5, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        learntModelDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="FCN         ",
            compound=LEFT,
            command=lambda : self.setThisDirectory(6),
        )
        learntModelDirButton.image = folderImg
        learntModelDirButton.grid(row=6, pady=1, padx=2, sticky=NSEW)

        self.learntModelPath = Entry(trainParamFrame, bd=4, width=30)
        self.learntModelPath.insert(0, "Learnt Model Directory")
        self.learntModelPath.grid(
            row=6, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        inferModelDirButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Infer         ",
            compound=LEFT,
            command=lambda : self.setThisDirectory(7),
        )
        inferModelDirButton.image = folderImg
        inferModelDirButton.grid(row=7, pady=1, padx=2, sticky=NSEW)

        self.inferModelPath = Entry(trainParamFrame, bd=4, width=30)
        self.inferModelPath.insert(0, "Inference Model Directory")
        self.inferModelPath.grid(
            row=7, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        learnRateLabel = Label(
            trainParamFrame,
            text='Learning Rate',
            font='Helvetica 9 bold',
            fg='blue'
            )
        learnRateLabel.grid(row=9, column=0)

        self.initLearnRate = Entry(trainParamFrame, bd=4, width=5)
        self.initLearnRate.insert(0, "0.001")
        self.initLearnRate.grid(
            row=9, column=1, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
            )


        trainParamFrame.grid_rowconfigure(8, minsize=12)

        # Status messages frame design
        statusMsgFrame = Frame(
            trainTab,
            highlightbackground="red",
            width=50,
            height=50,
            highlightthickness=2,
        )
        statusMsgFrame.pack(side=RIGHT, fill=BOTH, expand=1, padx=2, pady=4)

        statusMsgBox = Text(statusMsgFrame, bg="white", relief=SUNKEN)
        statusMsgBox.pack(side=TOP, fill=BOTH, expand=1)

        clearMsgButton = Button(statusMsgFrame, bg="cyan", text="Clear Messages")
        clearMsgButton.pack(side=LEFT, fill=X, expand=1, padx=4, pady=4)

        saveMsgButton = Button(statusMsgFrame, bg="cyan", text="Save Messages")
        saveMsgButton.pack(side=LEFT, fill=X, expand=1, padx=4, pady=4)

        # Inference Tab Design
        inferTab = ttk.Frame(tabControl)
        tabControl.add(inferTab, text="     Infer     ")

        tabControl.pack(expand=1, fill="both")
        return

    def setThisDirectory(self, Id):
        currentDirectory = filedialog.askdirectory()

        if len(currentDirectory) > 0:
            if Id == 0:
                self.vggModelPath.delete(0, END)
                self.vggModelPath.insert(0, currentDirectory)
            
            elif Id == 1:
                self.trainDataPath.delete(0, END)
                self.trainDataPath.insert(0, currentDirectory)

            elif Id == 2:
                self.trainLabelPath.delete(0, END)
                self.trainLabelPath.insert(0, currentDirectory)

            elif Id == 3:
                self.trainDataPath.delete(0, END)
                self.trainDataPath.insert(0, currentDirectory)

            elif Id == 4:
                self.testDataPath.delete(0, END)
                self.testDataPath.insert(0, currentDirectory)

            elif Id == 5:
                self.testResPath.delete(0, END)
                self.testResPath.insert(0, currentDirectory)

            elif Id == 6:
                self.learntModelPath.delete(0, END)
                self.learntModelPath.insert(0, currentDirectory)

            elif Id == 7:
                self.inferModelPath.delete(0, END)
                self.inferModelPath.insert(0, currentDirectory)

if __name__=='__main__':
    deepSemSeg_GUI_root = Tk()
    deepSemSeg_GUI(deepSemSeg_GUI_root)
    deepSemSeg_GUI_root.mainloop()