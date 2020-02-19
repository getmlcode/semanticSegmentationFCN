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
        deepSemSeg_GUI_root.geometry("550x540+500+150")

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
        learnRateLabel.grid(row=9, column=0,sticky=E)

        self.initLearnRate = Entry(trainParamFrame, bd=4, width=5)
        self.initLearnRate.insert(0, "0.001")
        self.initLearnRate.grid(
            row=9, column=1, padx=2, sticky=W, ipadx=3, ipady=3
            )

        optAlgoList = ['Adam', 'Gradient Descent', 'Momentum']
        optAlgo = StringVar(trainParamFrame)
        optAlgo.set(optAlgoList[0])
        optAlgoOption = OptionMenu(
            trainParamFrame,
            optAlgo,
            *optAlgoList
            )
        optAlgoOption.config(width=12, font=('Helvetica', 9))
        optAlgoOption.grid(
            row=9, column=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
            )

        maxNormLabel = Label(
            trainParamFrame,
            text='Max Norm',
            font='Helvetica 9 bold',
            fg='blue'
            )
        maxNormLabel.grid(row=10, column=0, sticky=E)

        self.maxNorm = Entry(trainParamFrame, bd=4, width=5)
        self.maxNorm.insert(0, "0.1")
        self.maxNorm.grid(
            row=10, column=1, padx=2, sticky=W, ipadx=3, ipady=3
            )

        perfMetricList = ['IOU', 'F1-Measure']
        perfMetric = StringVar(trainParamFrame)
        perfMetric.set(perfMetricList[0])
        perfMetricOption = OptionMenu(
            trainParamFrame,
            perfMetric,
            *perfMetricList
            )
        perfMetricOption.config(width=12, font=('Helvetica', 9))
        perfMetricOption.grid(
            row=10, column=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
            )

        numOfEpochsLabel = Label(
            trainParamFrame,
            text='# Epochs',
            font='Helvetica 9 bold',
            fg='blue'
            )
        numOfEpochsLabel.grid(row=11, column=0, sticky=E)

        self.numOfEpochs = Entry(trainParamFrame, bd=4, width=5)
        self.numOfEpochs.insert(0, "5")
        self.numOfEpochs.grid(
            row=11, column=1, padx=2, sticky=W, ipadx=3, ipady=3
            )

        batchSizeList = ['32', '64', '128']
        batchSize = StringVar(trainParamFrame)
        batchSize.set(batchSizeList[0])
        batchSizeOption = OptionMenu(
            trainParamFrame,
            batchSize,
            *batchSizeList
            )
        batchSizeOption.config(width=12, font=('Helvetica', 9))
        batchSizeOption.grid(
            row=11, column=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
            )

        saveModelStatus = IntVar(trainParamFrame)
        saveModelChk = Checkbutton(
            trainParamFrame,
            variable=saveModelStatus,
            text='Save Model',
            fg='blue',
            font='Helvetica 9 bold',
            width=10,
            command=lambda : self.updateThresholdEntryStatus(saveModelStatus))
        saveModelChk.grid(
            row=12, column=2, sticky=W, ipadx=3, ipady=3
            )

        perfThreshLabel = Label(
            trainParamFrame,
            text='Threshold',
            font='Helvetica 9 bold',
            fg='blue'
            )
        perfThreshLabel.grid(row=12, column=0, sticky=E)

        self.perfThresh = Entry(trainParamFrame, bd=4, width=5)
        self.perfThresh.insert(0, "0.8")
        self.perfThresh.config(state='disabled')
        self.perfThresh.grid(
            row=12, column=1, padx=1, sticky=W, ipadx=3, ipady=2
            )

        showSegValImgsStatus = IntVar(trainParamFrame)
        showSegValImgs = Checkbutton(
            trainParamFrame,
            variable=showSegValImgsStatus,
            text='Show Images',
            fg='blue',
            font='Helvetica 9 bold',
            width=12
            )
        showSegValImgs.grid(
            row=13, column=2, sticky=W, ipadx=3, ipady=3
            )

        startTrainingButton = Button(
            trainParamFrame,
            image=folderImg,
            text="Start Training", 
            compound=LEFT,
        )
        startTrainingButton.image = folderImg
        startTrainingButton.grid(row=14, column=0, columnspan=3, pady=1, padx=10, sticky=NSEW)
        
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

        clearMsgButton = Button(statusMsgFrame, bg="cyan", text="Clear")
        clearMsgButton.pack(side=LEFT, fill=X, expand=1, padx=4, pady=4)

        saveMsgButton = Button(statusMsgFrame, bg="cyan", text="Save")
        saveMsgButton.pack(side=LEFT, fill=X, expand=1, padx=4, pady=4)

        # Inference Tab Design
        inferTab = ttk.Frame(tabControl)
        tabControl.add(inferTab, text="     Infer     ")

        tabControl.pack(expand=1, fill="both")
        return
    
    def updateThresholdEntryStatus(self, saveModelStatus):

        if(saveModelStatus.get()):
            self.perfThresh.config(state=NORMAL)
        else:
            self.perfThresh.config(state=DISABLED)

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
                self.validationDataPath.delete(0, END)
                self.validationDataPath.insert(0, currentDirectory)

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