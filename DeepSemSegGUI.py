from ImageSemanticSegmentor.FCN.FCN import FullyConvNet
import threading
import tensorflow as tf
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
        deepSemSeg_GUI_root.geometry("650x540+500+150")

        # Register entry widget's validation functions
        self.onlyFloat = (deepSemSeg_GUI_root.register(self.validateForFloat), '%P')
        self.onlyInt = (deepSemSeg_GUI_root.register(self.validateForInt), '%P')

        # Read and resize open folder image
        folderImg = PhotoImage(file=os.getcwd() + "\\gui_imgs\\open.png").subsample(
            25, 25
        )
        trainImg = PhotoImage(file=os.getcwd() + "\\gui_imgs\\train.png").subsample(
            18, 18
        )

        # Create Tab Control
        tabControl = ttk.Notebook(deepSemSeg_GUI_root)

        # Train Tab Design

        # Parameter input frame design
        trainTab = ttk.Frame(tabControl)
        tabControl.add(trainTab, text="     Train     ")

        trainParamFrame = Frame(
            trainTab,
            #highlightbackground="blue",
            highlightthickness=2,
        )
        trainParamFrame.grid(row=0, column=0, sticky=NSEW)

        # Set minisize for row-8
        trainParamFrame.grid_rowconfigure(8, minsize=12)

        # Create GUI elements
        self.renderDirectoryInputGUI(trainParamFrame, folderImg)
        self.renderTrainParamInputGUI(trainParamFrame)
        self.renderValidateAndSaveModelInputGUI(trainParamFrame)

        startTrainingButton = Button(
            trainParamFrame,
            image=trainImg,
            fg='black',
            #font='Helvetica 10',
            compound=LEFT,
        )
        startTrainingButton.image = trainImg
        startTrainingButton.grid(row=14, column=0, columnspan=3, pady=4, padx=10, sticky=NSEW)
        startTrainingButton.config(command=lambda : self.startTraingInNewThread(statusMsgBox))

        
        # Status messages frame design
        
        statusMsgFrame = Frame(
            trainTab,
            #highlightbackground="red",
            relief=SUNKEN,
            highlightthickness=1,
            bd=2,
            width=350,
            height=400
        )
        statusMsgFrame.grid(row=0, column=1, columnspan=100, rowspan=100, sticky=NSEW)
        statusMsgFrame.columnconfigure(1, weight=1)
        statusMsgFrame.columnconfigure(0, weight=1)
        statusMsgFrame.rowconfigure(0, weight=1)
        statusMsgFrame.grid_propagate(False)

        statusMsgBox = Text(
            statusMsgFrame,
            bg="white",
            relief=SUNKEN,
            state=DISABLED,
            width=350,
            height=300
            )
        statusMsgBox.grid(row=0, column=0, columnspan=2, sticky=NSEW)
        scrollbar = Scrollbar(
            statusMsgBox,
            cursor='hand2'
            )
        scrollbar.pack(side = RIGHT, fill = Y)
        statusMsgBox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=statusMsgBox.yview)

        clearMsgButton = Button(statusMsgFrame, bg="#CCFFFF", text="Clear")
        clearMsgButton.grid(row=1, column=0, pady=4, padx=2, sticky=NSEW)
        clearMsgButton.config(command=lambda : self.clearMessages(statusMsgBox))

        saveMsgButton = Button(statusMsgFrame, bg="#CCFFFF", text="Save")
        saveMsgButton.grid(row=1, column=1, pady=4, padx=2, sticky=NSEW)
        saveMsgButton.config(command=lambda : self.saveMessages(statusMsgBox))

        # Inference Tab Design
        inferTab = ttk.Frame(tabControl)
        tabControl.add(inferTab, text="     Infer     ")

        tabControl.pack(expand=1, fill="both")


    # ---- Rendering Functions ----
    
    def renderValidateAndSaveModelInputGUI(self,trainParamFrame):
        self.saveModelStatus = IntVar(trainParamFrame)
        saveModelChk = Checkbutton(
            trainParamFrame,
            variable=self.saveModelStatus,
            text='Save Model',
            fg='blue',
            font='Helvetica 9 bold',
            width=10,
            command=lambda : self.updateThresholdEntryStatus(self.saveModelStatus)
            )
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

        self.perfThresh = Entry(
            trainParamFrame,
            validate='key',
            validatecommand=self.onlyFloat,
            bd=4,
            width=5)
        self.perfThresh.insert(0, "0.8")
        self.perfThresh.config(state='disabled')
        self.perfThresh.grid(
            row=12, column=1, padx=1, sticky=W, pady=2, ipadx=3, ipady=2
            )

        self.showSegValImgsStatus = IntVar(trainParamFrame)
        showSegValImgs = Checkbutton(
            trainParamFrame,
            variable=self.showSegValImgsStatus,
            text='Show Images',
            fg='blue',
            font='Helvetica 9 bold',
            width=12
            )
        showSegValImgs.grid(
            row=13, column=2, sticky=W, ipadx=3, ipady=3
            )


    def renderTrainParamInputGUI(self,trainParamFrame):
        learnRateLabel = Label(
            trainParamFrame,
            text='Learning Rate',
            font='Helvetica 9 bold',
            fg='blue'
            )
        learnRateLabel.grid(row=9, column=0, sticky=E)

        self.initLearnRate = Entry(
            trainParamFrame,
            validate='key',
            validatecommand=self.onlyFloat,
            bd=4,
            width=5)
        self.initLearnRate.insert(0, "0.001")
        self.initLearnRate.grid(
            row=9, column=1, padx=2, sticky=W, ipadx=3, ipady=3
            )

        optAlgoList = ['Adam', 'Gradient Descent', 'Momentum']
        self.optAlgo = StringVar(trainParamFrame)
        self.optAlgo.set(optAlgoList[0])
        optAlgoOption = OptionMenu(
            trainParamFrame,
            self.optAlgo,
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

        self.maxNorm = Entry(
            trainParamFrame,
            validate='key',
            validatecommand=self.onlyFloat,
            bd=4,
            width=5)
        self.maxNorm.insert(0, "0.1")
        self.maxNorm.grid(
            row=10, column=1, padx=2, sticky=W, ipadx=3, ipady=3
            )


        perfMetricList = ['IOU', 'F1']
        self.perfMetric = StringVar(trainParamFrame)
        self.perfMetric.set(perfMetricList[0])
        perfMetricOption = OptionMenu(
            trainParamFrame,
            self.perfMetric,
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

        self.numOfEpochs = Entry(
            trainParamFrame,
            validate='key',
            validatecommand=self.onlyInt,
            bd=4,
            width=5)
        self.numOfEpochs.insert(0, "5")
        self.numOfEpochs.grid(
            row=11, column=1, padx=2, sticky=W, ipadx=3, ipady=3
            )

        batchSizeList = [32, 64, 128]
        self.batchSize = IntVar(trainParamFrame)
        self.batchSize.set(batchSizeList[0])
        batchSizeOption = OptionMenu(
            trainParamFrame,
            self.batchSize,
            *batchSizeList
            )
        batchSizeOption.config(width=12, font=('Helvetica', 9))
        batchSizeOption.grid(
            row=11, column=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
            )

        numOfClassLabel = Label(
            trainParamFrame,
            text='# Class',
            font='Helvetica 9 bold',
            fg='blue'
            )
        numOfClassLabel.grid(row=13, column=0, sticky=E)

        self.numOfClasses = IntVar(trainParamFrame, value=2)
        showSegValImgs = Spinbox(
            trainParamFrame,
            textvariable=self.numOfClasses,
            from_=2,
            to=10000,
            bd=4,
            width=5,
            cursor='hand2'
            )
        showSegValImgs.grid(
            row=13, column=1, sticky=W, ipadx=3, ipady=3
            )

    def renderDirectoryInputGUI(self, trainParamFrame, folderImg):

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
        self.vggModelPath.config(state=DISABLED)
        self.vggModelPath.grid(
            row=0, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.trainDataPath.config(state=DISABLED)
        self.trainDataPath.grid(
            row=1, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.trainLabelPath.config(state=DISABLED)
        self.trainLabelPath.grid(
            row=2, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.validationDataPath.config(state=DISABLED)
        self.validationDataPath.grid(
            row=3, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.testDataPath.config(state=DISABLED)
        self.testDataPath.grid(
            row=4, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.testResPath.config(state=DISABLED)
        self.testResPath.grid(
            row=5, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.learntModelPath.config(state=DISABLED)
        self.learntModelPath.grid(
            row=6, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
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
        self.inferModelPath.config(state=DISABLED)
        self.inferModelPath.grid(
            row=7, column=1, columnspan=2, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

    # ----- Event functions -----  
  
    def updateThresholdEntryStatus(self, saveModelStatus):

        if(saveModelStatus.get()):
            self.perfThresh.config(state=NORMAL)
        else:
            self.perfThresh.config(state=DISABLED)

    def setThisDirectory(self, Id):
        currentDirectory = filedialog.askdirectory()

        if len(currentDirectory) > 0:
            if Id == 0:
                self.vggModelPath.config(state=NORMAL)
                self.vggModelPath.delete(0, END)
                self.vggModelPath.insert(0, currentDirectory)
                self.vggModelPath.config(state=DISABLED)
            
            elif Id == 1:
                self.trainDataPath.config(state=NORMAL)
                self.trainDataPath.delete(0, END)
                self.trainDataPath.insert(0, currentDirectory)
                self.trainDataPath.config(state=DISABLED)

            elif Id == 2:
                self.trainLabelPath.config(state=NORMAL)
                self.trainLabelPath.delete(0, END)
                self.trainLabelPath.insert(0, currentDirectory)
                self.trainLabelPath.config(state=DISABLED)

            elif Id == 3:
                self.validationDataPath.config(state=NORMAL)
                self.validationDataPath.delete(0, END)
                self.validationDataPath.insert(0, currentDirectory)
                self.validationDataPath.config(state=DISABLED)

            elif Id == 4:
                self.testDataPath.config(state=NORMAL)
                self.testDataPath.delete(0, END)
                self.testDataPath.insert(0, currentDirectory)
                self.testDataPath.config(state=DISABLED)

            elif Id == 5:
                self.testResPath.config(state=NORMAL)
                self.testResPath.delete(0, END)
                self.testResPath.insert(0, currentDirectory)
                self.testResPath.config(state=DISABLED)

            elif Id == 6:
                self.learntModelPath.config(state=NORMAL)
                self.learntModelPath.delete(0, END)
                self.learntModelPath.insert(0, currentDirectory)
                self.learntModelPath.config(state=DISABLED)

            elif Id == 7:
                self.inferModelPath.config(state=NORMAL)
                self.inferModelPath.delete(0, END)
                self.inferModelPath.insert(0, currentDirectory)
                self.inferModelPath.config(state=DISABLED)
      
    def startTraingInNewThread(self, statusMsgBox):
        self.thread = threading.Thread(target = self.startTraining, args=(statusMsgBox,))
        self.thread.start()
        return
    
    def clearMessages(self, statusMsgBox):
        statusMsgBox.config(state=NORMAL)
        statusMsgBox.delete(1.0,END)
        statusMsgBox.config(state=DISABLED)

    def saveMessages(self, statusMsgBox):
        filename = filedialog.asksaveasfilename(defaultextension='.txt')
        f = open(filename, 'w')
        f.write(statusMsgBox.get('1.0', 'end'))
        f.close()
        messagebox.showinfo('FYI', 'Messages Saved')

    def validateForFloat(self, currentInput):
        print(currentInput)
        if currentInput.replace('.','',1).isdigit():
            return True
        elif currentInput is "":
            return True
        else:
            return False
    
    def validateForInt(self, currentInput):
        print(currentInput)
        if currentInput.isdigit():
            return True
        elif currentInput is "":
            return True
        else:
            return False

    def startTraining(self, statusMsgBox):

        statusMsgBox.config(state=NORMAL)
        statusMsgBox.insert(END,'Creating Tensorflow session\n')
        trainSession = tf.Session()
        
        statusMsgBox.insert(END,'TF training session created'
                            + '\n' + self.optAlgo.get()
                            + '\n' + str(float(self.initLearnRate.get()))
                            + '\n' + self.perfMetric.get()
                            + '\n' + str(self.batchSize.get())
                            + '\n' + str(self.saveModelStatus.get())
                            + '\n' + self.perfThresh.get()
                            + '\n' + str(self.showSegValImgsStatus.get())
                            + '\n'
                            )

        statusMsgBox.see(END)
        statusMsgBox.config(state=DISABLED)

        vggModelDir   = self.vggModelPath.get()
        trainDataDir  = self.trainDataPath.get()
        trainLabelDir = self.trainLabelPath.get()
        validationDir = self.validationDataPath.get()
        testDataDir   = self.testDataPath.get()
        testResultDir = self.testResPath.get()
        fcnModelDir   = self.learntModelPath.get()
        fcnInferDir   = self.inferModelPath.get()
        numOfClasses  = self.numOfClasses.get()

        ImgSize       = (160,576)
        initLearningRate    = float(self.initLearnRate.get())
        maxGradNorm         = float(self.maxNorm.get())
        if self.optAlgo.get() == 'Adam':
            optAlgo = 'adam'
        elif self.optAlgo.get() == 'Momentum':
            optAlgo = 'mntm'
        elif self.optAlgo.get() == 'Gradient Descent':
            optAlgo = 'grad'

        # Set training parameters
        batchSize           = self.batchSize.get()
        keepProb            = .5
        metric              = self.perfMetric.get()
        numOfEpochs         = int(self.numOfEpochs.get())
        saveModel           = self.saveModelStatus.get()
        perfThresh          = float(self.perfThresh.get())
        showSegValImages    = self.showSegValImgsStatus.get()

        statusMsgBox.config(state=NORMAL)
        statusMsgBox.insert(END,'Creating object for training\n')
        statusMsgBox.config(state=DISABLED)

        imageSegmenter = FullyConvNet(
            trainSession,
            vggModelDir,
            trainDataDir,
            trainLabelDir,
            validationDir,
            fcnModelDir,
            testDataDir,
            fcnInferDir, 
            numOfClasses
            )
        
        statusMsgBox.config(state=NORMAL)
        statusMsgBox.insert(END,'Object created\n')

        statusMsgBox.insert(END,'Setting optimizer parameters\n')
        statusMsgBox.config(state=DISABLED)

        imageSegmenter.setOptimizer(optAlgo, initLearningRate, ImgSize, maxGradNorm)
        
        statusMsgBox.config(state=NORMAL)
        statusMsgBox.insert(END,'Optimizer parameters set\n')

        statusMsgBox.insert(END,'Training in progress\n')
        statusMsgBox.config(state=DISABLED)

        imageSegmenter.trainFCN(
            batchSize,
            keepProb,
            metric,
            numOfEpochs,
            saveModel,
            perfThresh,
            showSegValImages
            )

        statusMsgBox.config(state=NORMAL)
        statusMsgBox.insert(END,'Training completed\n')
        statusMsgBox.config(state=DISABLED)

if __name__=='__main__':
    deepSemSeg_GUI_root = Tk()
    deepSemSeg_GUI(deepSemSeg_GUI_root)
    deepSemSeg_GUI_root.mainloop()