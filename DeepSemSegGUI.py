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
            text="select",
            compound=LEFT,
            command=self.setVggModelDir,
        )
        vggModelDirButton.image = folderImg
        vggModelDirButton.grid(row=0, pady=2, padx=2, sticky=E)

        self.vggModelPath = Entry(trainParamFrame, bd=4, width=20)
        self.vggModelPath.insert(0, "Vgg Model Directory")
        self.vggModelPath.grid(
            row=0, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        trainDataDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        trainDataDirButton.image = folderImg
        trainDataDirButton.grid(row=1, pady=1, padx=2, sticky=E)

        self.trainDataPath = Entry(trainParamFrame, bd=4, width=20)
        self.trainDataPath.insert(0, "Train Data Directory")
        self.trainDataPath.grid(
            row=1, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        trainLabelDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        trainLabelDirButton.image = folderImg
        trainLabelDirButton.grid(row=2, pady=1, padx=2, sticky=E)

        self.trainLabelPath = Entry(trainParamFrame, bd=4, width=20)
        self.trainLabelPath.insert(0, "Train Label Directory")
        self.trainLabelPath.grid(
            row=2, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        validationDataDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        validationDataDirButton.image = folderImg
        validationDataDirButton.grid(row=3, pady=1, padx=2, sticky=E)

        self.validationDataPath = Entry(trainParamFrame, bd=4, width=20)
        self.validationDataPath.insert(0, "Validation Data Directory")
        self.validationDataPath.grid(
            row=3, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        testDataDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        testDataDirButton.image = folderImg
        testDataDirButton.grid(row=4, pady=1, padx=2, sticky=E)

        self.testDataPath = Entry(trainParamFrame, bd=4, width=20)
        self.testDataPath.insert(0, "Test Data Directory")
        self.testDataPath.grid(
            row=4, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        testResDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        testResDirButton.image = folderImg
        testResDirButton.grid(row=5, pady=1, padx=2, sticky=E)

        self.testResPath = Entry(trainParamFrame, bd=4, width=20)
        self.testResPath.insert(0, "Test Result Directory")
        self.testResPath.grid(
            row=5, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        learntModelDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        learntModelDirButton.image = folderImg
        learntModelDirButton.grid(row=6, pady=1, padx=2, sticky=E)

        self.learntModelPath = Entry(trainParamFrame, bd=4, width=20)
        self.learntModelPath.insert(0, "Learnt Model Directory")
        self.learntModelPath.grid(
            row=6, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

        inferModelDirButton = Button(
            trainParamFrame, image=folderImg, text="select", compound=LEFT
        )
        inferModelDirButton.image = folderImg
        inferModelDirButton.grid(row=7, pady=1, padx=2, sticky=E)

        self.inferModelPath = Entry(trainParamFrame, bd=4, width=20)
        self.inferModelPath.insert(0, "Inference Model Directory")
        self.inferModelPath.grid(
            row=7, column=1, columnspan=8, pady=2, padx=2, sticky=NSEW, ipadx=3, ipady=3
        )

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

    def setVggModelDir(self):
        vggModelDir = filedialog.askdirectory()

        if len(vggModelDir) > 0:
            self.vggModelPath.delete(0, END)
            self.vggModelPath.insert(1, vggModelDir)
        return


deepSemSeg_GUI_root = Tk()
deepSemSeg_GUI(deepSemSeg_GUI_root)
deepSemSeg_GUI_root.mainloop()