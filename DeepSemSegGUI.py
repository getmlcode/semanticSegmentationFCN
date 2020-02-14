#from ImageSemanticSegmentor.FCN.FCN import FullyConvNet
from tkinter import *
from tkinter import ttk


class deepSemSeg_GUI:
    TestImage = []
    SegmentedImage = []
    def __init__(self, deepSemSeg_GUI_root):
        deepSemSeg_GUI_root.title('Deep Semantic Segmentation')
        deepSemSeg_GUI_root.minsize(500,500)

        # Create Tab Control
        tabControl          = ttk.Notebook(deepSemSeg_GUI_root)

        # Train Tab Design

        # Parameter input frame design
        trainTab            = ttk.Frame(tabControl)
        tabControl.add(trainTab, text='     Train     ')

        trainParamFrame     = Frame(trainTab, highlightbackground = "blue", 
                                width = 50, height = 50, highlightthickness = 1)
        trainParamFrame.pack(side = LEFT, fill = BOTH, expand = 1)

        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'VGG Model Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)

        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Train Data Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)

        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Train Label Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)

        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Validation Data Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)
        
        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Test Data Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)
        
        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Train Result Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)

        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Learnt Model Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)
        
        trainButton         = Button(trainParamFrame, bg = "cyan",  text = 'Inference Model Directory')
        trainButton.pack( side = TOP, fill = X, expand = 1)

        # Status messages frame design
        statusMsgFrame      = Frame(trainTab, highlightbackground = "red", 
                                width = 50, height = 50, highlightthickness = 1)
        statusMsgFrame.pack(side = RIGHT, fill = BOTH, expand = 1)
        
        statusMsgBox        = Text(statusMsgFrame, bg = "white", relief = SUNKEN)
        statusMsgBox.pack(side = TOP, fill = BOTH, expand = 1)

        clearMsgButton      = Button(statusMsgFrame, bg = "cyan", text = 'Clear Messages')
        clearMsgButton.pack( side = LEFT, fill = X, expand = 1)

        saveMsgButton       = Button(statusMsgFrame, bg = "cyan", text = 'Save Messages')
        saveMsgButton.pack( side = RIGHT, fill = X, expand = 1)

        # Inference Tab Design
        inferTab            = ttk.Frame(tabControl)
        tabControl.add(inferTab, text='     Infer     ')

        tabControl.pack(expand=1, fill="both")
        return

    def setVggModelDir(self):
        self.vggModelDir = filedialog.askdirectory()
        return
deepSemSeg_GUI_root = Tk()
deepSemSeg_GUI(deepSemSeg_GUI_root)
deepSemSeg_GUI_root.mainloop()