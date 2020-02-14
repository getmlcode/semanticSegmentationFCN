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
        tabControl = ttk.Notebook(deepSemSeg_GUI_root)

        trainTab = ttk.Frame(tabControl)
        inferTab = ttk.Frame(tabControl)

        Button(trainTab, bg = "cyan",  text = 'Start Training').\
            pack(side = BOTTOM, fill = X)

        tabControl.add(trainTab, text='     Train     ')
        tabControl.add(inferTab, text='     Infer     ')

        tabControl.pack(expand=1, fill="both")
        return

deepSemSeg_GUI_root = Tk()
deepSemSeg_GUI(deepSemSeg_GUI_root)
deepSemSeg_GUI_root.mainloop()