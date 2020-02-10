from ImageSemanticSegmentor.FCN.FCN import FullyConvNet
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
import os

if __name__=="__main__":
    
    inferSession        = tf.Session()

    inferModelDir       =  os.getcwd()+'\\model\\FCN\\Infer'
    inferModelName      = 'IOU_0.8870986477691906'
    ImgSize             = (160,576) # Size(any) to which resize train images
    numOfClasses        = 2

    print('Creating object for inference')

    inferImgSegment  = FullyConvNet(inferSession, inferModelDir, inferModelName, ImgSize, 
                                    numOfClasses)

    print('Object created successfully')

    image_file          = 'C:\\DataSets\\data_road\\testing\\image_2\\um_000013.png'

    testImage           = scipy.misc.imresize(scipy.misc.imread(image_file), ImgSize)

    print('Inference in progress')
    segmentedTestImg    = inferImgSegment.segmentThisImage(testImage)
    plt.imshow(segmentedTestImg)
    plt.show()

    inferSession.close()


