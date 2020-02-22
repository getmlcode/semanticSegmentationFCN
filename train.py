from ImageSemanticSegmentor.FCN.FCN import FullyConvNet
import tensorflow as tf
import os

if __name__ == "__main__":

    trainSession = tf.Session()
    
    # set directories
    vggModelDir         = os.getcwd()+'\\model\\vgg'
    trainDataDir        = 'C:\\DataSets\\data_road\\training\\image_2'
    trainLabelDir       = 'C:\\DataSets\\data_road\\training\\gt_image_2'
    validationDir       = 'C:\\DataSets\\data_road\\validation'
    testDataDir         = 'C:\\DataSets\\data_road\\testing\\image_2'
    testResultDir       = 'C:\\DataSets\\data_road\\testing\\testResults'
    fcnModelDir         = os.getcwd()+'\\model\\FCN'
    fcnInferDir         = os.getcwd()+'\\model\\FCN\\Infer'
    numOfClasses        = 2

    ImgSize             = (160,576) # Size(any) to which resize train images
    
    # Set optimzer
    optAlgo             = 'adam'
    initLearningRate    = .001
    maxGradNorm         = .1
    
    # Set training parameters
    batchSize           = 32
    keepProb            = .5
    metric              = 'IOU'
    numOfEpochs         = 1
    saveModel           = 0
    perfThresh          = 0.8
    showSegValImages    = 0 # 0 means Don't show segmented validaiton images after each epoch

    # Number Of learnt models to move to inference directory
    topN                = 7
    
    print('Creating object for training')

    imageSegmenter = FullyConvNet(trainSession, vggModelDir, trainDataDir, trainLabelDir, 
                                  validationDir, fcnModelDir, testDataDir, 
                                  fcnInferDir, numOfClasses)

    print('Object created successfully')
    
    imageSegmenter.setOptimizer(optAlgo, initLearningRate, ImgSize, maxGradNorm)
    
    imageSegmenter.trainFCN(batchSize, keepProb, metric, numOfEpochs, saveModel, 
                            perfThresh, showSegValImages)
    
    print('Segmenting and saving test images')
    imageSegmenter.generateAndSaveSegmentedTestImages(testResultDir)
    
    print('Moving Top {} models to Infer Directory'.format(topN))
    imageSegmenter.moveToInferenceDir(topN)

    print('closing session')    
    trainSession.close()