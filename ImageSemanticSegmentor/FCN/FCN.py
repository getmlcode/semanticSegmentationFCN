import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import glob
import os
from preprocessor.fcnPreprocess import DataLoader
import semSegMetric

class FullyConvNet:
    def __init__(self, sess, *argv):
        self.sess = sess
        self.nArgs = len(argv)
        try:
            if self.nArgs == 8:
                self.vggModelDir        = argv[0]
                self.trainDataDir       = argv[1]
                self.trainLabelDir      = argv[2]
                self.validationDir      = argv[3]
                self.fcnModelDir        = argv[4]
                self.testDataDir        = argv[5]
                self.fcnInferDir        = argv[6]
                self.numClasses         = argv[7]

                # load/restore Vgg model from vggModelDir and retrieve layers needed for FCN
                #self.vggModel = 
                print('\nLoading VGG model and retrieving layers from : \n{}'.format(self.vggModelDir))
                tf.saved_model.loader.load(self.sess, ['vgg16'], self.vggModelDir)

                self.graph            = tf.get_default_graph()
                self.image_input      = self.graph.get_tensor_by_name('image_input:0')
                self.keep_prob        = self.graph.get_tensor_by_name('keep_prob:0')
                self.vggLayer3_Out    = self.graph.get_tensor_by_name('layer3_out:0')
                self.vggLayer4_Out    = self.graph.get_tensor_by_name('layer4_out:0')
                self.vggLayer7_Out    = self.graph.get_tensor_by_name('layer7_out:0')

                # Add FCN layers and skip connections
                print('\nAdding FCN layers and skip connections')
                self.fcn8 = tf.layers.conv2d(self.vggLayer7_Out,
                                             filters=self.numClasses,
                                             kernel_size=1,
                                             name="fcn8")

                self.fcn9 = tf.layers.conv2d_transpose(self.fcn8,
                                                       filters=self.vggLayer4_Out.get_shape().as_list()[-1],
                                                       kernel_size=4,
                                                       strides=(2, 2),
                                                       padding='SAME',
                                                       name="fcn9")

                self.fcn9_skip_connected = tf.add(self.fcn9,
                                                  self.vggLayer4_Out,
                                                  name="fcn9_plus_vgg_layer4")
        
                self.fcn10 = tf.layers.conv2d_transpose(self.fcn9_skip_connected,
                                                        filters=self.vggLayer3_Out.get_shape().as_list()[-1],
                                                        kernel_size=4,
                                                        strides=(2, 2),
                                                        padding='SAME',
                                                        name="fcn10_conv2d")

                self.fcn10_skip_connected = tf.add(self.fcn10,
                                                   self.vggLayer3_Out,
                                                   name="fcn10_plus_vgg_layer3")

                self.fcn11 = tf.layers.conv2d_transpose(self.fcn10_skip_connected,
                                                        filters=self.numClasses,
                                                        kernel_size=16,
                                                        strides=(8, 8),
                                                        padding='SAME',
                                                        name="fcn11")
            elif self.nArgs == 4:
                # Code for creating object for inference

                self.inferenceModelDir  = argv[0]
                self.modelName          = argv[1]
                self.imageShape         = argv[2]
                self.numClasses         = argv[3]

                netLoader = tf.train.import_meta_graph(os.path.join(self.inferenceModelDir, 
                                                                    self.modelName +'.meta'))
                self.graph = tf.get_default_graph()
                self.image_input = self.graph.get_tensor_by_name('image_input:0')
                self.logits = self.graph.get_tensor_by_name('fcn_logits:0')
                self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')

                #2nd argument should be everything before .data
                netLoader.restore(sess,os.path.join(self.inferenceModelDir, 
                                                    self.modelName))
            else:
                print('Invalid Number of Arguments')
                raise ValueError('Invalid Number of Arguments')
        except(ValueError):
            exit('Failed To Construct Object')

    def getOptimizer(self, loss_op, optAlgo, learningRate):

        if optAlgo == 'adam':
            return tf.train.AdamOptimizer(learningRate)
        elif optAlgo == 'grad':
            return tf.train.GradientDescentOptimizer(learningRate)
        elif optAlgo == 'mntm':
            return tf.train.MomentumOptimizer(learningRate)

    def setOptimizer(self, *argv):
        try:
            if self.nArgs == 8:
                self.optAlgo          = argv[0]
                self.initLearningRate = argv[1]
                self.imageShape       = argv[2]
                self.maxGradNorm      = argv[3]

                self.correct_label = tf.placeholder(tf.float32, 
                                                    [None,
                                                     self.imageShape[0],
                                                     self.imageShape[1],
                                                     self.numClasses])

                # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
                self.logits = tf.reshape(self.fcn11,[-1, self.numClasses], name="fcn_logits")
                self.correct_label_reshaped = tf.reshape(self.correct_label, [-1, self.numClasses])

                # Applies softmax for each pixel prediction to get probability distribution over classes
                # for that label and then cross entropy with its one-hot-label

                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.correct_label_reshaped[:])
                # Calculate mean cross entropy loss
                self.loss_op = tf.reduce_mean(self.cross_entropy, name="fcn_loss")
                
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.initLearningRate, global_step, 10000, 0.96)
                
                # return optimizer based on optAlgo argument passed
                self.optFunc = self.getOptimizer(self.loss_op, self.optAlgo, learning_rate)
                
                # Refer below links to understand above three lines (hoping they stay alive when you refer it) 
                # https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
                # https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow
                # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay

                # Use gradient clipping to prevent exploding gradient problem
                # maxGradNorm is the maximum norm for gradient vector
                trainVars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, trainVars), self.maxGradNorm)
                self.optimizer = self.optFunc.apply_gradients(zip(grads, trainVars), global_step = global_step, 
                                                              name="fcn_train_op")
            else:
                raise ValueError('Object Not Meant For Training')
        except(ValueError):
            exit('Sorry, This Object Was Not Created For Training Purpose !!')

    def trainFCN(self, batchSize, keep_prob_value, metric, numOfEpochs, saveModel, perfThresh, 
                 showSegImgs):
        self.batchSize              = batchSize
        self.keep_prob_value        = keep_prob_value
        self.metric                 = metric
        self.numOfEpochs            = numOfEpochs
        self.saveModel              = saveModel
        self.showSegValImages       = showSegImgs

        # Initialize all variables
        print('Initializing TF Variables')
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        print('Initialization Completed')

        self.imageLoader = DataLoader(self.trainDataDir, self.trainLabelDir, 
                                      self.validationDir, self.testDataDir, 
                                      self.imageShape)
        if self.saveModel == 1:
            modelSaver = tf.train.Saver()

        # Training Epochs
        for epoch in range(self.numOfEpochs):

            # Loads and reshapes images in batches from disk rather than memory
            trainImageBatch = self.imageLoader.load_train_batches_from_disk(self.batchSize)

            print('\n\t\t---FCN Training In Progress---')
            print('Batch Size : ', self.batchSize)
            
            print("EPOCH {}/{} In Progress...".format(epoch + 1, self.numOfEpochs))
            totalLoss = 0
            batch = 1

            # Iterate through train image batches
            for images,labels in trainImageBatch:
                print('\tBackpropagating On Batch : ', batch)
                loss, _, logits = self.sess.run([self.loss_op, self.optimizer,self.logits],
                                                feed_dict={self.image_input: images,
                                                           self.correct_label: labels,
                                                           self.keep_prob: self.keep_prob_value})
                totalLoss += loss
                batch+=1

                print("\t  Loss = ", loss)
                print("\t  Total Loss = ", totalLoss)

                del logits
                del loss

            print('\t    Validation For Epoch {} In Progress..'.format(epoch+1))
            
            validationPerformance, numOfValidationImages = self.validateFCN(self.showSegValImages)

            print("\n\t    Total Loss For Epoch {} = {}".format(epoch+1, totalLoss))
            print('\n\t    Total Number Of Validation Images = ', numOfValidationImages)
            print('\t    {} On Validation Set In Epoch {} = {}'.format(self.metric, epoch+1,
                                                                       validationPerformance))

            if self.saveModel:
                if ((epoch+1)%5 == 0 or validationPerformance >= perfThresh):
                    # Show segmented validation images
                    validationImageBatch = self.imageLoader.load_validation_batches_from_disk(self.batchSize)
                    for validationImages, validationLabels in validationImageBatch:
                        # Get prediction logit values and prediction labels
                        _, predictionLogits = self.getPredictionLabelsAndLogits(validationImages)
                        imageLogits = predictionLogits.reshape(-1, self.imageShape[0], 
                                                               self.imageShape[1], 
                                                               self.numClasses)
                        del predictionLogits
                        # Display segmented results
                        self.showSegmentedValidationImages(validationImages, imageLogits)
                        del imageLogits, validationImages

                    del validationImageBatch

                    # Save current model
                    print('\n\tSaving Current Model to ', self.fcnModelDir)
                    modelFullName = self.fcnModelDir + "\\FCN_" + self.metric + '_' + \
                        str(validationPerformance) + '_CrossEntropyLoss_' + str(totalLoss)
                    modelSaver.save(self.sess, modelFullName)

        return validationPerformance, totalLoss
    
    def validateFCN(self, showSegmentedValImages):
        # Validation
        # Read image batches from disk
        # Accumulate performance for whole validation set

        validationImageBatch    = self.imageLoader.load_validation_batches_from_disk(self.batchSize)
        totCorrectPredictions   = 0.0
        numOfValidationImages   = 0
        
        for validationImages, validationLabels in validationImageBatch:

            # Get prediction logit values and prediction labels
            predictionLabels, predictionLogits = self.getPredictionLabelsAndLogits(validationImages)

            validationBatchPerf = self.getPerformanceMetric(validationLabels.reshape(-1,
                                                                                     self.imageShape[0],
                                                                                     self.imageShape[1],
                                                                                     self.numClasses),
                                                            predictionLabels, self.metric)
        
            numOfValidationImages += validationImages.shape[0]
            totCorrectPredictions += validationBatchPerf * validationImages.shape[0]

            del predictionLabels

            if showSegmentedValImages:
                imageLogits = predictionLogits.reshape(-1, self.imageShape[0], self.imageShape[1], self.numClasses)
                del predictionLogits

                # Display segmented results
                self.showSegmentedValidationImages(validationImages, imageLogits)
                del imageLogits, validationImages
            else:
                del predictionLogits, validationImages
        
        validationPerf = totCorrectPredictions/numOfValidationImages
        del validationImageBatch

        return validationPerf, numOfValidationImages

    def showSegmentedValidationImages(self, validationImages, imageLogits):

        for image, imageLogit in zip(validationImages, imageLogits):
            segmentedImage = self.getSegmentedImage(image, imageLogit)
            plt.imshow(segmentedImage)
            plt.show()

    def getPredictionLabelsAndLogits(self, validationImages):

        predictionLogits = self.sess.run(self.logits, feed_dict={self.image_input: validationImages,
                                                                 self.keep_prob: 1.0})
        
        softMax = self.sess.run(tf.nn.softmax(predictionLogits.reshape(-1,
                                                                       self.imageShape[0],
                                                                       self.imageShape[1],
                                                                       self.numClasses), axis=3))
        
        predictionLabels = (softMax == softMax.max(axis=3, keepdims=1))

        del softMax

        return predictionLabels, predictionLogits
    
    def moveToInferenceDir(self, topN):
        # Move top topN performing models to inference directory

        wtFileNameList = glob.glob(self.fcnModelDir+'//*.data*')

        # If thee are less number of models then place all of them
        # into inference directory
        topN = len(wtFileNameList) if topN > len(wtFileNameList) else topN
        perfMetricList = []

        # Extract perfomance metric value
        for wtFileName in wtFileNameList:
            fileName = wtFileName.split('\\')[-1]
            modelPerfMetric = float(fileName.split('_')[2])

            perfMetricList.append(modelPerfMetric)

        # Resverse sort perfMetricList
        perfMetricList.sort(reverse=True)

        # Place top topN performing models in inference directory
        for rank in range(topN):
            perfMetricVal = perfMetricList[rank]

            # Get filenames of model with given performance metric value
            modelFileNames = glob.glob(self.fcnModelDir+'//*'+str(perfMetricVal)+'*')

            for fileName in modelFileNames:
                fileExtension = fileName.split('.')[-1]
                # Move current file to inference directory
                newFilePath = self.fcnInferDir + '//FCN_' + self.metric + '_' + str(perfMetricVal) + '.' + fileExtension
                os.replace(fileName, newFilePath)

    def getSegmentedImage(self, image, imageLogit):

        # Input
        #       image       :   single image
        #       imageLogit  :   logit values for image pixel elements
        # Output
        #       segmented image with overlay mask

        softMax = self.sess.run(tf.nn.softmax(imageLogit, axis = 2))
        prediction = (softMax == softMax.max(axis=2, keepdims=1))
        prediction = prediction[:, :, 1].reshape(self.imageShape[0], self.imageShape[1])
        segmentation = (prediction == True).reshape(self.imageShape[0], self.imageShape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        segmentedImage = scipy.misc.toimage(image)
        segmentedImage.paste(mask, box=None, mask=mask)

        return segmentedImage
    
    def segmentThisImage(self, image):

        # Same as getSegmentedImage, but only takes single image as input

        predictionLogits = self.sess.run(self.logits, feed_dict={self.image_input: np.expand_dims(image, axis=0), 
                                                                 self.keep_prob: 1.0})

        imageLogit = predictionLogits.reshape(-1, self.imageShape[0], self.imageShape[1], 
                                               self.numClasses)

        segmentedImage = self.getSegmentedImage(image, imageLogit[0])

        return segmentedImage

    def segmentDirectoryImages(self, imageDir):
        # Read images in given imageDir directoy and return an array of masked 
        # segmented images

        return

    def generateAndSaveSegmentedTestImages(self, testResultDir):

        testImageBatch = self.imageLoader.load_test_batches_from_disk(self.batchSize)

        name  = 'epoch'+str(self.numOfEpochs)+'_TestImage_'
        count = 0

        for testImages in testImageBatch:
            predictionLogits = self.sess.run(self.logits, feed_dict={self.image_input: testImages, 
                                                                     self.keep_prob: 1.0})
            imageLogits = predictionLogits.reshape(-1, self.imageShape[0], self.imageShape[1], 
                                                   self.numClasses)

            del predictionLogits
            for image, imageLogit in zip(testImages, imageLogits):
                count = count+1
                segmentedImage = self.getSegmentedImage(image, imageLogit)
                scipy.misc.imsave(os.path.join(testResultDir, name+str(count)+'.png'), segmentedImage)

    def getPerformanceMetric(self, validationLabels, predictionLabels, metric):
    
        if metric == 'IOU':
            return semSegMetric.IntersectionOverUnion(validationLabels, predictionLabels, self.numClasses)