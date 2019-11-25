import tensorflow as tf
import numpy as np
import os
from preprocess import DataLoader
import semSegMetric
import matplotlib.pyplot as plt


class FullyConvNet:
    def __init__(self, sess, *argv):
        self.sess = sess
        self.nArgs = len(argv)
        try:
            if self.nArgs == 5:
                self.vggModelDir      = argv[0]
                self.trainDataDir     = argv[1]
                self.fcnModelDir      = argv[2]
                self.validationDir    = argv[3]
                self.numClasses       = argv[4]

                #load/restore Vgg model from vggModelDir and retrieve layers needed for FCN
                #self.vggModel = 
                print('\nLoading VGG model and retrieving layers from : \n{}'.format(self.vggModelDir))
                tf.saved_model.loader.load(self.sess, ['vgg16'], self.vggModelDir)

                self.graph = tf.get_default_graph()
                self.image_input = self.graph.get_tensor_by_name('image_input:0')
                self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
                self.vggLayer3_Out = self.graph.get_tensor_by_name('layer3_out:0')
                self.vggLayer4_Out = self.graph.get_tensor_by_name('layer4_out:0')
                self.vggLayer7_Out = self.graph.get_tensor_by_name('layer7_out:0')

                #Add FCN layers and skip connections
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
            elif self.nArgs == 1:
                #Add code for restoring FCN model for inference
                print('TODO : Add FCN model restoration code')
                pass
            else:
                print('Invalid Number of Arguments')
                raise ValueError('Invalid Number of Arguments')
        except(ValueError):
            exit('Failed To Construct Object')

    def getOptimizer(self,loss_op, optAlgo, learningRate):

        if optAlgo == 'adam':
            return tf.train.AdamOptimizer(learningRate)
        elif optAlgo == 'grad':
            return tf.train.GradientDescentOptimizer(learningRate)
        elif optAlgo == 'mntm':
            return tf.train.MomentumOptimizer(learningRate)

    def setOptimizer(self,*argv):
        try:
            if self.nArgs == 5:
                self.optAlgo          = argv[0]
                self.initLearningRate = argv[1]
                self.imageShape       = argv[2]
                self.maxGradNorm      = argv[3]

                self.correct_label = tf.placeholder(tf.float32, [None, self.imageShape[0], 
                                                                 self.imageShape[1], self.numClasses])

                # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
                self.logits = tf.reshape(self.fcn11,[-1, self.numClasses], name="fcn_logits")
                self.correct_label_reshaped = tf.reshape(self.correct_label, [-1, self.numClasses])

                # Calculate cross entropy loss using actual labels
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.correct_label_reshaped[:])
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

    def trainFCN(self, batchSize, keep_prob_value, metric, numOfEpochs):
        self.batchSize           = batchSize
        self.keep_prob_value     = keep_prob_value
        self.metric              = metric
        self.numOfEpochs         = numOfEpochs

        # Initialize all variables
        print('Initializing TF Variables')
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        print('Initialization Completed')

        imageLoader = DataLoader(self.trainDataDir, self.validationDir, self.imageShape)

        #Training Epochs        
        for epoch in range(self.numOfEpochs):

            #Loads and reshapes images in batches from disk rather than memory
            trainImageBatch = imageLoader.load_train_batches_from_disk(self.batchSize)
            validationImageBatch = imageLoader.load_validation_batches_from_disk(self.batchSize)

            print('\n\t\t---FCN Training In Progress---')
            print('Batch Size : ', self.batchSize)
            
            print("EPOCH {}/{} In Progress...".format(epoch + 1, self.numOfEpochs))
            totalLoss = 0 
            batch = 1

            #Iterate through train image batches
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

                #softMax = self.sess.run(tf.nn.softmax(logits.reshape(-1,160,576,2),axis=3))
                #predictions =  (softMax == softMax.max(axis=3, keepdims=1)) #[True/False] array
                ##predictions =  (softMax == softMax.max(axis=3, keepdims=1)).astype(float)
                #correctClassification = np.sum((predictions == labels.reshape(-1,160,576,2)).all(axis=3))
                #print("Logits =\n", logits)
                #print("Logits Shape =\n", logits.shape)
                #print("SoftMax =\n", softMax)
                #print("SoftMax Shape =\n", softMax.shape)
                #print("Predictions =\n", predictions)
                #print("Predictions Shape =\n", predictions.shape)
                #print("Labels =\n", labels)
                #print("Labels Shape =\n", labels.shape)
                #print("Number Of Correct Classification =\n", correctClassification)

                del logits
                del loss

            print("\t    Total Loss For Epoch {} = {}".format(epoch+1, totalLoss))
            print()
            
            #Validation
            #Read image batches from disk
            #Accumulate performance for whole validation set

            totCorrectPredictions = 0.0
            numOfValidationImages = 0

            print('\t    Validation For Epoch {} In Progress..'.format(epoch+1))

            for validationImages,validationLabels in validationImageBatch:
                predictionLogits = self.sess.run(self.logits,
                                                 feed_dict={self.image_input: validationImages,
                                                            self.keep_prob: 1.0})

                softMax = self.sess.run(tf.nn.softmax(predictionLogits.reshape(-1,
                                                                               self.imageShape[0],
                                                                               self.imageShape[1],
                                                                               self.numClasses), axis=3))

                predictionLabels = (softMax == softMax.max(axis=3, keepdims=1))

                validationBatchPerf = self.validateModel(validationLabels.reshape(-1,
                                                                                  self.imageShape[0],
                                                                                  self.imageShape[1],
                                                                                  self.numClasses),
                                                         predictionLabels,
                                                         self.metric)

                numOfValidationImages += validationImages.shape[0]
                totCorrectPredictions += validationBatchPerf * validationImages.shape[0]

                print('|',end=' ')

            validationPerf = totCorrectPredictions/numOfValidationImages

            print('\n\t    Total Number Of Validation Images = ', numOfValidationImages)
            print('\t    {} On Validation Set In Epoch {} = {}'.format(metric, epoch+1, validationPerf))
    
    def validateModel(self, validationLabels, predictionLabels, metric):

        if metric == 'IOU':
            return semSegMetric.IntersectionOverUnion(validationLabels, predictionLabels)

    def getSegmentedImage(self, inputImage):
        print('\nLoading Model From : ', self.fcnModelDir)

        return


if __name__=="__main__":

    sess = tf.Session()

    # set directories
    modelDir          = os.getcwd()+'\\model\\vgg'
    trainDir          = 'C:\\DataSets\\data_road\\training'
    validationDir     = 'C:\\DataSets\\data_road\\validation'
    fcnModelDir       = os.getcwd()+'\\model\\FCN'
    numOfClasses      = 2

    # Set optimzer
    optAlgo           = 'adam'
    initLearningRate  = .001
    ImgSize           = (160,576) # Size(any) to which resize train images
    maxGradNorm       = .1


    # Set training parameters
    batchSize         = 32
    keepProb          = .5
    metric            = 'IOU'
    numOfEpochs       = 5


    print('Creating object for training')
    fcnImageSegmenter = FullyConvNet(sess, modelDir, trainDir, 
                                     fcnModelDir, validationDir, numOfClasses)
    print('Object created successfully')

    fcnImageSegmenter.setOptimizer(optAlgo, initLearningRate, ImgSize,
                                  maxGradNorm)

    fcnImageSegmenter.trainFCN(batchSize, keepProb, metric, 
                               numOfEpochs)