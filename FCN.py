import tensorflow as tf
import numpy as np
import os
import preprocess


class FullyConvNet:
    def __init__(self,sess,*argv):
        self.sess = sess
        self.nArgs = len(argv) 
        try:
            if self.nArgs == 6:
                self.vggModelDir      = argv[0]
                self.trainDataDir     = argv[1]
                self.fcnModelDir      = argv[2]
                self.numClasses       = argv[3]
                self.optAlgo          = argv[4]
                self.initLearningRate = argv[5]
                #load/restore Vgg model from vggModelDir and retrieve layers needed for FCN
                print('Loading VGG model and retrieving layers from : \n{}'.format(self.vggModelDir))
                self.vggModel = tf.saved_model.loader.load(self.sess, ['vgg16'], self.vggModelDir)
                self.graph = tf.get_default_graph()
                self.image_input = self.graph.get_tensor_by_name('image_input:0')
                self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
                self.vggLayer3_Out = self.graph.get_tensor_by_name('layer3_out:0')
                self.vggLayer4_Out = self.graph.get_tensor_by_name('layer4_out:0')
                self.vggLayer7_Out = self.graph.get_tensor_by_name('layer7_out:0')

                #Add FCN layers and skip connections
                print('Adding FCN layers and skip connections')
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
                raise ValueError('Invalid Number of Arguments')
        except(ValueError):
            exit('Failed To Construct Object')

    def getOptimizer(loss_op,optAlgo,rate,global_step):

        if optAlgo == 'adam':
            return tf.train.AdamOptimizer(rate).minimize(loss_op, global_step=global_step, name="fcn_train_op")
        elif optAlgo == 'grad':
            return tf.train.GradientDescentOptimizer(rate).minimize(loss_op, global_step=global_step, name="fcn_train_op")
        elif optAlgo == 'mntm':
            return tf.train.MomentumOptimizer(rate).minimize(loss_op, global_step=global_step, name="fcn_train_op")

    def trainFCN(self):
        try:
            if self.nArgs == 4:

                self.correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], self.numClasses])

                # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
                self.logits = tf.reshape(self.fcn11,(-1, self.numClasses),name="fcn_logits")
                self.correct_label_reshaped = tf.reshape(self.correct_label,(-1, self.numClasses))

                # Calculate distance from actual labels using cross entropy
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.correct_label_reshaped[:])

                self.loss_op = tf.reduce_mean(self.cross_entropy,name="fcn_loss")
                
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.initLearningRate, global_step,100000, 0.96, staircase=True)
                self.optimizer = self.getOptimizer(self.loss_op,self.optAlgo,learning_rate,global_step)
                
                # Refer below links to understand above three lines (hoping they stay alive when you refer it) 
                # https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
                # https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow
                # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay



            else:
                raise ValueError('Object Not Meant For Training')
        except(ValueError):
            exit('Sorry, This Object Was Not Created For Training Purpose !!')

    def getSegmentedImage(self,inputImage):
        return

    def evaluateModel(self,predictedLabel,trueLabel):
        return


if __name__=="__main__":

    sess = tf.Session()
    modelDir=os.getcwd()+'\\model\\vgg'
    trainDir='C:\\DataSets\\data_road\training'
    fcnModelDir=os.getcwd()+'\\model\\FCN'

    print('Creating object for training')
    fcnImageSegmenter = FullyConvNet(sess,modelDir,trainDir,fcnModelDir,3,'adam',.1)
    print('Object created successfully')

    #fcnImageSegmenter.trainFCN()