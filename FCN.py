import tensorflow as tf
import numpy as np


class FullyConvNet:
    def __init__(self,sess,vggModelDir,trainDataDir,fcnModelDir,numClasses):
        self.vggModelDir = vggModelDir
        self.trainDataDir = trainDataDir
        self.fcnModelDir = fcnModelDir
        self.sess = sess
        self.numClasses = numClasses
        #load/restore Vgg model from vggModelDir and retrieve layers needed for FCN
        print('Loading VGG model and retrieving layers')
        self.vggModel = tf.saved_model.loader.load(self.sess, ['vgg16'], vggModelDir)
        self.graph = tf.get_default_graph()
        self.image_input = self.graph .get_tensor_by_name('image_input:0')
        self.keep_prob = self.graph .get_tensor_by_name('keep_prob:0')
        self.vggLayer3_Out = self.graph .get_tensor_by_name('layer3_out:0')
        self.vggLayer4_Out = self.graph .get_tensor_by_name('layer4_out:0')
        self.vggLayer7_Out = self.graph .get_tensor_by_name('layer7_out:0')

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

    def trainFCN(self):
        return

    def getSegmentedImage(self,inputImage):
        return

    def evaluateModel(self,predictedLabel,trueLabel):
        return


if __name__=="__main__":

    sess = tf.Session()
    modelDir='D:\\Acads\\IISc ME\\Projects\\SemanticSegmentation\\fullyConvolutionalNet\\model\\vgg'
    trainDir='C:\\DataSets'
    fcModelDir='D:\\Acads\\IISc ME\\Projects\\SemanticSegmentation\\fullyConvolutionalNet\\model\\FCN'
    fcnImageSegmenter = FullyConvNet(sess,modelDir,trainDir,fcModelDir,3)