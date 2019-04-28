
import tensorflow as tf
import re
import scipy
import os
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt

class DataLoader:

    def __init__(self, folder_path, image_shape):
        self.data_folder = folder_path
        self.image_shape = image_shape

    def load_batches_from_disk(self,batch_size):
        #Create batches of training data
        #:param batch_size: Batch Size
        #:return: Batches of training data
        # Grab image and label paths
        image_paths = glob(os.path.join(self.data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(self.data_folder, 'gt_image_2', '*_road_*.png'))}

        background_color = np.array([255, 0, 0])

        # Shuffle training data
        random.shuffle(image_paths)
        # Loop through batches and grab images, yielding each batch

        #print(len(image_paths))
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                # Re-size to image_shape
                image = scipy.misc.imresize(scipy.misc.imread(image_file), self.image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), self.image_shape)

                # Create "one-hot-like" labels by class
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                
                images.append(image)
                gt_images.append(gt_image)
            yield np.array(images), np.array(gt_images)
    
    def load_batches_from_memory(self,batch_size):
        return

if __name__=="__main__":
    trainFolder = 'C:\\DataSets\\data_road\\training'
    batchSize = 10
    
    imageLoader = DataLoader(trainFolder,(160,576))


    print('Batch Size : ', batchSize)

    for epoch in range(20):
        print('Epoch : ',epoch+1)
        imageBatch = imageLoader.load_batches_from_disk(batchSize)
        batch=1
        for images in imageBatch:
            print('\tShowing Images From Batch : ', batch)
            i=1
            for image in images[0]:
                print('\t\tYou Are Seeing Image ',i)
                #plt.imshow(image)
                #plt.show()
                i+=1
            print('\tTotal Images Seen So Far : ', batchSize*(batch-1) + i-1 )
            print('\n')
            batch+=1


