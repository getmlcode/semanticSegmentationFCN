import tensorflow as tf
import re
import scipy
import os
from glob import glob
import numpy as np
import random
from random import sample
import matplotlib.pyplot as plt
import itertools

class DataLoader:

    def __init__(self, folder_path, validationDir, testDataDir, image_shape):
        self.data_folder    = folder_path
        self.image_shape    = image_shape
        self.validationDir  = validationDir
        self.testDataDir    = testDataDir

    def createValidationSet(self, numOfFiles):
        #Creates directory if it doesn't exist

        #Exits if non-empty directory exists else 
        #removes numOfFiles training images and corresponding mask images from data_folder
        #and place them in validationDir. So validationDir will have both train and mask 
        #for validation images after this function call

        if not(os.path.isdir(self.validationDir)):
            os.mkdir(self.validationDir)
        elif os.path.isdir(self.validationDir) and len(os.listdir(self.validationDir)) != 0:
            print("\nValidation set already exists !")
            exit(0)

        image_paths = glob(os.path.join(self.data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(self.data_folder, 'gt_image_2', '*_road_*.png'))}
        
        valImgsPaths = sample(image_paths, numOfFiles)

        # Remove from trainDir and place in validationDir
        for imgPath in valImgsPaths:
            print(imgPath)
            print(os.path.join(self.validationDir, os.path.basename(imgPath)))
            os.rename(imgPath,os.path.join(self.validationDir, os.path.basename(imgPath)))
            print()

            print(label_paths[os.path.basename(imgPath)])
            print(os.path.join(self.validationDir, os.path.basename(label_paths[os.path.basename(imgPath)])))
            os.rename(label_paths[os.path.basename(imgPath)], 
                      os.path.join(self.validationDir, os.path.basename(label_paths[os.path.basename(imgPath)])))
            print()

    def load_validation_batches_from_disk(self, batch_size):
        #Create batches of validation data
        #param batch_size: Batch Size
        #return: Batches of validation data
        
        #Grab image and label paths
        all_paths = glob(os.path.join(self.validationDir, '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(self.validationDir, '*_road_*.png'))}

        image_paths = list( set(all_paths) - set( label_paths.values() ) )
        background_color = np.array([255, 0, 0])

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

                # Each element in gt_image is [background, road]
                # for background : [True, False] for Road : [False, True]
                gt_images.append(gt_image) 
            yield np.array(images), np.array(gt_images)

    def load_train_batches_from_disk(self, batch_size):
        #Create batches of training data
        #param batch_size: Batch Size
        #return: Batches of training data 

        #Grab image and label paths
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
                
                # Dimension batch_size X imageRow X imageCol X 3
                images.append(image) 

                # Each element in gt_image is [background, road]
                # for background : [True, False] for Road : [False, True]
                # Dimension batch_size X imageRow X imageCol X 2
                gt_images.append(gt_image) 

            yield np.array(images), np.array(gt_images)


    def load_test_batches_from_disk(self, batch_size):
        #Create batches of training data
        #param batch_size: Batch Size
        #return: Batches of training data 

        #Grab image and label paths
        image_paths = glob(os.path.join(self.testDataDir, 'image_2', '*.png'))

        # Loop through batches and grab images, yielding each batch

        #print(len(image_paths))
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                # Re-size to image_shape
                image = scipy.misc.imresize(scipy.misc.imread(image_file), self.image_shape)
                
                # Dimension batch_size X imageRow X imageCol X 3
                images.append(image) 
            yield np.array(images)


if __name__=="__main__":
    trainFolder = 'C:\\DataSets\\data_road\\training'
    validationDir = 'C:\\DataSets\\data_road\\validation'
    testDataDir       = 'C:\\DataSets\\data_road\\testing'

    batchSize = 10
    imageShape = (80,288)
    
    imageLoader = DataLoader(trainFolder, validationDir, imageShape)

    #imageLoader.createValidationSet(55)


    print('Batch Size : ', batchSize)
    
    for epoch in range(10):
        imageBatch = imageLoader.load_validation_batches_from_disk(batchSize)
        print('Epoch : ',epoch+1)
        #imageBatch = imageLoader.load_validation_batches_from_disk(batchSize)
        batch=1
        for images,labels in imageBatch:
            print('\tShowing Images From Batch : ', batch)
            print('\tNumber Of Images : ', images.shape[0])
            print('\t',images.shape,labels.shape)
            i=1
            for image,label in zip(images,labels):
                print('\n\t\tYou Are Seeing Image {} with label :'.format(i))
                print(label)
                plt.imshow(image)
                plt.show()
                i+=1
            print('\tTotal Images Seen So Far : ', batchSize*(batch-1) + i-1 )
            print('\n')
            batch+=1