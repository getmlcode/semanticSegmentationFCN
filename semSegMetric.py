import numpy as np
def IntersectionOverUnion(validationLabels, predictionLabels, numOfClasses):

    # returns mean-IOU for batch

    # range(2) = [0,1]
    
    meanIOUForImage = []
    for validationLabel, predictionLabel in zip(validationLabels, predictionLabels):
        labelIOUForImage = []
        # Calculate IOU for each label for current image
        for label in range(numOfClasses):
            # Create one-hot-label for current class label, e.g [1, 0] for class 0
            oneHotLabel = np.eye(numOfClasses)[label]

            intersection = (validationLabel==oneHotLabel).all(axis=2)*(predictionLabel==oneHotLabel).all(axis=2)
            union = (validationLabel==oneHotLabel).all(axis=2)+(predictionLabel==oneHotLabel).all(axis=2)

            intersectionSum = np.sum(intersection)
            unionSum = np.sum(union)
            # List of IOUs for labels in current image
            if unionSum != 0:
                labelIOUForImage.append(intersectionSum/unionSum)

        # List of means IOUs of labels in each image
        meanIOUForImage.append(np.mean(labelIOUForImage))

    # Mean IOU for batch
    meanIOUForBatch = np.mean(meanIOUForImage)

    return meanIOUForBatch

if __name__=="__main__":
    val = np.array([
        [
            [[1, 0],[0, 1],[0, 1]],
            [[1, 0],[0, 1],[0, 1]],
            [[1, 0],[0, 1],[0, 1]]
            ],

         [
             [[1, 0],[0, 1],[0, 1]],
             [[1, 0],[0, 1],[0, 1]],
             [[1, 0],[0, 1],[0, 1]]
             ]
        ])

    pred = np.array([
        [
            [[1, 0],[0, 1],[0, 1]],
            [[0, 1],[1, 0],[0, 1]],
            [[1, 0],[0, 1],[1, 0]]
            ],

         [
             [[0, 1],[0, 1],[1, 0]],
             [[1, 0],[0, 1],[0, 1]],
             [[0, 1],[0, 1],[1, 0]]
             ]
        ])

    meanIOU = IntersectionOverUnion(val, pred, 2)
    print('Mean IOU = ',meanIOU)

