import numpy as np
def IntersectionOverUnion(validationLabels, predictionLabels):

    totalNumOfPixels = validationLabels.shape[0]*validationLabels.shape[1]*validationLabels.shape[2]
    correctClassifications = np.sum( (predictionLabels == validationLabels).all(axis=3) )

    meanIOU = correctClassifications/totalNumOfPixels
    
    return meanIOU