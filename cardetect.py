'''
cardetect.py

This script takes in a set of images or a video
file and will draw bounding boxes to track the
vehicles in the frames.

Bryant Pong
10/7/17
'''

import cv2
import glob
import sys
import numpy as np
import pickle
import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Constants

# Folder to training images:
TRAINING_DATA_FOLDER = "./training_data/"
VEHICLE_IMAGES_FOLDER = TRAINING_DATA_FOLDER + "vehicles/"
NONVEHICLE_IMAGES_FOLDER = TRAINING_DATA_FOLDER + "non-vehicles/"

TEST_IMAGES_FOLDER = "./test_images/"
OUTPUT_IMAGES_FOLDER = "./output_images/"

# Globs to get all PNG files in the training data folders:
vehicleImages = glob.glob( VEHICLE_IMAGES_FOLDER + "/**/*.png" )
nonVehicleImages = glob.glob( NONVEHICLE_IMAGES_FOLDER + "/**/*.png" )

# Set this flag to True to enable debug images:
displayImages = True

'''
Helper Function to display a RGB image: 

Set the second flag to True to convert
the image from BGR to RGB. 
'''
def DisplayImage( img, convert=False ):

    if not displayImages:
        return

    if convert:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )

    plt.imshow( img )
    plt.show()

'''
Helper function to display a grayscale image:
'''
def DisplayGrayImage( img ):

    if not displayImages:
        return
    
    plt.imshow( img, cmap='gray' )
    plt.show()

'''
Helper function to draw a series of bounding boxes on the provided
image.

"img" is expected to be in RGB format.

The boxes are in the format [ ( ( x1, y1 ), ( x2, y2 ) ), ... ]

The return value of this function is a copy of img with the bounding
boxes drawn on it.
'''
def DrawBoundingBoxes( img, boxes, color = ( 0, 0, 255 ), thickness = 6 ):
    imgCopy = np.copy( img )

    for box in boxes:
        cv2.rectangle( img, box[ 0 ], box[ 1 ], color, thickness )

    return imgCopy

'''
Helper function to compute a color histogram feature vector
for an input RGB image.

Comparing histograms will allow you to filter out objects
with different colors, but will be the same for images with
the same amount of colors.  For example, both a red car
and a red billboard will have a similar color histogram, but
a red car and a blue object will have a different color histogram.

TODO: May want to look at different color spaces (HLS)  
'''
def ComputeColorHistogram( img, nbins = 32, binsRange = ( 0, 256 ) ):

    # Compute the individual RGB histograms for the image:
    redHist = np.histogram( img[ :, :, 0 ], bins = nbins, range = binsRange )
    greenHist = np.histogram( img[ :, :, 1 ], bins = nbins, range = binsRange )
    blueHist = np.histogram( img[ :, :, 2 ], bins = nbins, range = binsRange )

    # All the histograms have the same number of bins:
    binEdges = redHist[ 1 ]

    # Compute the bin centers:
    binCenters = ( binEdges[ 1: ] + binEdges[ 0 : len( binEdges ) - 1 ] ) / 2

    # Concatenate the histograms to form the feature vector:
    histFeatures = np.concatenate( redHist[ 0 ], greenHist[ 0 ], blueHist[ 0 ] )

    return redHist, greenHist, blueHist, binCenters, histFeatures
    
'''
Helper function to resize a given image and return
a 1-Dimensional spatial histogram. 
'''
def ComputeResizedSpatialHistogram( img, size = (32, 32 ) ):

    features = cv2.resize( img, size ).ravel()

    return features

'''
This function computes a histogram of oriented gradients (HOG)
feature vector.  You can also specify whether you want to visualize
the HOG or not.
'''
def GetHOGFeatures( img, orient, pixelsPerCell, cellsPerBlock, vis=False, featureVec=True ):

    # Get the features and a visualiation (if desired):
    features, hogImage = hog( img, orientations = orient, pixels_per_cell = ( pixelsPerCell, pixelsPerCell ), cells_per_block = ( cell_per_block, cell_per_block ), visualize = vis, feature_vector = featureVec )

    if vis:
        return features, hogImage
    else:
        return features

'''
Given a list of feature vectors, return an array
stack of the normalized feature vectors.
'''
def NormalizeFeatureVectors( featureList ):
    
    # Create an array stack (StandardScaler needs np.float64):
    featureStack = np.vstack( featureList ).astype( np.float64 )

    # Fit a per-column scaler:
    featureScaler = StandardScaler().fit( featureStack )

    # Apply the scaler to the featureStack:
    scaledFeatures = featureScaler.transform( featureStack )

    return scaledFeatures

'''
This function takes in a list of strings of filenames from the
vehicles and non-vehicles datasets.  A dictionary is constructed
to give the following information:

1) Number of vehicle images     "nCars"
2) Number of non-vehicle images "nNotCars" 
3) Shape of a sample image      "imageShape"
4) Type of a sample image       "dataType"
'''
def GetDatasetStats( carImageList, notCarImageList ):
    dataDict = {}

    # Get the number of car images:
    dataDict[ "nCars" ] = len( carImageList )

    # Get the number of non-car images:
    dataDict[ "nNotCars" ] = len( notCarImageList )

    # Take a sample image and get the shape and type:
    sampleImage = mpimg.imread( carImageList[ 0 ] )

    dataDict[ "imageShape" ] = sampleImage.shape
    dataDict[ "dataType" ] = sampleImage.dtype

    return dataDict

'''
This function takes in a list of images, then
performs the following steps:

1) Open the image in RGB format
2) Convert from RGB to specified colorspace
3) 
'''
def GetFeatureVectors( imgList, colorspace="RGB" ):

    # Feature vector list:
    featureVec = []

    # Step 1: Open the image in RGB format:
    for fileName in imgList:
        nextImage = mpimg.imread( fileName )

        # Step 2: Convert the image to the desired colorspace:
        if colorspace != "RGB":

            if colorspace == "HSV":
                featureImg = cv2.cvtColor( nextImage, cv2.COLOR_RGB2HSV )
            elif colorspace == "LUV":
                featureImg = cv2.cvtColor( nextImage, cv2.COLOR_RGB2LUV )
            elif colorspace == "HLS":
                featureImg = cv2.cvtColor( nextImage, cv2.COLOR_RGB2HLS )
            elif colorspace == "YUV":
                featureImg = cv2.cvtColor( nextImage, cv2.COLOR_RGB2YUV )
            elif colorspace == "YCrCb":
                featureImg = cv2.cvtColor( nextImage, cv2.COLOR_RGB2YCrCb )
        else:
            featureImg = np.copy( nextImage )

        # Extract Spatial Features:
        spatialFeatures = ComputeResizedSpatialHistogram( featureImg, size = ( 32, 32 ) )

        featureVec.append( spatialFeatures )

    return featureVec


'''
This function is used to train the Support Vector Machine.

This function will save the trained data in a pickle file. 
'''
def TrainClassifier():

    '''
    TODO: We can combine features (say HSV + HOG) into a
    single feature vector, but need to normalize the data. 
    '''
    
    # Print dataset statistics:
    datasetStats = GetDatasetStats( vehicleImages, nonVehicleImages )
    print( "Number of vehicle images: " + str( datasetStats[ "nCars" ] ) )
    print( "Number of non-vehicle images: " + str( datasetStats[ "nNotCars" ] ) )
    print( "Image Shape: " + str( datasetStats[ "imageShape" ] ) )
    print( "Data Type: " + str( datasetStats[ "dataType" ] ) )

    # Display a sample image:
    #DisplayImage( mpimg.imread( vehicleImages[ 0 ] ) )

    '''
    Extract HOG and Color Features from the vehicle and
    non-vehicle datasets.

    The following parameters can be tuned:
    1) Colorspace to use (RGB, HSV, LUV, YUV)
    2) orient 
    '''
    colorspace = "RGB"
    orient = 9
    pixelsPerCell = 8
    cellsPerBlock = 2
    hogChannel = 0 

    # Extract Features for Vehicles:
    vehicleFeatures = GetFeatureVectors( vehicleImages, "RGB" )

    # Extract Features for Non-Vehicles:
    nonVehicleFeatures = GetFeatureVectors( nonVehicleImages, "RGB" )

    # Create a normalized array stack of features:
    featureList = [ vehicleFeatures, nonVehicleFeatures ]
    normalizedFeatures = NormalizeFeatureVectors( featureList )

    print( "normalizedFeatures length: " + str( len( normalizedFeatures ) ) )

    '''
    Create the labels vector. A 1 indicates a vehicle image; a 0 indicates
    a non-vehicle image.
    '''
    y_labels = np.hstack( ( np.ones( datasetStats[ "nCars" ] ), np.zeros( datasetStats[ "nNotCars" ] ) ) ) 

    print( "y_labels length: " + str( len( y_labels ) ) )

    # Create training and testing data sets:
    randNum = np.random.randint( 0, 100 )
    X_train, X_test, y_train, y_test = train_test_split( normalizedFeatures, y_labels, test_size = 0.2, random_state = randNum )

    # The Linear Support Vector Machine:
    svm = LinearSVC()

    # Fit the data sets:
    print( "Now fitting X_train and y_train" )
    svm.fit( X_train, y_train )

    # Test the accuracy of the SVM:
    print( "Accuracy of SVM:" )
    print( round( svm.score( X_test, y_test ), 4 ) )

    print( "Done training support vector machine" )

'''
Vehicle Detection Pipeline

This function takes in the full path of the image
to process.  This function returns the same image
but with bounding rectangles overlayed around the detected
vehicles
'''
def CarDetectPipeline( imageName ):

    # Open the image in BGR format:
    image = cv2.imread( imageName )

    # TODO: Replace this when the pipeline is complete:
    return np.copy( image )

'''
Main function.  We are expecting a single command-line argument,
which tells the script whether we want to process images or a video file.
'''
def main():
    
    if len( sys.argv ) > 1:

        if "train" in sys.argv:
            print( "Now training SVM Linear Classifier" )

            TrainClassifier()

        if "images" in sys.argv:
            print( "Now detecting vehicles in " + TEST_IMAGES_FOLDER )

            # All the test JPG images are in TEST_IMAGES_FOLDER:
            images = glob.glob( TEST_IMAGES_FOLDER + "*.jpg" )

            '''
            For each of these images, run the image through the
            CarDetectPipeline and save the output image to the
            folder specified by OUTPUT_IMAGES_FOLDER.
            '''
        
            for imagePath in images:
                
                detectedImage = CarDetectPipeline( imagePath )

                DisplayImage( detectedImage, True )

        if "video" in sys.argv:
            # TODO: We're expecting the video file to analyze to be in the following argument:  
            print( "Now detecting vehicles in video" )

    else:
        # Display usage:
        print( "Usage: " + sys.argv[ 0 ] + " <images> <video> <video file>" )
        return

# Main function runner:
if __name__ == "__main__":
    main()
