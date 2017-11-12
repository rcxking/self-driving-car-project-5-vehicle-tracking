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
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
Helper function to generate a 3-D Plot of an image:
'''
def Plot3D( pixels, rgb, axis_labels = list( "RGB" ), axis_limits = ( ( 0, 255 ), ( 0, 255 ), ( 0, 255 ) ) ):

    # Create figure and 3-D Axes:
    fig = plt.figure( figsize = ( 8, 8 ) )
    ax = Axes3D( fig )

    # Set axis limits:
    ax.set_xlim( *axis_limits[ 0 ] )
    ax.set_ylim( *axis_limits[ 1 ] )
    ax.set_zlim( *axis_limits[ 2 ] )

    # Set axis labels and sizes:
    ax.tick_params( axis = 'both', which = 'major', labelsize = 14, pad = 8 )
    ax.set_xlabel( axis_labels[ 0 ], fontsize = 16, labelpad = 16 )
    ax.set_ylabel( axis_labels[ 1 ], fontsize = 16, labelpad = 16 )
    ax.set_zlabel( axis_labels[ 2 ], fontsize = 16, labelpad = 16 )

    # Plot pixel values with colors given in rgb:
    ax.scatter( pixels[ :, :, 0 ].ravel(), pixels[ :, :, 1 ].ravel(), pixels[ :, :, 2 ].ravel(), c = rgb.reshape( ( -1, 3 ) ), edgecolors = 'none' )

    return ax

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
    if vis:
        features, hogImage = hog( img, orientations = orient, pixels_per_cell = ( pixelsPerCell, pixelsPerCell ), cells_per_block = ( cellsPerBlock, cellsPerBlock ), transform_sqrt = True, visualise = vis, feature_vector = featureVec )

        return features, hogImage
    else:
        features = hog( img, orientations = orient, pixels_per_cell = ( pixelsPerCell, pixelsPerCell ), cells_per_block = ( cellsPerBlock, cellsPerBlock ), transform_sqrt = True, visualise = vis, feature_vector = featureVec )

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
def GetFeatureVectors( imgList, colorspace, orient, pixelsPerCell, cellsPerBlock ):

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

        # Extract HOG Features:
	#def GetHOGFeatures( img, orient, pixelsPerCell, cellsPerBlock, vis=False, featureVec=True )
        hogFeatures = GetHOGFeatures( featureImg[ :, :, 0 ], orient, pixelsPerCell, cellsPerBlock, False, True )

        featureVec.append( np.concatenate( ( spatialFeatures, hogFeatures ) ) )

    return featureVec


'''
This function is used to train the Support Vector Machine.

This function will save the trained data in a pickle file. 
'''
def TrainClassifier():

    # Print dataset statistics:
    datasetStats = GetDatasetStats( vehicleImages, nonVehicleImages )
    print( "Number of vehicle images: " + str( datasetStats[ "nCars" ] ) )
    print( "Number of non-vehicle images: " + str( datasetStats[ "nNotCars" ] ) )
    print( "Image Shape: " + str( datasetStats[ "imageShape" ] ) )
    print( "Data Type: " + str( datasetStats[ "dataType" ] ) )

    # Display a sample image:
    #DisplayImage( mpimg.imread( vehicleImages[ 0 ] ) )

    '''
    DEBUG ONLY: Uncomment these lines to generate 3D plots for
    a vehicle and non-vehicle image.  Use this to help determine
    which colorspace to work in.
    '''

    if False:
        randNum = random.randint( 0, min( datasetStats[ "nCars" ], datasetStats[ "nNotCars" ] ) ) 
        vehicleImg = cv2.imread( vehicleImages[ randNum ] ) #mpimg.imread( vehicleImages[ randNum ] )
        nonVehicleImg = cv2.imread( nonVehicleImages[ randNum ] ) #mpimg.imread( nonVehicleImages[ randNum ] )

        vehicleImg = cv2.cvtColor( vehicleImg, cv2.COLOR_BGR2RGB )
        nonVehicleImg = cv2.cvtColor( nonVehicleImg, cv2.COLOR_BGR2RGB )

        vehicleRGB = vehicleImg / 255.
        nonVehicleRGB = nonVehicleImg / 255.

        # Plot is linearly seperable
        vehicleImgHSV = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2HSV )
        nonVehicleImgHSV = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2HSV )

        # Plot is linearly seperable
        vehicleImgHLS = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2HLS )
        nonVehicleImgHLS = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2HLS )
    
        # Plot is not linearly seperable
        vehicleImgYUV = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2YUV )
        nonVehicleImgYUV = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2YUV )

        vehicleImgLUV = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2LUV )
        nonVehicleImgLUV = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2LUV )

        Plot3D( vehicleImgLUV, vehicleRGB, axis_labels = list( "LUV" ) )
        plt.show()

        Plot3D( nonVehicleImgLUV, nonVehicleRGB, axis_labels = list( "LUV" ) )
        plt.show()

        return

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

    # def GetFeatureVectors( imgList, colorspace="RGB", orient, pixelsPerCell, cellsPerBlock ):
    # Extract Features for Vehicles:
    vehicleFeatures = GetFeatureVectors( vehicleImages, "RGB", orient, pixelsPerCell, cellsPerBlock )

    # Extract Features for Non-Vehicles:
    nonVehicleFeatures = GetFeatureVectors( nonVehicleImages, "RGB", orient, pixelsPerCell, cellsPerBlock )

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
