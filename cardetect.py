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
import glob
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Constants

# Folder to training images:
TRAINING_DATA_FOLDER = "./training_data/"
VEHICLE_IMAGES_FOLDER = TRAINING_DATA_FOLDER + "vehicles/"
NONVEHICLE_IMAGES_FOLDER = TRAINING_DATA_FOLDER + "non-vehicles/"

TEST_IMAGES_FOLDER = "./test_images/"
OUTPUT_IMAGES_FOLDER = "./output_images/"

# Tuning Parameters:
colorspace = "HSV"
orient = 16
pixelsPerCell = 4
cellsPerBlock = 2
hogChannel = 2


# Name of the pickle file for SVM:
PICKLE_FILE = "./svm.pickle"

# Scaler File:
SCALER_FILE = "./scaler.pickle"

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
    ch1 = np.histogram( img[ :, :, 0 ], bins = nbins, range = binsRange )
    ch2 = np.histogram( img[ :, :, 1 ], bins = nbins, range = binsRange )
    ch3 = np.histogram( img[ :, :, 2 ], bins = nbins, range = binsRange )

    # Concatenate the histograms to form the feature vector:
    histFeatures = np.concatenate( ( ch1[ 0 ], ch2[ 0 ], ch3[ 0 ] ) )

    return histFeatures
    
'''
Helper function to resize a given image and return
a 1-Dimensional spatial histogram.

This function assumes that the input image is a three-
channel image.

The output histogram will have elements such that:

Ch1 - Ch2 - Ch3
'''
def ComputeSpatialHistogram( img, size = (32, 32 ) ):

    ch1 = cv2.resize( img[ :, :, 0 ], size ).ravel()
    ch2 = cv2.resize( img[ :, :, 1 ], size ).ravel()
    ch3 = cv2.resize( img[ :, :, 2 ], size ).ravel()

    return np.hstack( ( ch1, ch2, ch3 ) )

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

    # Dump the featureScaler:
    joblib.dump( featureScaler, SCALER_FILE )

    # Apply the scaler to the featureStack:
    scaledFeatures = featureScaler.transform( featureStack )

    return scaledFeatures

'''
This function adds heat to the heatmap
image for all pixels inside each bounding
box given in the bboxList.

bboxList is a list of tuples:
    ( ( x1, y1 ), ( x2, y2 ) )
'''
def AddHeat( heatmap, bboxList ):
    
    for box in bboxList:
        heatmap[ box[ 0 ][ 1 ]:box[ 1 ][ 1 ], box[ 0 ][ 0 ]:box[ 1 ][ 0 ] ] += 1

    return heatmap

'''
This function takes in a heatmap image, and zero's out
any pixels in cells that have less hits than the given
threshold.
'''
def ApplyHeatmapThreshold( heatmap, threshold ):
    heatmap[ heatmap <= threshold ] = 0
    return heatmap

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
Helper function to extract a feature vector from a single image. 
'''
def GetSingleImageFeatures( img, orient, pixelsPerCell, cellsPerBlock, hogChannel ):

    # Extract Spatial Features from all 3-Channels:
    spatialFeatures = ComputeSpatialHistogram( img, size = ( 32, 32 ) )

    # Compute the Color Histogram:
    colorHist = ComputeColorHistogram( img )

    # Extract HOG Features:
    #def GetHOGFeatures( img, orient, pixelsPerCell, cellsPerBlock, vis=False, featureVec=True )
    ch1Features = GetHOGFeatures( img[ :, :, 0 ], orient, pixelsPerCell, cellsPerBlock, False, True )
    ch2Features = GetHOGFeatures( img[ :, :, 1 ], orient, pixelsPerCell, cellsPerBlock, False, True )
    ch3Features = GetHOGFeatures( img[ :, :, 2 ], orient, pixelsPerCell, cellsPerBlock, False, True )

    return np.concatenate( ( spatialFeatures, colorHist, ch1Features, ch2Features, ch3Features ) )

'''
This function takes in a list of images, then
performs the following steps:

1) Open the image in RGB format
2) Convert from RGB to specified colorspace
3) 
'''
def GetFeatureVectors( imgList, colorspace, orient, pixelsPerCell, cellsPerBlock, hogChannel ):

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

        featureVec.append( GetSingleImageFeatures( featureImg, orient, pixelsPerCell, cellsPerBlock, hogChannel ) )

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

        # Possible candidate; car pixels clustered in S-Plane  
        vehicleImgHSV = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2HSV )
        nonVehicleImgHSV = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2HSV )

        # Car Pixels clustered in L-Plane; not as clear as HSV
        vehicleImgHLS = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2HLS )
        nonVehicleImgHLS = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2HLS )
    
        # Car Pixels clustered in Y-Plane
        vehicleImgYUV = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2YUV )
        nonVehicleImgYUV = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2YUV )

        vehicleImgLUV = cv2.cvtColor( vehicleImg, cv2.COLOR_RGB2LUV )
        nonVehicleImgLUV = cv2.cvtColor( nonVehicleImg, cv2.COLOR_RGB2LUV )

        Plot3D( vehicleImgHSV, vehicleRGB, axis_labels = list( "HSV" ) )
        plt.show()

        Plot3D( nonVehicleImgHSV, nonVehicleRGB, axis_labels = list( "HSV" ) )
        plt.show()

        return

    '''
    Extract HOG and Color Features from the vehicle and
    non-vehicle datasets.

    The following parameters can be tuned:
    1) Colorspace to use (RGB, HSV, LUV, YUV)
    2) orient 
    '''
    # def GetFeatureVectors( imgList, colorspace="RGB", orient, pixelsPerCell, cellsPerBlock ):
    # Extract Features for Vehicles:
    vehicleFeatures = GetFeatureVectors( vehicleImages, colorspace, orient, pixelsPerCell, cellsPerBlock, hogChannel )

    # Extract Features for Non-Vehicles:
    nonVehicleFeatures = GetFeatureVectors( nonVehicleImages, colorspace, orient, pixelsPerCell, cellsPerBlock, hogChannel )

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

    # Save the SVM data into file:
    joblib.dump( svm, PICKLE_FILE ) 

    print( "Done training support vector machine" )

'''
This function takes in a (X,Y) start and end location
and performs a sliding window search across an input
image.  Returns a list of sliding windows.   
'''
def GetSlidingWindows( img, xStartStop=[None, None], yStartStop=[None, None], xyWindow=(64,64), xyOverlap=(0.5, 0.5) ):

    # Set image limits if they don't already exist:
    if xStartStop[ 0 ] == None:
        xStartStop[ 0 ] = 0
    if xStartStop[ 1 ] == None:
        xStartStop[ 1 ] = img.shape[ 1 ]
    if yStartStop[ 0 ] == None:
        yStartStop[ 0 ] = 0
    if yStartStop[ 1 ] == None:
        yStartStop[ 1 ] = img.shape[ 0 ]

    # Compute Length of Search Area:
    xSearch = xStartStop[ 1 ] - xStartStop[ 0 ]
    ySearch = yStartStop[ 1 ] - yStartStop[ 0 ]

    '''
    Compute the number of pixels each window
    covers.
    '''
    xStepPixels = np.int( xyWindow[ 0 ] * ( 1 - xyOverlap[ 0 ] ) )
    yStepPixels = np.int( xyWindow[ 1 ] * ( 1 - xyOverlap[ 1 ] ) )


    # Compute number of windows:
    xBuff = np.int( xyWindow[ 0 ] * xyOverlap[ 0 ] )
    yBuff = np.int( xyWindow[ 1 ] * xyOverlap[ 1 ] )

    xNumWindows = np.int( ( xSearch - xBuff ) / xStepPixels )
    yNumWindows = np.int( ( ySearch - yBuff ) / yStepPixels )

    windowList = []

    print( "xNumWindows: " + str( xNumWindows ) )
    print( "yNumWindows: " + str( yNumWindows ) )

    for y in range( yNumWindows ):
        for x in range( xNumWindows ):

            # Compute next window position:
            startX = ( x * xStepPixels ) + xStartStop[ 0 ]
            endX = startX + xyWindow[ 0 ]
            startY = ( y * yStepPixels ) + yStartStop[ 0 ]
            endY = startY + xyWindow[ 1 ]

            windowList.append( ( ( startX, startY ), ( endX, endY ) ) )
        

    return windowList

'''
Vehicle Detection Pipeline

This function takes in the full path of the image
to process.  This function returns the same image
but with bounding rectangles overlayed around the detected
vehicles
'''
def CarDetectPipeline( imageName ):

    # Open the image in RGB format:
    image = mpimg.imread( imageName )

    # Load the classifier data from file:
    svm = joblib.load( PICKLE_FILE )  

    # Load the scaler from file:
    scaler = joblib.load( SCALER_FILE )

    # Copy of image to draw bounding boxes on:
    drawImg = np.copy( image )
    
    # Normalize the input image:
    image = image.astype( np.float32 ) / 255

    '''
    We're only interested in finding vehicles in the bottom
    half of the image.  Crop out this region.

    Region found by examining images with the GIMP program.
    '''
    startY = int( image.shape[ 0 ] / 2 )
    endY = 625
    imgToSearch = image[ startY:endY, :, : ]

    #DisplayImage( imgToSearch )

    imgScale = 1.1

    # Convert the image to search to the desired colorspace:
    ctransToSearch = cv2.cvtColor( imgToSearch, cv2.COLOR_RGB2HSV )

    # Scale the image if necessary:
    if imgScale != 1.0:
        imgShape = ctransToSearch.shape
        ctransToSearch = cv2.resize( ctransToSearch, ( np.int( imgShape[ 1 ] / imgScale ), np.int( imgShape[ 0 ] / imgScale ) ) ) 

    # Now get the three channels (HSV):
    ch1 = ctransToSearch[ :, :, 0 ]
    ch2 = ctransToSearch[ :, :, 1 ]
    ch3 = ctransToSearch[ :, :, 2 ]

    # Get the number of blocks to analyze:
    nxBlocks = ( ch1.shape[ 1 ] // pixelsPerCell ) - cellsPerBlock + 1
    nyBlocks = ( ch1.shape[ 0 ] // pixelsPerCell ) - cellsPerBlock + 1
    numFeaturesPerBlock = orient * cellsPerBlock ** 2
    print( "nxBlocks: " + str( nxBlocks ) )
    print( "nyBlocks: " + str( nyBlocks ) )

    # Each image is 64 pixels (8 cells @ 8 pixels per cell):
    window = 64
    nBlocksPerWindow = ( window // pixelsPerCell ) - cellsPerBlock + 1
    cellsPerStep = 2
    nxSteps = ( nxBlocks - nBlocksPerWindow ) // cellsPerStep
    nySteps = ( nyBlocks - nBlocksPerWindow ) // cellsPerStep

    # Get the HOG features for the cropped image:
    ch1Features = GetHOGFeatures( ch1, orient, pixelsPerCell, cellsPerBlock, False, False )
    ch2Features = GetHOGFeatures( ch2, orient, pixelsPerCell, cellsPerBlock, False, False )
    ch3Features = GetHOGFeatures( ch3, orient, pixelsPerCell, cellsPerBlock, False, False )
    #features = GetSingleImageFeatures( convertImgToSearch, orient, pixelsPerCell, cellsPerBlock, hogChannel )

    for xb in range( nxSteps ):
        for yb in range( nySteps ):
            yPos = yb * cellsPerStep
            xPos = xb * cellsPerStep

            #print( "xPos: " + str( xPos ) )
            #print( "yPos: " + str( yPos ) )

            # Extract HOG features for this patch:
            #hogFeat1 = features[ yPos : yPos + nBlocksPerWindow, xPos : xPos + nBlocksPerWindow ].ravel()
            
            hogFeat1 = ch1Features[ yPos : yPos + nBlocksPerWindow, xPos : xPos + nBlocksPerWindow ].ravel()
            hogFeat2 = ch2Features[ yPos : yPos + nBlocksPerWindow, xPos : xPos + nBlocksPerWindow ].ravel()
            hogFeat3 = ch3Features[ yPos : yPos + nBlocksPerWindow, xPos : xPos + nBlocksPerWindow ].ravel()

            hogFeatures = np.hstack( ( hogFeat1, hogFeat2, hogFeat3 ) )

            xLeft = xPos * pixelsPerCell
            yTop = yPos * pixelsPerCell

            # Extract the image patch:
            subImg = cv2.resize( ctransToSearch[ yTop : yTop + window, xLeft: xLeft + window ], (64, 64) ) 

            # Get Color Features:
            spatialFeatures = ComputeSpatialHistogram( subImg )
            colorHist = ComputeColorHistogram( subImg )

            # Scale features and make a prediction:
            testFeatures = scaler.transform( np.hstack( ( spatialFeatures, colorHist, hogFeatures ) ).reshape( 1, -1 ) ) 

            testPrediction = svm.predict( testFeatures )

            # Does the SVM predict a vehicle in the sub image?
            if testPrediction == 1:
                xboxLeft = np.int( xLeft * imgScale )
                yTopDraw = np.int( yTop * imgScale )
                winDraw = np.int( window * imgScale )

                cv2.rectangle( drawImg, ( xboxLeft, yTopDraw + startY ), ( xboxLeft + winDraw, yTopDraw + winDraw + startY ), ( 0, 0, 255 ), 6 )

    return drawImg
    
    # GetSlidingWindows( img, xStartStop=[None, None], yStartStop=[None, None], xyWindow=(64,64), xyOverlap=(0.5, 0.5) )

    '''
    Examine only the lower half of the image (so the we'll look at anything
    from ( 0, imageHeight / 2 ) to ( imageWidth, imageHeight ).
    '''
    startX = 0
    startY = int( image.shape[ 0 ] / 2 )

    slidingWindows = GetSlidingWindows( image, xStartStop = [ startX, None ], yStartStop = [ startY, None ]  )

    for window in slidingWindows:
        # Extract the next window:
        imgWindow = cv2.resize( image[ window[ 0 ][ 1 ]:window[ 1 ][ 1 ], window[ 0 ][ 0 ] : window[ 1 ][ 0 ] ], ( 64, 64 ) )

        #DisplayImage( imgWindow )

        # Extract features for this window:
        features = GetSingleImageFeatures( imgWindow, orient, pixelsPerCell, cellsPerBlock, hogChannel )

        # Scale the features:
        scaledFeatures = scaler.transform( np.array( features ).reshape( 1, -1 ) )

        # Prediction:
        pred = svm.predict( scaledFeatures )
        print( "Vehicle detected: " + str( pred ) )

        if pred == 1 :
            DisplayImage( imgWindow )

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

                DisplayImage( detectedImage )

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
