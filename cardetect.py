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
from matplotlib import pyplot as plt
from skimage.feature import hog

# Constants
TEST_IMAGES_FOLDER = "./test_images/"
OUTPUT_IMAGES_FOLDER = "./output_images/"

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
This function is used to train the Support Vector Machine.

This function will save the trained data in a pickle file. 
'''
def TrainClassifier():
    pass

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
