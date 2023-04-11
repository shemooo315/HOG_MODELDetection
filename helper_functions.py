import cv2

from skimage.transform import resize
from sklearn.svm import LinearSVC
from skimage import feature

import os

class ImagesExtraction:
    def __init__(self):
       pass

    def features(self, image):
        # resizing image
        resized_img = resize(image, (128, 64))
        feature_des,image_hog = feature.hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)
        return feature_des

class FileHelpers:

    def __init__(self):
        pass

    def getFiles(self, path):

        imlist = {}
        count = 0
        for each in os.listdir(path):
            print(" #### Reading image category ", each, " ##### ")
            imlist[each] = []
            for imagefile in os.listdir(path + '/' + each):
                print("Reading file ", imagefile)
                im = cv2.imread(path + '/' + each + '/' + imagefile, 0)
                imlist[each].append(im)
                count += 1

        return [imlist, count]
