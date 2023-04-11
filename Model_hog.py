import cv2
import numpy as np
from glob import glob
import argparse
from helper_functions import *


class Classification:
    def __init__(self):
        self.train_path = None
        self.test_path = None
        self.im_helper = ImagesExtraction()
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list_images = []
        self.labels = []
        self.svm_model = LinearSVC(random_state=42, tol=1e-5)

    def trainModel(self):
        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print("Computing Features for ", word)
            for im in imlist:
                self.train_labels = np.append(self.train_labels, label_count)
                feature_des_image = self.im_helper.features(im)
                self.descriptor_list_images.append(feature_des_image)
                self.labels.append(label_count)

            label_count += 1
        self.descriptor_list_images = np.array(self.descriptor_list_images)
        self.labels = np.array(self.labels)
        # train Linear SVC
        print('Training on train images...')
        self.svm_model.fit(self.descriptor_list_images, self.labels)

    def testModel(self):

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []

        for word, imlist in self.testImages.items():
            print("processing ", word)
            for im in imlist:
                feature_des_image = self.im_helper.features(im)
                predictions.append(feature_des_image)

        pred = self.svm_model.predict(predictions)
        print("The list of predictions for images :.....")
        print(pred)
        Counter_Zeros_Brainscans = 0
        Counter_Ones_Breastscans = 0

        for Counter in pred:
            if (Counter == 0):
                Counter_Zeros_Brainscans = Counter_Zeros_Brainscans + 1
            else:
                Counter_Ones_Breastscans = Counter_Ones_Breastscans + 1

        Accuracy = ((Counter_Ones_Breastscans + Counter_Zeros_Brainscans) / self.testImageCount) * 100
        print("The Accuracy for Model = ")
        print(Accuracy)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=" HOG MODEL EXAMPLE")
    parser.add_argument('--train_path', default="CV2023CSYSDataset2\\Trainn", action="store", dest="train_path")
    parser.add_argument('--test_path', default="CV2023CSYSDataset2\\Testt", action="store", dest="test_path")
    args = vars(parser.parse_args())
    print(args)

    classification = Classification()

    # set training paths
    classification.train_path = args['train_path']
    # set testing paths
    classification.test_path = args['test_path']
    # train the model
    classification.trainModel()
    # test model
    classification.testModel()