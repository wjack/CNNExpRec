import numpy as np
from scipy.misc import imread
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
class Fer_Parser:
    def parse_all(self):
        f = open('data/fer2013/fer2013.csv')
        X_tr = []
        Y_tr = []
        X_te = []
        Y_te = []


        for line in f.readlines():
            data = line.split(',')
            tag = int(data[0])
            pixels = list(data[1])[0: len(list(data[1])):2]
            pixel_holder = np.array(pixels)
            pixels = list(np.reshape(pixel_holder, (48,48)))
            data_set = data[2]

            if data_set = 'Training':
                X_tr.append(pixels)
                Y_tr.append(tag)

            if data_set = 'PublicTest':
                X_te.append(pixels)
                Y_te.append(tag)

        #convert Y to one hot
        X_tr = np.array(X_tr)
        Y_tr = np.eye(7)[np.array(Y_tr)]
        X_te = np.array(X_te)
        Y_te = np.eye(7)[np.array(Y_te)]
        return X_tr, X_te, Y_tr, Y_te








    def images_to_tensor(self):
        images = []

        for file in sorted(os.listdir('data/jaffe_images_small')) :

            if(file != '.DS_Store'):

                image = mpimg.imread('data/jaffe_images_small/' + file)
                if (len(np.shape(image)) > 2):
                    image = image[:,:,0]
                image = image.tolist()
                images.append(image)

        image_tensor = np.array(images)
        image_tensor = image_tensor.reshape(213, 64, 64, 1)

        return image_tensor

    def text_to_tensor(self):
        labels = []
        text = open('data/jaffe_labels/JAFFE_labels.txt')
        for line in text.readlines()[2:]:
            line_labels = line.split()
            labels.append(line_labels[1:-1])
        label_tensor = np.array(labels)

        return label_tensor

    def text_to_one_hot(self):
        #labels are of the form [NEU HAP SAD SUR ANG DIS FEA]
        labels = []
        text = open('data/jaffe_labels/JAFFE_labels.txt')
        for line in text.readlines()[2:]:
            line_label = line[-4:-2]
            tag = line [-7:-1]

            index = 0
            if line_label == 'NE':
                index == 0
            if line_label == 'HA':
                index = 1
            if line_label == 'SA':
                index = 2
            if line_label == 'SU':
                index = 3
            if line_label == 'AN':
                index = 4
            if line_label == 'DI':
                index = 5
            if line_label == 'FE':
                index = 6
            labels.append([tag,index])

        label_tensor = np.array(labels)

        i  = np.argsort(label_tensor[:,0])

        label_tensor = label_tensor[i]

        label_tensor = label_tensor[:,1]

        label_tensor = label_tensor.astype(int)

        one_hot = np.zeros([np.size(label_tensor), 7])
        for i in range(np.size(label_tensor)):
            one_hot[i, label_tensor[i]] = 1
        return one_hot
