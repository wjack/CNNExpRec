import numpy as np
from scipy.misc import imread
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
class Jaffee_Parser:
    def images_to_tensor():
        images = []

        for file in os.listdir('data/jaffe_images') :

            if(file != '.DS_Store'):
                image = mpimg.imread('data/jaffe_images/' + file)
                if (len(np.shape(image)) > 2):
                    image = image[:,:,0]
                image = image.tolist()
                images.append(image)

        image_tensor = tf.convert_to_tensor(images)

        return image_tensor

    def text_to_tensor():
        labels = []
        text = open('data/jaffe_labels/JAFFE_labels.txt')
        for line in text.readlines()[2:]:
            line_labels = line.split()
            labels.append(line_labels[1:-1])
        label_tensor = tf.convert_to_tensor(labels)
        return label_tensor
