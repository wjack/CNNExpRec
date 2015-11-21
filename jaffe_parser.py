import numpy as np
from scipy.misc import imread
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg

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

imgs = images_to_tensor()
print tf.shape(imgs)

sess = tf.Session()
init = tf.initialize_all_variables()

sess.run(init)
