import argparse

import tensorflow as tf
import numpy as np
import cv2

"""
    Le code de base permettant de tester le 'frozen model'
    provient du site cv-tricks.
    
    Paramètres d'exécution : 
        --image_path "<chemin de l'image>" --frozen_model_path "output/frozen_model.pb"
"""


def load_labels(chemin_liste):
    with tf.gfile.Open(chemin_liste, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')

    labels_to_class_names = []
    for line in lines:
        index = line.index(':')
        labels_to_class_names.append(line[index+1:])
        # print(line[index+1:])

    return labels_to_class_names


def predict_image(image_path, frozen_model_path):
    image_size = 250
    num_channels = 3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(image_path)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = np.resize(images, (16, image_size, image_size, num_channels))
    print(x_batch.shape)

    # Load graph with frozen graph
    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:

        y_pred, x = tf.import_graph_def(graph_def, return_elements=["InceptionV3/Logits/SpatialSqueeze:0",
                                                                    "input:0"])
        # for op in graph.get_operations():
        #    print(op)

        sess = tf.Session(graph=graph)

        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        print(result)
        sess.close()

        # cette partie est dédiée à l'affichage des résultats
        # TODO améliorer et décoder l'affichage
        result = np.squeeze(result)


"""
        top_k = result.argsort()[-5:][::-1]
        labels = load_labels('labels/liste_labels.txt')
        for line in top_k:
            print(line)

        print(result)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to image that we want predict")
    parser.add_argument("--frozen_model_path", type=str, help="Path to frozen model")
    args = parser.parse_args()

    predict_image(args.image_path, args.frozen_model_path)
