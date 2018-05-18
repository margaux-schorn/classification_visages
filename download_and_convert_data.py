# This code come from the Tensorflow github, more precisely :
#   https://github.com/tensorflow/models/tree/master/research/slim
# It has been adapted to suit my project of faces classification.

# ==============================================================================

"""Downloads and converts a particular dataset (here there's only one dataset)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets.download_and_convert_visages import CreatorTFRecords
from utils.labels_extractor import LabelsExtractor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'image_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')
tf.app.flags.DEFINE_string(
    'chemin_csv',
    "labels/labels.csv",
    'The path to csv file'
)
tf.app.flags.DEFINE_string(
    'chemin_tfrecords',
    None,
    'The path to output tfrecords file'
)
tf.app.flags.DEFINE_string(
    'labels_dir',
    None,
    'The directory that contain annotations files for dataset'
)
tf.app.flags.DEFINE_string(
    'labels_list',
    "labels/liste_labels.txt",
    'The path to the complete list of labels (associated to number)'
)
tf.app.flags.DEFINE_string(
    'classes',
    "labels/liste_labels.txt",
    'The path to the list of classes to import'
)

"""
Paramètres pour l'exécution : 
    --dataset_name "visages" --dataset_dir "/Users/margaux/datasets/visages_test/images"     
    --labels_dir "/Users/margaux/datasets/visages_test/annotations"
    --classes ""
    --chemin_tfrecords "labels/labels_tfrecords.tfrecord"
"""


def main(_):
    if not FLAGS.image_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if not os.path.isfile(FLAGS.chemin_csv):
        # Création de l'extracteur des labels contenus dans les fichiers txt
        extractor = LabelsExtractor(FLAGS.image_dir)
        extractor.extract(FLAGS.labels_dir, FLAGS.labels_list, FLAGS.chemin_csv)

    creator = CreatorTFRecords(FLAGS.chemin_tfrecords, FLAGS.image_dir)
    creator.create(FLAGS.chemin_csv, FLAGS.classes)


if __name__ == '__main__':
    tf.app.run()
