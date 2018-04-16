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
    'dataset_name',
    None,
    'The name of the dataset to convert ("visages")')

tf.app.flags.DEFINE_string(
    'dataset_dir',
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

"""
Paramètres pour l'exécution : 
    --dataset_name "visages" --dataset_dir "/Users/margaux/datasets/visages_test/images"
    --chemin_tfrecords "labels/labels_tfrecords.tfrecord" 
    --labels_dir "/Users/margaux/datasets/visages_test/annotations"
"""


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if FLAGS.dataset_name == 'visages':
        if not os.path.isfile(FLAGS.chemin_csv):
            # Création de l'extracteur des labels contenus dans les fichiers txt
            extractor = LabelsExtractor(FLAGS.dataset_dir)
            extractor.extract(FLAGS.labels_dir, FLAGS.labels_list, FLAGS.chemin_csv)

        creator = CreatorTFRecords(FLAGS.chemin_tfrecords, FLAGS.dataset_dir)
        creator.create(FLAGS.chemin_csv)
    else:
        raise ValueError(
            'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)


if __name__ == '__main__':
    tf.app.run()
