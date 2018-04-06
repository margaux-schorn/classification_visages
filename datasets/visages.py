from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'labels/labels_records.tfrecord'

CHEMIN_LISTE_LABELS = "labels/liste_labels.txt"

SPLITS_TO_SIZES = {'train': 225, 'test': 75}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [250 x 250 x 3] color image.',
    'label': 'A integer associate to a string value in liste_labels.txt',
}


def get_nbre_labels(chemin_liste_labels):
    """Cette méthode permet d'obtenir le nombre de labels possible
        contenus dans la liste associant chaque label à un entier."""
    with tf.gfile.Open(chemin_liste_labels, 'r') as f:
        lines = f.read()

    lines = lines.split('\n')

    nbre_labels = len(lines)
    print("Nombre de labels {}".format(nbre_labels))

    return nbre_labels


def labels_to_class_name(chemin_liste):
    """Cette méthode permet d'obtenir un dictionnaire associant
        les entiers et labels correspondants contenus dans la
        liste de tous les labels. """
    with tf.gfile.Open(chemin_liste, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    print(labels_to_class_names)

    return labels_to_class_names


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Cette méthode provient initialement du code du projet de recherches
    de TF-Slim du GitHub de Tensorflow. Elle a été adaptée pour n'employer
    que le code nécessaire pour mon dataset. """

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern = os.path.join(_FILE_PATTERN)
    print(file_pattern)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[250, 250, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    print("Read label file from {}".format(CHEMIN_LISTE_LABELS))
    liste_labels = labels_to_class_name(CHEMIN_LISTE_LABELS)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=get_nbre_labels(CHEMIN_LISTE_LABELS),
        labels_to_names=liste_labels)
