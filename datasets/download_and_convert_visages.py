import cv2
import tensorflow as tf
import os

from datasets import visages
from utils.csv_reader import CsvReader


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class CreatorTFRecords:
    def __init__(self, chemin_tfrecords, chemin_dataset):
        self.chemin_tfrecords = chemin_tfrecords
        self.chemin_dataset = chemin_dataset

    def create(self, chemin_csv, classes):
        """Permet la création d'un fichier TFRecords grâces aux associations
            image - label du dataset, inscrit dans un fichier CSV."""

        writer = tf.python_io.TFRecordWriter(self.chemin_tfrecords)
        lignes_csv = CsvReader.recuperer_lignes_csv(chemin_csv)

        print("Enregistrement du fichier TFRecords en cours...")

        classes_map = visages.labels_to_class_name(classes)

        for ligne in lignes_csv:
            chemin_image = os.path.join(self.chemin_dataset, "{}.{}".format(ligne[0], ligne[1]))

            image = cv2.imread(chemin_image)

            height, width, channels = image.shape

            # encodage de l'image pour créer le record
            encoded_image = cv2.imencode('.{}'.format(ligne[1]), image)[1].tostring()

            if int(ligne[4]) in classes_map:

                print('Adding image {}x{} at path {} for label {}'.format(width, height, chemin_image, ligne[4].encode()))

                record = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image/height': _int64_feature(height),
                        'image/width': _int64_feature(width),
                        'image/encoded': _bytes_feature(encoded_image),
                        'image/format': _bytes_feature(ligne[1].encode('utf8')),
                        'image/class/label': _int64_feature(int(ligne[4]))
                    }
                ))

                writer.write(record.SerializeToString())

        writer.close()