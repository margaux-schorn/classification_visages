import tensorflow as tf
import os
import io
from PIL import Image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def recuperer_lignes_csv(chemin_csv):
    """Cette méthode permet de récupérer le contenu d'un fichier
        csv et de place chaque élément d'une ligne dans un tuple.

        Returns : une liste de tuples"""
    lignes_csv = []

    with open(chemin_csv) as csv:
        for line in csv.readlines():
            elements_separes = line.split(",")

            if len(elements_separes) == 5:
                lignes_csv.append((elements_separes[0], elements_separes[1],
                                   elements_separes[2], elements_separes[3],
                                   elements_separes[4].split('\n')[0]))  # Enlever le \n de fin de ligne

        # print(lignes_csv)
    return lignes_csv


class CreatorTFRecords:
    def __init__(self, chemin_tfrecords, chemin_dataset):
        self.chemin_tfrecords = chemin_tfrecords
        self.chemin_dataset = chemin_dataset

    def create(self, chemin_csv):
        """Permet la création d'un fichier TFRecords grâces aux associations
            image - label du dataset, inscrit dans un fichier CSV."""

        writer = tf.python_io.TFRecordWriter(self.chemin_tfrecords)
        lignes_csv = recuperer_lignes_csv(chemin_csv)

        print("Enregistrement du fichier TFRecords en cours...")

        for ligne in lignes_csv:
            chemin_image = os.path.join(self.chemin_dataset, "{}.{}".format(ligne[0], ligne[1]))

            with tf.gfile.GFile(chemin_image, 'rb') as fid:
                try:
                    encoded_jpg = fid.read()
                except tf.errors.NotFoundError:
                    print('File {} not found'.format(chemin_image))
                    continue

            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size

            print('Adding image {}x{} at path {} for label {}'.format(width, height, chemin_image, ligne[4].encode()))

            record = tf.train.Example(features=tf.train.Features(
                feature={
                    'image/height': _int64_feature(height),
                    'image/width': _int64_feature(width),
                    'image/encoded': _bytes_feature(encoded_jpg),
                    'image/format': _bytes_feature('jpeg'.encode('utf8')),
                    'image/class/label': _int64_feature(int(ligne[4]))
                }
            ))

            writer.write(record.SerializeToString())

        writer.close()