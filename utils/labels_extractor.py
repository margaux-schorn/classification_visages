import csv
import os
from PIL import Image  # Pour récupérer les dimensions de l'image
import warnings
import tensorflow as tf


def labels_to_class_name(chemin_liste):
    """
        Cette méthode permet de récupérer la liste de labels sous forme
        de dictionnaire en lui associant le nombre correspondant dans le
        fichier.
    :param chemin_liste: le chemin vers la liste de labels
    :return: un dictionnaire associant un nombre à un label sous forme de
            chaîne de caractères.
    """
    with tf.gfile.Open(chemin_liste, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')

    labels_to_class_names = {}
    i = 1
    for line in lines:
        index = line.index(':')
        numero = int(line[:index])
        # pour les tests avec nbre réduit de labels sans changer les nbre assignés dans le fichier
        if numero != i:
            numero = i
        labels_to_class_names[numero] = line[index + 1:]
        i += 1

    print(labels_to_class_names)

    return labels_to_class_names


def read_annotations(chemin_liste):
    """
        Cette méthode permet de récupérer les annotations
        contenues dans un fichier txt.
    :param chemin_liste: le chemin vers le fichier contenant les annotations
    :return: une liste de chaines de caractères correspondants aux labels
            récupérés dans le fichier d'annotations.
    """
    with tf.gfile.Open(chemin_liste, 'r') as f:
        lines = f.read()

    lines = lines.split('\n')

    return lines


def write_labels_in_csv(cpt_labels, fichier_csv, image, img_annots,
                        inv_labels_image, image_data):
    for label in img_annots:
        label_lower = label.lower()
        if label_lower in inv_labels_image:
            # ecrire une ligne : nom image - extension - hauteur - largeur - label
            fichier_csv.writerow([image_data['nom'], image_data['extension'], image_data['hauteur'],
                                  image_data['largeur'], inv_labels_image[label_lower]])

            if label_lower not in cpt_labels.keys():
                cpt_labels[label_lower] = 1
            else:
                cpt_labels[label_lower] += 1

        elif len(label_lower) > 0:
            # avertissement : le label n'est pas renseigné dans la liste complète
            print("Label posant problème : {} (image {} )".format(label_lower, image))
            """
            # mis en commentaire pour l'expérience avec 2 labels pour ne pas surcharger
            # la console de messages
            warnings.warn("Avertissement : le label de l'image n'est pas renseigné "
                          "dans la liste des labels, \nil sera ignoré lors de la création du "
                          "fichier csv")
            """


class LabelsExtractor:
    def __init__(self, chemin_dataset):
        self.chemin_dataset = chemin_dataset
        self.nbre_images = 0

    def extract(self, chemin_annotations, chemin_liste_labels, chemin_csv):
        """
            Permet de récupérer les labels contenu dans un fichier d'annotations
            définit pour une image, et de les rassembler dans un fichier CSV

            Chaque ligne du fichier CSV correspond à l'association d'un label
            à une image. On peut donc avoir plusieurs fois la même image associée
            à des labels différents.
        :param chemin_annotations: le chemin vers le dossier contenant les annotations du dataset
        :param chemin_liste_labels: le chemin vers le fichier contenant la liste des labels
        :param chemin_csv: le chemin qui correspondra au nouveau fichier csv
        :return: /
        """

        # récupérer la liste de labels depuis le fichier txt
        labels_image = labels_to_class_name(chemin_liste_labels)
        inv_labels_image = {v: k for k, v in labels_image.items()}

        cpt_labels = {}  # afin de vérifier le nombre de fois où un label est rencontré

        # créer un fichier csv dans un dossier du projet
        with open(chemin_csv, "w") as fichier:
            fichier_csv = csv.writer(fichier)

            # parcourir les images du dataset
            for image in os.listdir(self.chemin_dataset):
                self.create_records_for_image(chemin_annotations, cpt_labels, fichier_csv, image, inv_labels_image)

            print("Nombre d'images : {}".format(self.nbre_images))
            fichier_csv.writerow(["nbre_images", self.nbre_images])

        for key in cpt_labels.keys():
            print("Présence du label '{}' : {}".format(key, cpt_labels[key]))

    def create_records_for_image(self, chemin_annotations, cpt_labels, fichier_csv, image, inv_labels_image):
        if not image.startswith('.'):
            image_splited = image.split('.')

            open_image = Image.open(os.path.join(self.chemin_dataset, image))
            hauteur, largeur = open_image.size

            image_data = {'nom': image_splited[0], 'extension': image_splited[1],
                          'hauteur': hauteur, 'largeur': largeur}

            chemin_txt = os.path.join(chemin_annotations, "{}.{}".format(image_data['nom'], "txt"))

            # vérifier qu'on va analyser un fichier txt
            if os.path.isfile(chemin_txt):
                # récupérer les labels associés à l'image
                img_annots = read_annotations(chemin_txt)

                # écrire les records pour chaque labels
                write_labels_in_csv(cpt_labels, fichier_csv, image, img_annots, inv_labels_image, image_data)
                self.nbre_images += 1
