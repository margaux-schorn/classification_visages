import csv
import os
from PIL import Image  # Pour récupérer les dimensions de l'image
import warnings
import tensorflow as tf


# Etape 1 : Créer un fichier csv contenant toutes les associations images - labels
# (on peut donc avoir plusieurs lignes pour la même images, une par label)
# Attention : vérifier que le label ajouté existe bien dans le fichier général des labels


def labels_to_class_name(chemin_liste):
    with tf.gfile.Open(chemin_liste, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]

    return labels_to_class_names


def read_annotations(chemin_liste):
    with tf.gfile.Open(chemin_liste, 'r') as f:
        lines = f.read()

    lines = lines.split('\n')

    return lines


class LabelsExtractor:
    def __init__(self, chemin_dataset):
        self.chemin_dataset = chemin_dataset

    def extract(self, chemin_annotations, chemin_liste_labels, chemin_csv):
        """Permet de récupérer les labels contenu dans un fichier d'annotations
            définit pour une image, et de les rassembler dans un fichier CSV

            Chaque ligne du fichier CSV correspond à l'association d'un label
            à une image. On peut donc avoir plusieurs fois la même image associée
            à des labels différents."""

        # Récupérer la liste de labels depuis le fichier txt
        labels_image = labels_to_class_name(chemin_liste_labels)
        inv_labels_image = {v: k for k, v in labels_image.items()}

        # Créer un fichier csv dans un dossier du projet
        with open(chemin_csv, "w") as fichier:
            fichier_csv = csv.writer(fichier)

            # Itérer sur le nbre d'images demandées (puis sur les labels) et
            # ajouter une ligne au fichier csv comprennant : hauteur et largeur
            # de l'image, nom de l'image, label.
            for image in os.listdir(self.chemin_dataset):

                if not image.startswith('.'):
                    # Vérifier qu'on va analyser un fichier txt
                    image_splited = image.split('.')
                    nom_image = image_splited[0]
                    extension_image = image_splited[1]

                    open_image = Image.open(os.path.join(self.chemin_dataset, image))
                    hauteur, largeur = open_image.size

                    chemin_txt = os.path.join(chemin_annotations, "{}.{}".format(nom_image, "txt"))

                    if os.path.isfile(chemin_txt):
                        # print(nom_image)

                        # Récupérer les labels associés à l'image
                        img_annots = read_annotations(chemin_txt)

                        for label in img_annots:
                            label_lower = label.lower()
                            if label_lower in inv_labels_image:
                                # Ecrire une ligne : nom image - extension - hauteur - largeur - label
                                fichier_csv.writerow([nom_image, extension_image, hauteur, largeur,
                                                      inv_labels_image[label_lower]])
                            elif len(label_lower) > 0:
                                # Avertissement : le label n'est pas renseigné dans la liste complète
                                print("Label posant problème : {} (image {} )".format(label_lower, image))
                                warnings.warn("\nAvertissement : le label de l'image n'est pas renseigné "
                                              "dans la liste des labels, \nil sera ignoré lors de la création du "
                                              "fichier csv")
