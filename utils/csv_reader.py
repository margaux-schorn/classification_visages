
class CsvReader:

    @staticmethod
    def recuperer_lignes_csv(chemin_csv):
        """Cette méthode permet de récupérer le contenu d'un fichier
            csv et de place chaque élément d'une ligne dans un tuple.

            Returns : une liste de tuples"""
        lignes_csv = []

        with open(chemin_csv) as csv:
            for line in csv.readlines():
                elements_separes = line.split(",")

                if len(elements_separes) == 5: # le nombre d'images est donc ignoré
                    lignes_csv.append((elements_separes[0], elements_separes[1],
                                       elements_separes[2], elements_separes[3],
                                       elements_separes[4].split('\n')[0]))  # Enlever le \n de fin de ligne

            # print(lignes_csv)
        return lignes_csv

    @staticmethod
    def recuperer_derniere_ligne_csv(chemin_csv):
        """Cette méthode permet de récupérer la dernière ligne d'un fichier
            csv.
            Returns : un tuple"""

        with open(chemin_csv) as csv:
            for line in reversed(csv.readlines()):
                elements = line.split(",")

                if elements[0] == "nbre_images":
                    return elements[0], elements[1].split('\n')[0]
