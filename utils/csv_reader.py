
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

                if len(elements_separes) == 5:
                    lignes_csv.append((elements_separes[0], elements_separes[1],
                                       elements_separes[2], elements_separes[3],
                                       elements_separes[4].split('\n')[0]))  # Enlever le \n de fin de ligne

            # print(lignes_csv)
        return lignes_csv