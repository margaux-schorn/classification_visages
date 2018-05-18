# Classification de visages
Projet de classification des visages (basé sur TF-Slim) réalisé pour le stage de 15 semaines.

La majorité du projet est basée sur le code provenant du GitHub de recherches sur TF-Slim 
de Tensorflow. Les éléments inutilisés ont été enlevés, et du code spécifique ajouté 
selon les besoins. Notamment, pour l'extraction des labels du dataset et la création 
d'un fichier TFRecords.

## Chargement du dataset et conversion en TfRecord
Le script download_and_convert_data.py permet de créer un fichier csv 
contenant toutes les associations images-annotations, si ce fichier 
n'existe pas pour le chemin indiqué. 

Le csv est ensuite utilisé afin de créer un fichier tfrecord qui 
pourra être lu par les scripts d'entrainement et d'évaluation. 

    --image_dir "<PATH TO IMAGES DIR>" 
    --chemin_csv "labels/labels.csv" # Précisé par défaut dans le script
    --chemin_tfrecords "labels/labels_tfrecords.tfrecord" # Le fichier tfrecord qui sera créé
    --labels_dir "<PATH TO ANNOTATIONS DIR>" 
    --classes "labels/liste_labels.txt" # Précisé par défaut dans le script
    
#### Remarque :
Il est nécessaire d'exécuter ce script avant tout autre, car il permet 
de créer le fichier tfrecord, qui est abscent du repository. Cela 
signifie donc qu'il est essentiel d'indiquer un chemin vers un dataset
afin d'encoder les images dans le fichier tfrecord.

## Entrainement

Le script train_image_classifier.py permet de réaliser l'entrainement du réseau. Il est
nécessaire de lui indiquer les paramètres suivants : 
    
    --model_name "inception_v3" # Permet la sélection de l'architecture et de la fonction de préprocessing
    --max_number_of_steps 100 # Le nombre d'itération durant lesquelles le réseau va s'entrainer
    --clone_on_cpu True # A préciser uniquement si le script est exécuté sur CPU
    --train_dir "output/"
    --batch_size 8 
    --learning_rate=0.0001 
    --end_learning_rate=0.000001
    --train_image_size 250 # Taille de sortie des images (pour le redimensionnement)
    # Il semblerait que se dernier passer de ce paramètre permet d'obtenir 
    de meilleures performances
    
##### Si on souhaite partir d'un modèle pré-entrainé, il faut ajouter les paramètres suivants :
    --checkpoint_path="/Users/margaux/PycharmProjects/inception_v3.ckpt"
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
     
##### Si vous avez précisé un nom différent pour les fichiers csv et tfrecord, il faut le spécifier via les paramètres suivants :
    # précisé un nom différent de celui par défaut
    --path_to_csv "labels/labels.csv"
    --tfrecord_file "labels/labels_tfrecords.tfrecord"

#### Remarque :

Si vous souhaitez modifier l'architecture du réseau employée (par exemple par VGG16),
vous pouvez récupérer les fichiers dans le dossier "nets" du projet de recherches de TF-Slim
(voir https://github.com/tensorflow/models/tree/master/research/slim). Il suffit alors
de modifier le paramètre d'exécution. 

## Evaluation

Le script eval_image_classifier.py permet de réaliser l'évaluation du réseau. Il est
nécessaire de lui indiquer les paramètres suivants : 

    --model_name "inception_v3" # Permet la sélection de l'architecture et de la fonction de préprocessing
    --eval_dir "output_eval/" 
    --checkpoint_path "output/" 
    --batch_size 8
    --eval_image_size 250 # Taille de sortie des images (pour le redimensionnement)
    # Il semblerait que se dernier passer de ce paramètre permet d'obtenir 
    de meilleures performances
     
##### Si vous avez précisé un nom différent pour les fichiers csv et tfrecord, il faut le spécifier via les paramètres suivants :
    --path_to_csv "labels/labels.csv"
    --tfrecord_file "labels/labels_tfrecords.tfrecord"

#### Remarque :

Si vous souhaitez modifier l'architecture du réseau employée (par exemple par VGG16),
vous pouvez récupérer les fichiers dans le dossier "nets" du projet de recherches de TF-Slim
(voir https://github.com/tensorflow/models/tree/master/research/slim). Il suffit alors
de modifier le paramètre d'exécution. 

#### Attention : 
Oublier le paramètre eval_image_size 250, alors que vous avez précisé train_image_size 250
peut donner lieu à l'apparition d'erreurs lors de l'exécution du script. 
Elle indiquera alors un problème de dimensions des Tensors.
En effet, ce paramètre permet de donner une taille cible à la fonction
de preprocessing, afin de redimensionner les images. 

## Visualiser l'entrainement et l'évaluation

Afin d'afficher la progression du réseau durant l'exécution des scripts, 
il est possible d'employer Tensorboard en exécutant dans un terminal la commande :

    # Les dossiers indiqués doivent correspondrent à ceux définit dans les
    # scripts comme identifiant les données d'entrainement et d'évaluation.
    # Ce sont ces dossiers qui contiennent les fichiers utilisés par tensorboard. 
    tensorboard --logdir=train:output/,eval:output_eval/
    
## Exporter le réseau

Quand vous souhaitez exporter le réseau, il y a 3 étapes à accomplir : 

_1) Compiler la commande permettant de geler le graphe depuis le code source de 
Tensorflow. Pour cela, on utilise la commande suivante :_
    
    bazel build tensorflow/python/tools:freeze_graph
    
_2) Récupérer la structure du réseau. Cette partie consiste à exécuter le script présent
dans le projet "export_inference_graph". De cette façon, l'architecture est enregistrée 
dans un fichier '.pb'. Les paramètres à préciser sont les suivants :_

    --output_file "inception_v3_inf_graph.pb" # le fichier qui sera généré
    --batch_size 8 
    --csv_file "labels/labels.csv" 
    --record_file "labels/labels_records.tfrecord" 
    --classes_file "labels/liste_labels.txt" 

_3) Avec la commande compilée, on peut exécuter l'export comme suit (et ainsi récupérer
le graphe enrichit des résultats de l'entrainement) :_
    
    bazel-bin/tensorflow/python/tools/freeze_graph 
    # On indique le chemin vers graphe créé à l'étape 2
    --input_graph=/Users/margaux/classification_visages/inception_v3_inf_graph.pb 
    --input_binary=true 
    --input_checkpoint=/Users/margaux/Documents/output/model.ckpt-1000 # Le dernier checkpoint de l'entrainement
    --output_graph=/Users/margaux/Documents/inception_v3_frozen_graph.pb # Chemin où sera ajouté le frozen model
    --output_node_names=InceptionV3/Predictions/Reshape_1 # Il s'agit du noeud de sortie du réseau
    # Ce noeud change selon l'architecture choisie (ici Inception)

## Tester le modèle exporté

Si vous souhaitez tester rapidement que votre frozen model fonctionne bien, suivez la
procédure suivante :

_1) Compiler la commande permettant tester le modèle depuis le code source de 
Tensorflow. Pour cela, on utilise la commande suivante :_

    bazel build tensorflow/examples/label_image:label_image
    
_2) Vous pouvez ensuite obtenir les prédictions d'une image en exécutant une commande
semblable à :_

    bazel-bin/tensorflow/examples/label_image/label_image 
    --image= <PATH TO IMAGE> 
    --input_layer=input # il s'agit du nom que j'ai donné au noeud d'entrée du réseau
    --output_layer=InceptionV3/Predictions/Reshape_1 # noeud de sortie du réseau
    --graph=/Users/margaux/Documents/inception_v3_frozen_graph.pb # chemin vers le frozen model
    --labels=/Users/margaux/datasets/visages_test/liste_labels.txt # chemin vers la liste de labels
    (sans l'association à un nombre)
