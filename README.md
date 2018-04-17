# Classification de visages
Projet de classification des visages (basé sur TF-Slim) réalisé pour le stage de 15 semaines.

La majorité du projet est basée sur le code provenant du GitHub de recherches sur TF-Slim de Tensorflow. Les éléments inutilisés ont été enlevés, et du code spécifique ajouté selon les besoins. Notamment, pour l'extraction des labels du dataset et la création d'un fichier TFRecords.

## Chargement du dataset et conversion en TfRecord
Le script download_and_convert_data.py permet de créer un fichier csv 
contenant toutes les associations images-annotations, si ce fichier 
n'existe pas pour le chemin indiqué. 

Le csv est ensuite utilisé afin de créer un fichier tfrecord qui 
pourra être lu par les scripts d'entrainement et d'évaluation. 

    --dataset_name "visages" 
    --dataset_dir "<PATH TO IMAGES DIR>" 
    --chemin_csv "labels/labels.csv" # Précisé par défaut dans le script
    --chemin_tfrecords "labels/labels_tfrecords.tfrecord" # Le fichier tfrecord qui sera créé
    --labels_dir "<PATH TO ANNOTATIONS DIR>" 
    --labels_list "labels/liste_labels.txt" # Précisé par défaut dans le script

#### Remarque :
Il est nécessaire d'exécuter ce script avant tout autre, car il permet 
de créer le fichier tfrecord, qui est abscent du repository. Cela 
signifie donc qu'il est essentiel d'indiquer un chemin vers un dataset
afin d'encoder les images dans le fichier tfrecord.

## Entrainement

Le script train_image_classifier.py permet de réaliser l'entrainement du réseau. Il est
nécessaire de lui indiquer les paramètres suivants : 
    
    --dataset_dir "images/" 
    --dataset_name "visages" 
    --model_name "inception_v3" # Permet la sélection de la fonction de préprocessing
    --max_number_of_steps 70 # Le nombre d'itération durant lesquelles le réseau
                             # va s'entrainer
    --clone_on_cpu True # A préciser uniquement si le script est exécuté sur CPU
    --batch_size 16 
    --train_image_size 250 # Taille de sortie des images (pour le redimensionnement)
    --train_dir "output/"
     
    # Les deux paramètres suivants ne sont nécessaire que si vous avez
    # précisé un nom différent de celui par défaut
    --path_to_csv "labels/labels.csv"
    --tfrecord_file "labels/labels_tfrecords.tfrecord"

#### Remarque :

Si on souhaite modifier l'architecture du réseau employée (par exemple par VGG16),
il faut modifier le code suivant :

    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, 
            num_classes=dataset.num_classes, is_training=False)

Il suffit de remplacer l'instruction inception_resnet_v2_arg_scope() par la méthode
correspondante dans l'autre architecture, et idem pour inception_resnet_v2().

`Initialement, une fonction générique (provenant du code de recherches de Tf-Slim) permettait de déterminer selon le 
paramètre "model_name" qu'elle méthodes employées, mais son utilisation
provoquait des erreurs d'exécution.`

## Evaluation

Le script eval_image_classifier.py permet de réaliser l'évaluation du réseau. Il est
nécessaire de lui indiquer les paramètres suivants : 

    --dataset_dir "images/" 
    --dataset_name "visages" 
    --model_name "inception_v3" # Permet la sélection de la fonction de préprocessing
    --eval_dir "output_eval/" 
    --checkpoint_path "output/" 
    --eval_image_size 250 # Taille de sortie des images (pour le redimensionnement)
    --batch_size 60
     
    # Les deux paramètres suivants ne sont nécessaire que si vous avez
    # précisé un nom différent de celui par défaut
    --path_to_csv "labels/labels.csv"
    --tfrecord_file "labels/labels_tfrecords.tfrecord"

#### Attention : 
Oublier le paramètre eval_image_size 250 peut donner lieu à l'apparition d'erreurs
lors de l'exécution du script indiquant un problème de dimensions des Tensors.


## Visualiser l'entrainement et l'évaluation

Afin d'afficher la progression du réseau durant l'exécution des scripts, 
il est possible d'employer Tensorboard en exécutant dans un terminal la commande

    # Les dossiers indiqués doivent correspondrent à ceux définit dans les
    # scripts comme identifiant les données d'entrainement et d'évaluation.
    # Ce sont ces dossiers qui contiennent les fichiers utilisés par tensorboard. 
    tensorboard --logdir=train:output/,eval:output_eval/
    
## Exporter le réseau

TODO, cette partie ne fonctionne pas encore.
