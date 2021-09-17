Pour installer detectron:

Installer d'abord la version 1.7.1 de pytorch (sur mon pc les versions suivantes de pytorch ne fonctionnaient pas avec detectron)

    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

Ensuite Detectron peut être installé avec la commande suivante:

    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


Le fichier de config est divisé en 3 parties:

    - Training
    - Inference
    - Benchmarks/Graphs

Dans chacune des parties, des paramètres spécifiques à la tâche sont à spécifier.
Les benchmarks et graphs ne sont pas disponibles pour la segmentation d'instance




Script "train_sst.py":

usage: python train_sst.py config_file

Ce script permet de lancer l'entrainement.

config_file est un argument obligatoire indiquant le chemin jusqu'au fichier de config.

PATH_TRAIN dans le fichier de config indique le chemin jusqu'au dossier où se situent les données d'entrainement. Ce dossier doit contenir au moins 3 sous-dossiers: 

                        - 'png_data' contenant les images utilisées pour l'entrainement. Les images doivent être nommées i.png, où i appartient à [0, nombre d'images].

                        - 'csv_data' contenant les fichiers csv correspondant à chaque image. Les fichiers csv doivent être nommés i.csv, où i appartient à [0, nombre d'images].

                        - 'json_data' contenant les fichiers json correspondant à chaque image. Les fichiers json doivent être nommés i.json, où i appartient à [0, nombre d'images].

Dans le cas de la segmentation d'instance, un 4ème sous dossier doit être présent:

                        - 'label' contenant les images labélisées issues du simulateur. Les images doivent être nommées i.png, où i appartient à [0, nombre d'images].

Les 'i' de ces 3 ou 4 sous dossiers doivent correspondre. Par exemple 0.png (image), 0.csv (étoiles), 0.json (satellites) et 0.png (image labélisée) sont les données correspondant à la même image

Pour lancer l'entrainement sur seulement une ou deux classes, il est nécessaire de modifier la ligne 70 du script 'train_sst.py' pour la detection par boite englobante, ou la ligne 113 pour la segmentation d'instance. La modification doit être la suivante:

    - for cfg_class in cfgs[] --> indiquer dans les crochets sur quelle classe itérer. cfgs est une liste contenant les 3 ou 2 détecteurs selon le mode d'acquisition

Lorsque l'erreur 'inf/nan training has diverged' apparait, cela signifie que le learning rate est trop élevé, il doit être réduit.



Script "inference_real_image.py":

usage: python inference_real_image.py config_file

Ce script permet de réaliser l'inférence sur des images réelles.

config_file est un argument obligatoire indiquant le chemin jusqu'au fichier de config.

Dans le fichier de config, PATH_TEST indique le chemin jusqu'au dossier contenant les images sur lesquelles l'inférence sera faite. Toutes les images contenues dans ce dossier seront lues. Ici, les images n'ont pas besoin d'être numérotées d'une façon particulière.

Le champ WEIGHT_FOLDER dans le fichier de config indique le chemin jusqu'au dossier contenant les poids du modèle. Ce dossier doit contenir:

            - le fichier cfgs.txt écrit en binaire et qui contient les paramètres d'entrainement et de test.

            - le dossier 'output_defilant' contenant le fichier 'model_final.pth'. Il s'agit du fichier contenant les poids du modèle pour la detection des satellites défilants.

            - le dossier 'output_ponctuel' contenant le fichier 'model_final.pth'. Il s'agit du fichier contenant les poids du modèle pour la detection des satellites ponctuels.

            - le dossier 'output_star' contenant le fichier 'model_final.pth'. Il s'agit du fichier contenant les poids du modèle pour la detection des étoiles.


Les poids des modèles se situent dans le dossier "good_weights" sur /data_deep/SST_CNES_LOT2/



Script "run_benchmark.py":

usage: python run_benchmark.py config_file

Ce script permet de lancer le benchmark sur les données de test.

config_file est un argument obligatoire indiquant le chemin jusqu'au fichier de config.

PATH_BENCH dans le fichier de config indique le chemin jusqu'au dossier de test. Ce dossier sera utilisé pour réaliser les benchmarks des algorithmes de Deep Learning. La structure de ce dossier doit être la même que PATH_TRAIN (voir plus haut)

PATH_TRITON indique le chemin jusqu'au dossier contenant les fichiers json des résultats de Triton. Les fichiers json doivent être nommés i.json pour i appartenant à [0, nombre d'images]. Il doit y avoir correspondance avec les fichiers dans PATH_BENCH. Par exemple 0.json dans PATH_TRITON doit correspondre à 0.json, 0.csv et 0.png dans PATH_BENCH. Les numéros des fichiers dans ces 2 chemins doivent donc correspondre aux données de la même image de test (image dans PATH_BENCH donc).

Le champ WEIGHT_FOLDER dans le fichier de config indique le chemin jusqu'au dossier contenant les poids du modèle. Ce dossier doit contenir:

            - le fichier cfgs.txt écrit en binaire et qui contient les paramètres d'entrainement et de test.

            - le dossier 'output_defilant' contenant le fichier 'model_final.pth'. Il s'agit du fichier contenant les poids du modèle pour la detection des satellites défilants.

            - le dossier 'output_ponctuel' contenant le fichier 'model_final.pth'. Il s'agit du fichier contenant les poids du modèle pour la detection des satellites ponctuels.

            - le dossier 'output_star' contenant le fichier 'model_final.pth'. Il s'agit du fichier contenant les poids du modèle pour la detection des étoiles.

METHOD_1_NAME et METHOD_2_NAME correspondent au nom de chaque méthode comparée. Si l'une des méthodes comparée est triton alors le champ doit prendre la valeur 'TRITON'.

WEIGHT_FOLDER_1 correspond aux poids du modèle dont le nom est METHOD_1_NAME. Si METHOD_1_NAME prend la valeur 'TRITON' alors WEIGHT_FOLDER_1 est ignoré.

WEIGHT_FOLDER_2 correspond aux poids du modèle dont le nom est METHOD_2_NAME. Si METHOD_2_NAME prend la valeur 'TRITON' alors WEIGHT_FOLDER_2 est ignoré.



Script "graph.py":

usage: python graph.py config_file

Ce script permet d'obtenir le graph de comparaison entre les deux méthodes ainsi que les résultats quantitatifs dans des fichiers distinct pour chaque algorithme.

config_file est un argument obligatoire indiquant le chemin jusqu'au fichier de config.

Les statistiques utilisées pour generer les graphs et les fichiers de résultat sont celles calculés par le script 'run_benchmark.py'. Le script va chercher les statistiques dans le chemin PATH_OUT et sauvegarde les graphs au même endroit.

Ce script permettant de generer les graphs doit être exécuté APRES avoir executé le script "run_benchmark.py" qui permet de sauvegarder les résultats.

Les paramètres du fichier de config utilisés pour generer les graphs se situent dans la partie benchmark. Il s'agit de:

    - Le paramètre MODE qui doit prendre la même valeur pour la generation des graphs et l'execution des benchmarks (script run_benchmark.py executé précedemment)

    - Les paramètres METHOD_1_NAME et METHOD_2_NAME qui doivent prendre également la même valeur que celles utilisées lors de l'execution du script run_benchmark.py (voir partie script run_benchmark.py)

    - Le paramètre PATH_OUT qui indique le chemin où seront sauvegardés les graphs.




Il existe 2 paires de fichiers permettant de lancer les entrainements sur Polyaxon. 

Les fichiers polyaxon_create_image_scripts.yaml et polyaxon_run_scripts.yaml permettent de lancer les entrainements sur Polyaxon à partir des scripts python (py).

Les fichiers polyaxon_create_image_notebooks.yaml et polyaxon_run_notebooks.yaml permettent de lancer les entrainements sur Polyaxon à partir des notebooks (ipynb).

Dans les deux cas, le fichier polyaxon_create_image_*.yaml doit être executé avant le fichier polyaxon_run_*.yaml.


Dans le cas où les scripts doivent être executés sur Polyaxon, les chemins du fichier de config 'config.yaml' doivent commencer par '/data_deep/' et non par '/run/'.


Si les scripts doivent être exécutés sur Polyaxon, le code source ainsi que les fichiers yaml doivent être stockés dans un dossier sur /data_deep. Il faut ensuite indiquer dans 'polyaxon_run_script.yaml' le chemin jusqu'à ce dossier dans le paramètre 'context_folder'.
