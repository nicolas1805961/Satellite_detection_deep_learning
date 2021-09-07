# import some common libraries
import os
import pickle
import argparse
import yaml

from myutils_train_earth import build_lists
from myutils_inference_earth import get_predictors
import myutils_benchmark_earth
import myutils_benchmark_sideral

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.data import DatasetCatalog

the_parser = argparse.ArgumentParser("Run benchmarks on test dataset")
the_parser.add_argument('config_file', help="the config file to use for benchmarks")
args = the_parser.parse_args()

#Lecture du fichier de config et sauvegarde des paramètres
path_config = args.config_file
if path_config == None:
    path_config = 'config.yaml'
with open(path_config, 'r') as stream:
    config_dict = yaml.safe_load(stream)
    config_dict = config_dict['BENCHMARKS']

path_in = config_dict['PATH_BENCH']
path_out = config_dict['PATH_OUT']
path_triton = config_dict['PATH_TRITON']
mode = config_dict['MODE']
weight_folder_1 = config_dict['WEIGHT_FOLDER_1']
weight_folder_2 = config_dict['WEIGHT_FOLDER_2']
method_1_name = config_dict['METHOD_1_NAME'].lower()
method_2_name = config_dict['METHOD_2_NAME'].lower()
method_name_list = [method_1_name, method_2_name]
weights_list = [weight_folder_1, weight_folder_2]

weights = None

list_of_files_test = build_lists(path_in)
list_dict = myutils_benchmark_earth.build_list_dict(list_of_files_test)

if mode == 'EARTH':
    classes = ['defilant', 'ponctuel', 'star']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    metadatas = myutils_benchmark_earth.register_dataset_test(list_dict, classes, colors) # Enregistrement du dataset de test auprès de Detectron2
    dataset_test = DatasetCatalog.get("test")

    for method_name, weight_path in zip(method_name_list, weights_list):
        # Si le nom d'une méthode est triton alors on a pas besoin de charger les poids et on effectue directement les calculs des benchmarks
        if method_name == 'triton':
            myutils_benchmark_earth.run_bench_triton(dataset_test, classes, 0.5, path_out, path_triton)
        # Sinon on charge les poids, on effectue l'inférence et enfin on mesure les performances
        else:
            with open(os.path.join(weight_path, "cfgs.txt"), "rb") as fp:
                cfgs = pickle.load(fp)
            weights_defilant = os.path.join(weight_path, 'output_defilant', 'model_final.pth')
            weights_ponctuel = os.path.join(weight_path, 'output_ponctuel', 'model_final.pth')
            weights_star = os.path.join(weight_path, 'output_star', 'model_final.pth')
            weights = [weights_defilant, weights_ponctuel, weights_star]

            predictors = get_predictors(cfgs, weights, config_dict)
            # Lancement de l'inférence sur le dataset de test
            myutils_benchmark_earth.run_inference(dataset_test, predictors, path_out, method_name)
            # Calcul des métriques
            myutils_benchmark_earth.get_metrics_deep(dataset_test, path_out, classes, 0.5, method_name)
elif mode == 'SIDERAL':
    classes = ['defilant', 'star']
    colors = [(255, 0, 0), (0, 255, 0)]

    metadatas = myutils_benchmark_sideral.register_dataset_test(list_dict, classes, colors)
    dataset_test = DatasetCatalog.get("test")

    for method_name, weight_path in zip(method_name_list, weights_list):
        if method_name == 'triton':
            myutils_benchmark_sideral.run_bench_triton(dataset_test, 0.5, path_out, path_triton)
        else:
            with open(os.path.join(weight_path, "cfgs.txt"), "rb") as fp:
                cfgs = pickle.load(fp)
            weights_defilant = os.path.join(weight_path, 'output_defilant', 'model_final.pth')
            weights_star = os.path.join(weight_path, 'output_star', 'model_final.pth')
            weights = [weights_defilant, weights_star]

            predictors = get_predictors(cfgs, weights, config_dict)
            myutils_benchmark_sideral.run_inference(dataset_test, predictors[0], path_out, method_name)
            myutils_benchmark_sideral.get_metrics_deep(dataset_test, path_out, 0.5, method_name)
else:
    raise('INPUT ERROR: MODE must be sideral or earth')

