# Some basic setup:
# Setup detectron2 logger
import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data import Metadata
import argparse
import yaml
import myutils_inference_earth
import myutils_inference_sideral
import myutils_inference_seg_instance
import pickle

the_parser = argparse.ArgumentParser("Run inference on one big image")
the_parser.add_argument('config_file', help="the config file to use for inference")
args = the_parser.parse_args()

path_config = args.config_file
if path_config == None:
    path_config = 'config.yaml'
with open(path_config, 'r') as stream:
    config_dict = yaml.safe_load(stream)
    config_dict = config_dict['INFERENCE']

path_in = config_dict['PATH_TEST']
path_out = config_dict['PATH_OUT']
mode = config_dict['MODE']
tool = config_dict['TOOL']
percent_to_keep = config_dict['STAR']['PERCENT_TO_KEEP']
display_labels = config_dict['DISPLAY_LABELS']
display_lines = config_dict['DISPLAY_LINES']
weight_folder = config_dict['WEIGHT_FOLDER']

weights = None

if tool == 'BB_DETECTION':
    if mode == 'EARTH':
        metadatas = Metadata()
        metadatas.thing_classes = ['defilant', 'ponctuel', 'star']
        metadatas.thing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # Création d'une liste contenant les poids des modèles pour chaque classe
        with open(os.path.join(weight_folder, "cfgs.txt"), "rb") as fp:
            cfgs = pickle.load(fp)
            weights_defilant = os.path.join(weight_folder, 'output_defilant', 'model_final.pth')
            weights_ponctuel = os.path.join(weight_folder, 'output_ponctuel', 'model_final.pth')
            weights_star = os.path.join(weight_folder, 'output_star', 'model_final.pth')
            weights = [weights_defilant, weights_ponctuel, weights_star]

        predictors = myutils_inference_earth.get_predictors(cfgs, weights, config_dict) # Création des objets predictors de detectron2 à partir des poids pour chaque classe
        list_of_files_test = myutils_inference_earth.build_lists(path_in)
        myutils_inference_earth.get_infered_image(list_of_files_test, predictors, metadatas, path_out, display_lines, display_labels) # Fonction qui permet d'effectuer l'inférence sur des images présentes dans un dossier
    elif mode == 'SIDERAL':
        metadatas = Metadata()
        metadatas.thing_classes = ['defilant', 'star']
        metadatas.thing_colors = [(255, 0, 0), (0, 255, 0)]

        with open(os.path.join(weight_folder, "cfgs.txt"), "rb") as fp:
            cfgs = pickle.load(fp)
            weights_defilant = os.path.join(weight_folder, 'output_defilant', 'model_final.pth')
            weights_star = os.path.join(weight_folder, 'output_star', 'model_final.pth')
            weights = [weights_defilant, weights_star]

        predictors = myutils_inference_earth.get_predictors(cfgs, weights, config_dict)
        list_of_files_test = myutils_inference_earth.build_lists(path_in)
        myutils_inference_sideral.get_infered_image(list_of_files_test, predictors, metadatas, path_out, display_lines, display_labels)
    else:
        raise ValueError('INPUT ERROR: MODE must be sideral or earth')
elif tool == 'SEG_INSTANCE':
    metadatas = Metadata()
    if mode == 'EARTH':
        metadatas.thing_classes = ['defilant', 'ponctuel', 'star']
        metadatas.thing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        with open(os.path.join(weight_folder, "cfgs.txt"), "rb") as fp:
            cfgs = pickle.load(fp)
            weights_defilant = os.path.join(weight_folder, 'output_defilant', 'model_final.pth')
            weights_star = os.path.join(weight_folder, 'output_star', 'model_final.pth')
            weights_ponctuel = os.path.join(weight_folder, 'output_ponctuel', 'model_final.pth')
            weights = [weights_defilant, weights_ponctuel, weights_star]
    elif mode == 'SIDERAL':
        metadatas.thing_classes = ['defilant', 'star']
        metadatas.thing_colors = [(255, 0, 0), (0, 255, 0)]

        with open(os.path.join(weight_folder, "cfgs.txt"), "rb") as fp:
            cfgs = pickle.load(fp)
            weights_defilant = os.path.join(weight_folder, 'output_defilant', 'model_final.pth')
            weights_star = os.path.join(weight_folder, 'output_star', 'model_final.pth')
            weights = [weights_defilant, weights_star]
    
    predictors = myutils_inference_earth.get_predictors(cfgs, weights, config_dict)
    list_of_files_test = myutils_inference_earth.build_lists(path_in)
    myutils_inference_seg_instance.get_infered_image(list_of_files_test, predictors, metadatas, path_out, display_lines, display_labels, percent_to_keep)
