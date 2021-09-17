# import some common libraries
import os
import pickle
import argparse
import yaml

import myutils_benchmark_earth
import myutils_benchmark_sideral

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

the_parser = argparse.ArgumentParser("Run benchmarks on test dataset")
the_parser.add_argument('config_file', help="the config file to use for benchmarks")
args = the_parser.parse_args()

path_config = args.config_file
if path_config == None:
    path_config = 'config.yaml'
with open(path_config, 'r') as stream:
    config_dict = yaml.safe_load(stream)
    config_dict = config_dict['BENCHMARKS']

path_out = config_dict['PATH_OUT']
mode = config_dict['MODE']
method_1_name = config_dict['METHOD_1_NAME'].lower()
method_2_name = config_dict['METHOD_2_NAME'].lower()

weights = None
method_name = [method_1_name, method_2_name]

if mode == 'EARTH':
    classes = ['defilant', 'ponctuel', 'star']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    myutils_benchmark_earth.write_metrics(classes, path_out, 0.5, colors, method_name) # Ecriture des métriques dans un fichier texte et création du graph de comparaison
elif mode == 'SIDERAL':
    classes = ['defilant', 'star']
    colors = [(255, 0, 0), (0, 255, 0)]

    myutils_benchmark_sideral.write_metrics(path_out, 0.5, colors, method_name)
else:
    raise ValueError('INPUT ERROR: MODE must be sideral or earth')

