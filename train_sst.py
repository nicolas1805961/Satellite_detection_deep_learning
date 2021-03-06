# Some basic setup:
# Setup detectron2 logger
import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data import Metadata
import argparse
import yaml
import myutils_train_earth
import myutils_train_sideral
import myutils_train_seg_instance

the_parser = argparse.ArgumentParser("Train network for detection")
the_parser.add_argument('config_file', help="the config file to use for training")
args = the_parser.parse_args()

# Lecture du fichier de config et de ses champs
path_config = args.config_file
if path_config == None:
    path_config = 'config.yaml'
with open(path_config, 'r') as stream:
    config_dict = yaml.safe_load(stream)
    config_dict = config_dict['TRAINING']

path_in = config_dict['PATH_TRAIN']
path_out = config_dict['PATH_OUT']
mode = config_dict['MODE']
tool = config_dict['TOOL']
get_val_loss = config_dict['GET_VAL_LOSS']

# Detection par boite englobante
if tool == 'BB_DETECTION':
    list_of_files = myutils_train_earth.build_lists(path_in)
    # mode geostationnaire
    if mode == 'EARTH':
        classes = ['defilant', 'ponctuel', 'star']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        #list_of_files_new = myutils_train_earth.remove_small_mag(list_of_files) # Suppression du surplus d'exemples ayant une certaine magnitude
        list_dict = myutils_train_earth.build_list_dict(list_of_files)
        metadata_defilant = myutils_train_earth.register_datasets_defilant(list_dict)
        metadata_ponctuel = myutils_train_earth.register_datasets_ponctuel(list_dict)
        metadata_star = myutils_train_earth.register_datasets_star(list_dict) # Enregistrement du dataset des étoiles auprès de detectron2

        metadatas = Metadata()
        metadatas.thing_classes = classes
        metadatas.thing_colors = colors
        myutils_train_earth.display_classes(list_of_files, metadatas) # Sauvegarde d'une image de vérité terrain
    # Mode sidéral
    elif mode == 'SIDERAL':
        classes = ['defilant', 'star']
        colors = [(255, 0, 0), (0, 255, 0)]

        #list_of_files_new = myutils_train_sideral.remove_mag(list_of_files) 
        list_dict = myutils_train_earth.build_list_dict(list_of_files)
        metadata_defilant = myutils_train_earth.register_datasets_defilant(list_dict)
        metadata_star = myutils_train_earth.register_datasets_star(list_dict) 

        metadatas = Metadata()
        metadatas.thing_classes = classes
        metadatas.thing_colors = colors
        myutils_train_sideral.display_classes(list_of_files, metadatas) 
    else:
        raise ValueError('INPUT ERROR: MODE must be sideral or earth')

    cfgs = myutils_train_earth.get_cfgs(config_dict, path_out, classes) #chargement des configurations pour chaque classe
    myutils_train_earth.set_folder(cfgs, path_out) # Suppression du precedent dossier se trouvant à 'PATH_OUT'
    trainers = []
    for cfg_class in cfgs: # Lancement des entrainements pour chaque classe
        os.makedirs(cfg_class.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg_class.OUTPUT_DIR, 'cfg.txt'), 'w') as fd:
            fd.write(cfg_class.dump())
        trainer = myutils_train_earth.MyTrainer(cfg_class, get_val_loss)
        trainer.resume_or_load(resume=False)
        trainer.train()
        trainers.append(trainer)
# Mode segmentation d'instance
elif tool == 'SEG_INSTANCE':
    list_of_files = myutils_train_seg_instance.build_lists(path_in)
    list_dict = myutils_train_seg_instance.build_list_dict(list_of_files)
    metadata_defilant = myutils_train_seg_instance.register_datasets_defilant(list_dict) # Enregistrement du dataset defilant auprès de detectron2
    # Mode geostationnaire
    if mode == 'EARTH':
        classes = ['defilant', 'ponctuel', 'star']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        metadata_ponctuel = myutils_train_seg_instance.register_datasets_ponctuel(list_dict)
        metadata_star = myutils_train_seg_instance.register_datasets_etoile_earth(list_dict)

        metadatas = Metadata()
        metadatas.thing_classes = classes
        metadatas.thing_colors = colors

        myutils_train_seg_instance.display_classes_earth(list_of_files, metadatas) 
    # Mode sideral
    elif mode == 'SIDERAL':
        classes = ['defilant', 'star']
        colors = [(255, 0, 0), (0, 255, 0)]
        metadata_star = myutils_train_seg_instance.register_datasets_etoile_sideral(list_dict)

        #list_of_files_new = myutils_train_seg_instance.remove_mag(list_of_files)

        metadatas = Metadata()
        metadatas.thing_classes = classes
        metadatas.thing_colors = colors
        myutils_train_seg_instance.display_classes_sideral(list_of_files, metadatas)
    else:
        raise ValueError('INPUT ERROR: MODE must be sideral or earth')

    cfgs = myutils_train_earth.get_cfgs(config_dict, path_out, classes) 
    myutils_train_earth.set_folder(cfgs, path_out) 
    trainers = []
    for cfg_class in cfgs[::-1]:
        os.makedirs(cfg_class.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg_class.OUTPUT_DIR, 'cfg.txt'), 'w') as fd:
            fd.write(cfg_class.dump())
        trainer = myutils_train_seg_instance.MyTrainer(cfg_class, get_val_loss)
        trainer.resume_or_load(resume=False)
        trainer.train()
        trainers.append(trainer)
else:
    raise ValueError('INPUT ERROR: TOOL must be BB_DETECTION or SEG_INSTANCE')
