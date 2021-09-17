# import some common libraries
import numpy as np
import os, json
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from matplotlib import rcParams
#import queue
#import multiprocessing as mp
#from multiprocessing import Process
#import time

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, Instances, RotatedBoxes, pairwise_iou

from myutils_train_earth import get_defilants, get_ponctuels
from myutils_inference_earth import split_image, filter_boxes_rotated2, filter_ponctuels, get_corners, get_polygons, get_iou_rotated

def build_list_dict(list_of_files_test):
    out = {}
    out['json_test'] = list_of_files_test[0]
    out['png_test'] = list_of_files_test[2]
    out['csv_test'] = list_of_files_test[1]
    return out

# Création du dataset de test dans un format attendu par Detectron2 (cf doc)
def get_dataset_all_classes(json_list, png_list, csv_list):
    dataset = []
    for idx, (json_filepath, image_filepath) in enumerate(tqdm(zip(json_list, png_list), total=len(json_list), desc='Chargement des datasets')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(json_filepath, image, 0, True)
        image = get_ponctuels(json_filepath, image, 1, True)
        dataset.append(image)
    return dataset

# Enregistrement du dataset de test auprès de Detectron2
def register_dataset_test(list_dict, classes, colors):
    DatasetCatalog.register('test', lambda : get_dataset_all_classes(list_dict['json_test'], list_dict['png_test'], list_dict['csv_test']))
    MetadataCatalog.get("test").set(thing_classes=classes, thing_colors=colors)
    return MetadataCatalog.get("test")


'''def runStart(q1, q2):
    #while not q1.empty():
    print(f'q1_size = {q1.qsize()}')
    work = q1.get()
    print('ok')
    # do whatever work you have to do on work
    big_output = (work[0], filter_boxes_rotated2(work[1], 0.5))
    print('ok2')
    print("Start thread : {}. Retrieved & Processed Item : {}".format(mp.current_process().name, work[0]))
    q2.put(big_output, block=True)
    #q1.task_done()
    #q2.put(None, block=True)

def runEnd(q2, q3):
    while not q2.empty():
        print(f'q2_size = {q2.qsize()}')
        work = q2.get(True)
        print('ok2')
        # do whatever work you have to do on work
        big_output = (work[0], filter_ponctuels(work[1], 0.5, 0.85))
        print("End thread : {}. Retrieved & Processed Item : {}".format(mp.current_process().name, work[0]))
        q3.put(big_output, block=True)
        #q2.task_done()'''

# Cette fonction permet d'effectuer l'inference sur le dataset de test en mode earth.
# Il s'agit de la même fonction que dans les fichiers 'myutils_inference_*'
def run_inference(dataset_test, predictors, path_out, method):
    list_of_big_output = []
    for d in tqdm(dataset_test, desc='Running inference'):
        image_data = split_image(d["file_name"])
        big_output = {'instances': Instances((2048, 2048))}
        pred_boxes = []
        scores = []
        pred_classes = []
        for i in range(5):
            for j in range(5):
                output = {'instances': Instances((512, 512))}
                pred_boxes_list = []
                scores_list = []
                pred_classes_list = []
                for idx, class_predictor in enumerate(predictors):
                    out = class_predictor(image_data['patches'].numpy()[i, j, ...])
                    if idx == 1 or idx == 2:
                        boxes = BoxMode.convert(out['instances'].pred_boxes.tensor.to('cpu'), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                        pred_boxes_list.append(RotatedBoxes(BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)))
                        pred_classes_list.append(torch.full(out['instances'].pred_classes.shape, idx, device='cpu'))
                    else:
                        pred_boxes_list.append(out['instances'].pred_boxes.to('cpu'))
                        pred_classes_list.append(out['instances'].pred_classes.to('cpu'))
                    scores_list.append(out['instances'].scores.to('cpu'))

                output['instances'].pred_boxes = RotatedBoxes.cat(pred_boxes_list)
                output['instances'].scores = torch.cat(scores_list)
                output['instances'].pred_classes = torch.cat(pred_classes_list)

                output['instances'].pred_boxes.tensor[:, 0] += (j * 384)
                output['instances'].pred_boxes.tensor[:, 1] += (i * 384)
                pred_boxes.append(output['instances'].pred_boxes)
                scores.append(output['instances'].scores)
                pred_classes.append(output['instances'].pred_classes)
        big_output['instances'].pred_boxes = RotatedBoxes.cat(pred_boxes)
        big_output['instances'].scores = torch.cat(scores).to('cpu')
        big_output['instances'].pred_classes = torch.cat(pred_classes).to('cpu')

        big_output = filter_boxes_rotated2(big_output, 0.5)
        big_output = filter_ponctuels(big_output, 0.5, 0.85)
        list_of_big_output.append(big_output)
        
    with open(os.path.join(path_out, 'output_inference_' + method + '.txt'), "wb") as fp:
        pickle.dump(list_of_big_output, fp)


# Cette fonction permet de calculer les métriques pour l'inférence effectuée avec un modèle de deep.
# Les métriques calculées sont: les vrais positifs, faux positifs, faux négatifs, 
# le nombre d'exemples par snr et le nombre de vrais positifs par snr.
def get_metrics_deep(dataset_test, path_out, classes, iou_thresh, method):
    # Lecture des inferences sauvegardées sur disque
    with open(os.path.join(path_out, 'output_inference_' + method + '.txt'), "rb") as fp:
        list_of_big_output = pickle.load(fp)
    metrics = {classes[0]: {}, classes[1]: {}}
    for key in metrics.keys():
        metrics[key]['nb_per_snr'] = np.zeros(11)
        metrics[key]['tp_per_snr'] = np.zeros(11)
        metrics[key]['true_positives'] = 0
        metrics[key]['false_positives'] = 0
        metrics[key]['false_negatives'] = 0
    for big_output, d in tqdm(zip(list_of_big_output, dataset_test), total=len(list_of_big_output), desc='Running deep benchmarks'):
        objs = []
        gt_boxes = torch.empty((0, 5)).to('cpu')
        class_id_list = []
        snr_list = []
        class_ids = torch.empty((0, 1)).to('cpu')
        snr_t = torch.empty((0, 1)).to('cpu')
        for obj in d['annotations']:
            objs.append(torch.tensor(BoxMode.convert(obj['bbox'], obj['bbox_mode'], BoxMode.XYWHA_ABS)).reshape(1, -1))
            class_id_list.append(obj['category_id'])
            snr_list.append(float(obj['snr']))
        if len(objs) > 0:
            gt_boxes = torch.cat(objs).to('cpu')
            class_ids = torch.tensor(class_id_list).to('cpu')
            snr_t = torch.tensor(snr_list).to('cpu')

        pred_classes = big_output['instances'].pred_classes.to('cpu')
        pred_boxes = big_output['instances'].pred_boxes.tensor.to('cpu')
        corners_preds = get_corners(pred_boxes)
        polys_preds = get_polygons(corners_preds)
        corners_gt = get_corners(gt_boxes)
        polys_gt = get_polygons(corners_gt)
        ious = get_iou_rotated(polys_gt, polys_preds)

        for idx, key in enumerate(metrics.keys()):
            snr_class = snr_t[class_ids == idx]
            ious_class = ious[class_ids == idx][:, pred_classes == idx]
            if torch.numel(ious_class) == 0:
                metrics[key]['false_positives'] += ious_class.shape[1]
                metrics[key]['false_negatives'] += ious_class.shape[0]
            else:
                metrics[key]['true_positives'] += torch.any(ious_class > iou_thresh, dim=0).sum()
                metrics[key]['false_positives'] += torch.all(ious_class < iou_thresh, dim=0).sum()
                metrics[key]['false_negatives'] += torch.all(ious_class < iou_thresh, dim=1).sum()
            for i in range(10):
                ious_snr = ious_class[(snr_class > i) & (snr_class < i + 1)]
                metrics[key]['nb_per_snr'][i] += len(ious_snr)
                metrics[key]['tp_per_snr'][i] += torch.any(ious_snr > iou_thresh, dim=0).sum()
            ious_snr = ious_class[snr_class > 10]
            metrics[key]['nb_per_snr'][10] += len(ious_snr)
            metrics[key]['tp_per_snr'][10] += torch.any(ious_snr > iou_thresh, dim=0).sum()
    
    # sauvegarde des métriques sur disque
    with open(os.path.join(path_out, 'metrics_' + method + '.txt'), "wb") as fp:
        pickle.dump(metrics, fp)

# Cette fonction permet de calculer les mêmes métriques que dans la fonction 'get metrics_deep' mais pour triton.
# Les résultats sont également stockés sur disque
def run_bench_triton(dataset_test, classes, iou_thresh, path_out, path_in):
    metrics = {classes[0]: {}, classes[1]: {}}
    for key in metrics.keys():
        metrics[key]['nb_per_snr'] = np.zeros(11)
        metrics[key]['tp_per_snr'] = np.zeros(11)
        metrics[key]['true_positives'] = 0
        metrics[key]['false_positives'] = 0
        metrics[key]['false_negatives'] = 0
    for d in tqdm(dataset_test, desc='Running Triton benchmarks'):
        objs = []
        gt_boxes = torch.empty((0, 4)).to('cpu')
        class_id_list = []
        snr_list = []
        class_ids = torch.empty((0, 1)).to('cpu')
        snr_t = torch.empty((0, 1)).to('cpu')
        for obj in d['annotations']:
            objs.append(torch.tensor(BoxMode.convert(obj['bbox'], obj['bbox_mode'], BoxMode.XYXY_ABS)).reshape(1, -1))
            class_id_list.append(obj['category_id'])
            snr_list.append(float(obj['snr']))
        if len(objs) > 0:
            gt_boxes = torch.cat(objs).to('cpu')
            class_ids = torch.tensor(class_id_list).to('cpu')
            snr_t = torch.tensor(snr_list).to('cpu')
        
        # Etant donné que triton ne donne pas ses résultats sous la forme de boites englobantes (cf PATH_TRITON),
        #  il est nécessaire de convertir les résultats.
        # J'ai choisis +/- 3 pixels de marge par rapport au centroid pour les satellites ponctuels
        bbox_list = []
        class_id_list = []
        with open(os.path.join(path_in, d['file_name'].split('/')[-1].split('.')[0] + '.json')) as json_file:
            data = json.load(json_file)
            satellites = data['Objects']
            for satellite in satellites:
                x0, y0, x1, y1 = 0, 0, 0, 0
                if satellite['Type'] == 'ponctuel':
                    class_id_list.append(1)
                    x0 = float(satellite['x1']) - 3
                    x1 = float(satellite['x1']) + 3
                    y0 = float(satellite['y1']) - 3
                    y1 = float(satellite['y1']) + 3
                elif satellite['Type'] == 'defilant':
                    class_id_list.append(0)
                    if float(satellite['dx']) < 0:
                        x0 = float(satellite['x0']) + float(satellite['dx'])
                        x1 = float(satellite['x0'])
                    else:
                        x0 = float(satellite['x0'])
                        x1 = float(satellite['x0']) + float(satellite['dx'])
                    if float(satellite['dy']) < 0:
                        y0 = float(satellite['y0']) + float(satellite['dy'])
                        y1 = float(satellite['y0'])
                    else:
                        y0 = float(satellite['y0'])
                        y1 = float(satellite['y0']) + float(satellite['dy'])
                bbox_list.append(torch.tensor([x0, y0, x1, y1]).reshape(1, -1))
        
        if len(bbox_list) > 0:
            pred_boxes = torch.cat(bbox_list).to('cpu')
            pred_classes = torch.tensor(class_id_list).to('cpu')
        
        # Calcul de l'IOU entre triton et la vérité terrain
        ious = pairwise_iou(Boxes(gt_boxes), Boxes(pred_boxes))
        
        for idx, key in enumerate(metrics.keys()):
            snr_class = snr_t[class_ids == idx]
            ious_class = ious[class_ids == idx][:, pred_classes == idx]
            if torch.numel(ious_class) == 0:
                metrics[key]['false_positives'] += ious_class.shape[1]
                metrics[key]['false_negatives'] += ious_class.shape[0]
            else:
                metrics[key]['true_positives'] += torch.any(ious_class > iou_thresh, dim=0).sum()
                metrics[key]['false_positives'] += torch.all(ious_class < iou_thresh, dim=0).sum()
                metrics[key]['false_negatives'] += torch.all(ious_class < iou_thresh, dim=1).sum()
            for i in range(10):
                ious_snr = ious_class[(snr_class > i) & (snr_class < i + 1)]
                metrics[key]['nb_per_snr'][i] += len(ious_snr)
                metrics[key]['tp_per_snr'][i] += torch.any(ious_snr > iou_thresh, dim=0).sum()
            ious_snr = ious_class[snr_class > 10]
            metrics[key]['nb_per_snr'][10] += len(ious_snr)
            metrics[key]['tp_per_snr'][10] += torch.any(ious_snr > iou_thresh, dim=0).sum()
    
    # Sauvegarde des résultats sur disque
    with open(os.path.join(path_out, 'metrics_triton.txt'), "wb") as fp:
        pickle.dump(metrics, fp)

# Cette fonction permet de générer le graph de comparaison ainsi que le fichier texte de résultats pour chaque classe
def write_metrics(classes, path_out, iou_thresh, colors, methods):
    fontdict = {'fontsize': 15,
 'fontweight' : rcParams['axes.titleweight'],
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}
    metrics_list = []
    for method in methods:
        with open(os.path.join(path_out, 'metrics_' + method + '.txt'), "rb") as fp:
            metrics_list.append(pickle.load(fp))
    labels = ['[0;1[', '[1;2[', '[2;3[', '[3;4[', '[4;5[', '[5;6[', '[6;7[', '[7;8[', '[8;9[', '[9;10[', '[10;max]']
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'True positive rate against SNR for IOU threshold at {iou_thresh}', y=1, fontsize=20)
    ratios = {methods[0]: {}, methods[1]: {}}
    for key in ratios.keys():
        ratios[key] = {classes[0]: np.zeros(11), classes[1]: np.zeros(11)}
    for index, (metrics, method) in enumerate(zip(metrics_list, methods)):
        ax[index].set_title(f'{method}'.title(), fontdict=fontdict)
        ax[index].set_xticks(range(len(labels)))
        ax[index].set_xticklabels(labels)
        ax[index].set_xlabel('SNR')
        ax[index].yaxis.set_major_formatter(PercentFormatter())
        ax[index].set_ylabel('True positive rate')
        with open(os.path.join(path_out, 'results_' + method + '.txt'), 'w+') as fd:
            for idx, key in enumerate(metrics.keys()):
                precision = metrics[key]['true_positives'] / (metrics[key]['true_positives'] + metrics[key]['false_positives'])
                recall = metrics[key]['true_positives'] / (metrics[key]['true_positives'] + metrics[key]['false_negatives'])
                f1_score = metrics[key]['true_positives'] / (metrics[key]['true_positives'] + 0.5 * (metrics[key]['false_positives'] + metrics[key]['false_negatives']))
                fd.write(key + '\n\n')
                fd.write('tp: {}\n'.format(metrics[key]['true_positives']))
                fd.write('fp: {}\n'.format(metrics[key]['false_positives']))
                fd.write('fn: {}\n'.format(metrics[key]['false_negatives']))
                fd.write('precision: {}\n'.format(precision))
                fd.write('recall: {}\n'.format(recall))
                fd.write('F1 score: {}\n\n'.format(f1_score))
                for i in range(10):
                    ratios[method][key][i] = (metrics[key]['tp_per_snr'][i] / (metrics[key]['nb_per_snr'][i] + 1e-20)) * 100
                    fd.write(f'snr {i}-{i + 1}\n')
                    fd.write('True positives rate: {}%\n\n'.format(ratios[method][key][i]))
                ratios[method][key][10] = (metrics[key]['tp_per_snr'][10] / (metrics[key]['nb_per_snr'][10] + 1e-20)) * 100
                fd.write('snr 10+\n')
                fd.write('True positives rate: {}%\n\n'.format(ratios[method][key][10]))
                fd.write('\n\n')
                ax[index].plot(range(len(ratios[method][key])), np.nan_to_num(ratios[method][key]), ':', marker='o', color=tuple(t / 255 for t in colors[idx]), label=key)
                ax[index].legend(loc="upper left")
        
    fig.savefig(os.path.join(path_out, 'plot.png'), dpi=fig.dpi)
    plt.show()

def write_metrics_comparison(classes, path_backbone, path_deep, path_out, iou_thresh):
    fontdict = {'fontsize': 15,
 'fontweight' : rcParams['axes.titleweight'],
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}
    with open(os.path.join(path_backbone, 'metrics_deep.txt'), "rb") as fp:
        metrics_backbone = pickle.load(fp)
    with open(os.path.join(path_deep, 'metrics_deep.txt'), "rb") as fp:
        metrics_deep = pickle.load(fp)
    labels = ['[0;1[', '[1;2[', '[2;3[', '[3;4[', '[4;5[', '[5;6[', '[6;7[', '[7;8[', '[8;9[', '[9;10[', '[10;max]']
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    #fig.suptitle(f'True positive rate against SNR for IOU threshold at {iou_thresh}', y=0.98, fontsize=25)
    ratios = {'deep': {}, 'backbone': {}}
    for key in ratios.keys():
        ratios[key] = {classes[1]: np.zeros(11)}
    for index, (metrics, method, col) in enumerate(zip([metrics_deep, metrics_backbone], ['deep', 'backbone'], ['g', 'b'])):
        ax.set_title(f'True positive rate against SNR for IOU threshold at {iou_thresh}', fontdict=fontdict)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('SNR')
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel('True positive rate')
        with open(os.path.join(path_out, 'results_' + method + '.txt'), 'w+') as fd:
            precision = metrics[classes[1]]['true_positives'] / (metrics[classes[1]]['true_positives'] + metrics[classes[1]]['false_positives'])
            recall = metrics[classes[1]]['true_positives'] / (metrics[classes[1]]['true_positives'] + metrics[classes[1]]['false_negatives'])
            f1_score = metrics[classes[1]]['true_positives'] / (metrics[classes[1]]['true_positives'] + 0.5 * (metrics[classes[1]]['false_positives'] + metrics[classes[1]]['false_negatives']))
            fd.write(classes[1] + '\n\n')
            fd.write('tp: {}\n'.format(metrics[classes[1]]['true_positives']))
            fd.write('fp: {}\n'.format(metrics[classes[1]]['false_positives']))
            fd.write('fn: {}\n'.format(metrics[classes[1]]['false_negatives']))
            fd.write('precision: {}\n'.format(precision))
            fd.write('recall: {}\n'.format(recall))
            fd.write('F1 score: {}\n\n'.format(f1_score))
            for i in range(10):
                ratios[method][classes[1]][i] = (metrics[classes[1]]['tp_per_snr'][i] / metrics[classes[1]]['nb_per_snr'][i]) * 100
                fd.write(f'snr {i}-{i + 1}\n')
                fd.write('True positives rate: {}%\n\n'.format(ratios[method][classes[1]][i]))
            ratios[method][classes[1]][10] = (metrics[classes[1]]['tp_per_snr'][10] / metrics[classes[1]]['nb_per_snr'][10]) * 100
            fd.write('snr 10+\n')
            fd.write('True positives rate: {}%\n\n'.format(ratios[method][classes[1]][10]))
            fd.write('\n\n')
            ax.plot(range(len(ratios[method][classes[1]])), np.nan_to_num(ratios[method][classes[1]]), ':', marker='o', color=col, label='ponctuel_' + method)
            ax.legend(loc="upper left")
        
    fig.savefig(os.path.join(path_out, 'plot.png'), dpi=fig.dpi)
    plt.show()