# import some common libraries
import numpy as np
import os, json
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from matplotlib import rcParams

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, Instances, RotatedBoxes, pairwise_iou

from myutils_train_earth import get_defilants
from myutils_inference_earth import split_image, filter_ponctuels, get_corners, get_polygons, get_iou_rotated
from myutils_inference_sideral import filter_boxes_rotated2

def get_dataset_all_classes(json_list, png_list, csv_list):
    dataset = []
    for idx, (json_filepath, image_filepath, csv_filepath) in enumerate(tqdm(zip(json_list, png_list, csv_list), total=len(json_list), desc='Chargement des datasets')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(json_filepath, image, 0, True)
        #image = get_stars(csv_filepath, image, 1)
        dataset.append(image)
    return dataset

def register_dataset_test(list_dict, classes, colors):
    DatasetCatalog.register('test', lambda : get_dataset_all_classes(list_dict['json_test'], list_dict['png_test'], list_dict['csv_test']))
    MetadataCatalog.get("test").set(thing_classes=classes, thing_colors=colors)
    return MetadataCatalog.get("test")

def run_inference(dataset_test, predictor, path_out, method):
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
                out = predictor(image_data['patches'].numpy()[i, j, ...])

                output['instances'].pred_boxes = out['instances'].pred_boxes.to('cpu')
                output['instances'].scores = out['instances'].scores
                output['instances'].pred_classes = out['instances'].pred_classes

                output['instances'].pred_boxes.tensor[:, 0] += (j * 384)
                output['instances'].pred_boxes.tensor[:, 1] += (i * 384)
                pred_boxes.append(output['instances'].pred_boxes)
                scores.append(output['instances'].scores)
                pred_classes.append(output['instances'].pred_classes)
        big_output['instances'].pred_boxes = RotatedBoxes.cat(pred_boxes)
        big_output['instances'].scores = torch.cat(scores)
        big_output['instances'].pred_classes = torch.cat(pred_classes)

        big_output = filter_boxes_rotated2(big_output, 0.5, 1)
        #big_output = filter_ponctuels(big_output, 0.5, 0.85)
        list_of_big_output.append(big_output)
        
    with open(os.path.join(path_out, 'output_inference_' + method.lower() + '.txt'), "wb") as fp:
        pickle.dump(list_of_big_output, fp)

def get_metrics_deep(dataset_test, path_out, iou_thresh, method):
    with open(os.path.join(path_out, 'output_inference_' + method.lower() + '.txt'), "rb") as fp:
        list_of_big_output = pickle.load(fp)
    metrics = {}
    metrics['nb_per_snr'] = np.zeros(11)
    metrics['tp_per_snr'] = np.zeros(11)
    metrics['true_positives'] = 0
    metrics['false_positives'] = 0
    metrics['false_negatives'] = 0
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

        snr_class = snr_t[class_ids == 0]
        ious_class = ious[class_ids == 0][:, pred_classes == 0]
        if torch.numel(ious_class) == 0:
            metrics['false_positives'] += ious_class.shape[1]
            metrics['false_negatives'] += ious_class.shape[0]
        else:
            metrics['true_positives'] += torch.any(ious_class > iou_thresh, dim=0).sum()
            metrics['false_positives'] += torch.all(ious_class < iou_thresh, dim=0).sum()
            metrics['false_negatives'] += torch.all(ious_class < iou_thresh, dim=1).sum()
        for i in range(10):
            ious_snr = ious_class[(snr_class > i) & (snr_class < i + 1)]
            metrics['nb_per_snr'][i] += len(ious_snr)
            metrics['tp_per_snr'][i] += torch.any(ious_snr > iou_thresh, dim=0).sum()
        ious_snr = ious_class[snr_class > 10]
        metrics['nb_per_snr'][10] += len(ious_snr)
        metrics['tp_per_snr'][10] += torch.any(ious_snr > iou_thresh, dim=0).sum()
            
    with open(os.path.join(path_out, 'metrics_' + method.lower() + '.txt'), "wb") as fp:
        pickle.dump(metrics, fp)

def run_bench_triton(dataset_test, iou_thresh, path_out, path_in):
    metrics = {}
    metrics['nb_per_snr'] = np.zeros(11)
    metrics['tp_per_snr'] = np.zeros(11)
    metrics['true_positives'] = 0
    metrics['false_positives'] = 0
    metrics['false_negatives'] = 0
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
        
        bbox_list = []
        class_id_list = []
        with open(os.path.join(path_in, d['file_name'].split('/')[-1].split('.')[0] + '.json')) as json_file:
            data = json.load(json_file)
            satellites = data['Objects']
            for satellite in satellites:
                x0, y0, x1, y1 = 0, 0, 0, 0
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
        
        ious = pairwise_iou(Boxes(gt_boxes), Boxes(pred_boxes))
        
        snr_class = snr_t[class_ids == 0]
        ious_class = ious[class_ids == 0][:, pred_classes == 0]
        if torch.numel(ious_class) == 0:
            metrics['false_positives'] += ious_class.shape[1]
            metrics['false_negatives'] += ious_class.shape[0]
        else:
            metrics['true_positives'] += torch.any(ious_class > iou_thresh, dim=0).sum()
            metrics['false_positives'] += torch.all(ious_class < iou_thresh, dim=0).sum()
            metrics['false_negatives'] += torch.all(ious_class < iou_thresh, dim=1).sum()
        for i in range(10):
            ious_snr = ious_class[(snr_class > i) & (snr_class < i + 1)]
            metrics['nb_per_snr'][i] += len(ious_snr)
            metrics['tp_per_snr'][i] += torch.any(ious_snr > iou_thresh, dim=0).sum()
        ious_snr = ious_class[snr_class > 10]
        metrics['nb_per_snr'][10] += len(ious_snr)
        metrics['tp_per_snr'][10] += torch.any(ious_snr > iou_thresh, dim=0).sum()
    
    with open(os.path.join(path_out, 'metrics_triton.txt'), "wb") as fp:
        pickle.dump(metrics, fp)

def write_metrics(path_out, iou_thresh, colors, methods):
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
    ratios = {methods[0]: np.zeros(11), methods[1]: np.zeros(11)}
    for index, (metrics, method) in enumerate(zip(metrics_list, methods)):
        ax[index].set_title(f'{method}'.title(), fontdict=fontdict)
        ax[index].set_xticks(range(len(labels)))
        ax[index].set_xticklabels(labels)
        ax[index].set_xlabel('SNR')
        ax[index].yaxis.set_major_formatter(PercentFormatter())
        ax[index].set_ylabel('True positive rate')
        with open(os.path.join(path_out, 'results_' + method + '.txt'), 'w+') as fd:
            precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
            recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
            f1_score = metrics['true_positives'] / (metrics['true_positives'] + 0.5 * (metrics['false_positives'] + metrics['false_negatives']))
            fd.write('satellite\n\n')
            fd.write('tp: {}\n'.format(metrics['true_positives']))
            fd.write('fp: {}\n'.format(metrics['false_positives']))
            fd.write('fn: {}\n'.format(metrics['false_negatives']))
            fd.write('precision: {}\n'.format(precision))
            fd.write('recall: {}\n'.format(recall))
            fd.write('F1 score: {}\n\n'.format(f1_score))
            for i in range(10):
                ratios[method][i] = (metrics['tp_per_snr'][i] / (metrics['nb_per_snr'][i] + 1e-20)) * 100
                fd.write(f'snr {i}-{i + 1}\n')
                fd.write('True positives rate: {}%\n\n'.format(np.nan_to_num(ratios[method][i])))
            ratios[method][10] = (metrics['tp_per_snr'][10] / metrics['nb_per_snr'][10]) * 100
            fd.write('snr 10+\n')
            fd.write('True positives rate: {}%\n\n'.format(ratios[method][10]))
            fd.write('\n\n')
            ax[index].plot(range(len(ratios[method])), np.nan_to_num(ratios[method]), ':', marker='o', color=tuple(t / 255 for t in colors[0]), label='satellite')
            ax[index].legend(loc="upper left")
        
    fig.savefig(os.path.join(path_out, 'plot.png'), dpi=fig.dpi)
    plt.show()