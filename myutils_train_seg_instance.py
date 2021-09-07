# import some common libraries
import pandas as pd
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
from math import sqrt, ceil, atan2, degrees
import json
from sklearn.cluster import KMeans
from astropy.io import fits
from sklearn.model_selection import train_test_split
import shutil
import time
import torch
from torch.nn import functional as F
import pickle
import copy
from typing import Dict, List
from pathlib import Path
from shapely.geometry import Polygon
import math
from math import radians
from tqdm import tqdm
from sympy.solvers import solve
from sympy import Symbol

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.data.transforms import RandomFlip
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase, LRScheduler, hooks
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, _create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, Metadata, build_detection_train_loader, DatasetMapper
from detectron2.structures import BoxMode, Boxes, Instances, RotatedBoxes, PolygonMasks, BitMasks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats
from detectron2.layers import cat, ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, PROPOSAL_GENERATOR_REGISTRY, StandardROIHeads, build_model
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.rrpn import RRPN
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.rotated_fast_rcnn import RROIHeads
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import _dense_box_regression_loss
import detectron2.utils.comm as comm
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.colormap import random_color

def get_gt_satellite(json_filepath, png_filepath, label_filepath):
    label_image = cv2.imread(label_filepath, cv2.IMREAD_GRAYSCALE)
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_defilants(label_image, json_filepath, image, 0)
    return image

def get_gt_star(csv_filepath, png_filepath, label_filepath):
    label_image = cv2.imread(label_filepath, cv2.IMREAD_GRAYSCALE)
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_stars(label_image, csv_filepath, image, 0)
    return image

def get_angle(satellite):
    delta_x = satellite['x0'] - satellite['x']
    delta_y = satellite['y'] - satellite['y0']
    return degrees(atan2(delta_y, delta_x))

# Cette fonction permet D'obtenir les 4 coins d'une boite orientée sachant que
# l'on dispose pour une boite, de données sous la forme: [x, y, w, h, a]
# x et y sont les coordonnées du point supérieur gauche
# w et h sont la largeur et la hauteur
# a est l'angle d'orientation dans le sens inverse des aiguilles d'une montre
def get_corners(rotated_boxes):
    x = rotated_boxes[:, 0]
    y = rotated_boxes[:, 1]
    w = rotated_boxes[:, 2]
    l = rotated_boxes[:, 3]
    yaw = rotated_boxes[:, 4] * math.pi / 180.0
    device = rotated_boxes.device
    bev_corners = torch.zeros((len(rotated_boxes), 4, 2), dtype=torch.float, device=device)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # top left
    bev_corners[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[:, 0, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # bottom left
    bev_corners[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[:, 1, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    # bottom right
    bev_corners[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[:, 2, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # top right
    bev_corners[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[:, 3, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    
    return bev_corners

def get_seg_json(label_img, img_corners, label_value):
    black_img = np.zeros(label_img.shape, dtype=np.uint8)
    converted = img_corners.numpy().astype('int32')
    mask = cv2.fillPoly(black_img, [converted], 1).astype('bool')
    black_img[mask] = label_img[mask]
    temp = np.copy(black_img)
    black_img[black_img != label_value] = 0
    contours, hierarchy = cv2.findContours(black_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    indices_to_keep_nb_points = []
    for index, contour in enumerate(contours):
        contour = contour.reshape(-1, 2)
        if len(contour) < 3:
            continue
        indices_to_keep_nb_points.append(index)
    contours = [contours[i] for i in indices_to_keep_nb_points]
    list_of_poly = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        poly = Polygon([tuple(x) for x in contour])
        list_of_poly.append(poly.area)
    if len(list_of_poly) == 0:
        #print(np.unique(label_img))
        #print(np.unique(temp))
        #cv2.imwrite('/data_deep/SST_CNES_LOT2/mask_sideral.png', mask.astype(np.uint8) * 255)
        raise('Number of points in satellite contour < 3 !')
    else:
        largest_contour = contours[np.argmax(list_of_poly)]
    return [largest_contour.flatten().tolist()]

def get_seg_csv(label_img, img_corners, label_value):
    contour_list = []
    indices_to_remove = []
    for idx, obj_corners in enumerate(img_corners):
        black_img = np.zeros(label_img.shape, dtype=np.uint8)
        converted = obj_corners.numpy().astype('int32')
        mask = cv2.fillPoly(black_img, [converted], 1).astype('bool')
        black_img[mask] = label_img[mask]
        black_img[(black_img != label_value) & (black_img != 0)] = label_value
        if black_img.sum() == 0:
            indices_to_remove.append(idx)
            continue
        #plt.imshow(black_img, cmap='gray')
        #plt.show()
        contours, hierarchy = cv2.findContours(black_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        indices_to_keep_nb_points = []
        for index, contour in enumerate(contours):
            contour = contour.reshape(-1, 2)
            if len(contour) < 3:
                continue
            indices_to_keep_nb_points.append(index)
        contours = [contours[i] for i in indices_to_keep_nb_points]
        list_of_poly = []
        for contour in contours:
            contour = contour.reshape(-1, 2)
            poly = Polygon([tuple(x) for x in contour])
            list_of_poly.append(poly.area)
        if len(list_of_poly) == 0:
            indices_to_remove.append(idx)
        else:
            contour_list.append(contours[np.argmax(list_of_poly)])
    #print(f'nb of contours in function: {len(contour_list)}')
    return [[x.flatten().tolist()] for x in contour_list], indices_to_remove

def get_defilants(label_img, json_filepath, image, category_id, test=False):
    with open(json_filepath) as json_file:
        objs = []
        data = json.load(json_file)
        satellites = data['Objects']
        for satellite in satellites:
            obj = {"bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = category_id
            if test == False:
                angle = get_angle(satellite)
                second_point_coords = (satellite['x0'] + satellite['dx'], satellite['y0'] + satellite['dy'])
                width = sqrt((satellite['x0'] - second_point_coords[0])**2 + (satellite['y0'] - second_point_coords[1])**2) + 4
                height = -4.878 * satellite['mag'] + 70.0975
                boxes = [satellite['x'], satellite['y'], width, height, angle]
                corners = get_corners(torch.tensor(boxes).reshape(1, -1))
                obj['segmentation'] = get_seg_json(label_img, corners[0], 254)
            margin = 4
            width = satellite['dx'] + 2 * margin
            height = satellite['dy'] + 2 * margin
            obj["bbox"] = [satellite['x0'] - margin, satellite['y0'] - margin, width, height]
            objs.append(obj)
        image["annotations"].extend(objs)
    return image

def get_stars(label_img, csv_filepath, image, category_id, test=False):
    df = pd.read_csv(csv_filepath, sep=';', usecols=['X0', 'Y0', 'X1', 'Y1', 'Gmag'])
    df['bbox_mode'] = BoxMode.XYWH_ABS
    df['category_id'] = category_id
    margin = ((800 * np.exp(-0.4 * df['Gmag']) + 6) / 2).round().astype(int)
    width = 2 * margin
    height = 2 * margin
    X = (df['X1'] - margin)
    Y = (df['Y1'] - margin)
    bbox = pd.concat([X, Y, width, height], axis=1)
    df['bbox'] = bbox.values.tolist()
    if test == False:
        rotated_bbox = torch.from_numpy(pd.concat([df['X1'], df['Y1'], width, height, pd.Series(np.zeros(len(width)))], axis=1).values)
        corners = get_corners(rotated_bbox)
        objs_contours, indices_to_remove = get_seg_csv(label_img, corners, 127)
        assert len(objs_contours) + len(indices_to_remove) == len(df)
        df.drop(df.index[indices_to_remove], inplace=True)
        df['segmentation'] = objs_contours
    df.drop(['X0','Y0', 'X1', 'Y1', 'Gmag'], axis=1, inplace=True)
    objs = df.to_dict(orient='records')
    image["annotations"].extend(objs)
    return image

def get_dataset_all_classes(json_list, png_list, csv_list, label_list):
    dataset = []
    for idx, (json_filepath, image_filepath, csv_filepath, label_filepath) in enumerate(tqdm(zip(json_list, png_list, csv_list, label_list), total=len(json_list), desc='Chargement des datasets')):
        label_image = cv2.imread(label_filepath, cv2.IMREAD_GRAYSCALE)
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(label_image, json_filepath, image, 0, True)
        image = get_stars(label_image, csv_filepath, image, 1, True)
        dataset.append(image)
    return dataset

def get_dataset_satellites(json_list, png_list, label_list):
    dataset = []
    for idx, (json_filepath, image_filepath, label_filepath) in enumerate(tqdm(zip(json_list, png_list, label_list), total=len(json_list), desc='Chargement du dataset satellite')):
        label_image = cv2.imread(label_filepath, cv2.IMREAD_GRAYSCALE)
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(label_image, json_filepath, image, 0)
        dataset.append(image)
    return dataset

def get_dataset_etoiles(csv_list, png_list, label_list):
    dataset = []
    for idx, (csv_filepath, image_filepath, label_filepath) in enumerate(tqdm(zip(csv_list, png_list, label_list), total=len(csv_list), desc='Chargement du dataset etoile')):
        label_image = cv2.imread(label_filepath, cv2.IMREAD_GRAYSCALE)
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_stars(label_image, csv_filepath, image, 0)
        dataset.append(image)
    return dataset

def build_lists(path):
    json_list = sorted(glob(os.path.join(path, 'json_data', '*')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    csv_list = sorted(glob(os.path.join(path, 'csv_data', '*')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    png_list = sorted(glob(os.path.join(path, 'png_data', '*')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    label_list = sorted(glob(os.path.join(path, 'label', '*')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    print(len(json_list))
    print(len(csv_list))
    print(len(png_list))
    print(len(label_list))
    return [json_list, csv_list, png_list, label_list]

def remove_mag(list_of_files):
    file_idx_to_remove = []
    list_of_files_new = []
    for idx, filepath in enumerate(list_of_files[0]):
        with open(filepath) as json_file:
            data = json.load(json_file)
            satellites = data['Objects']
            for satellite in satellites:
                if float(satellite['mag']) > 13 and float(satellite['mag']) < 13.5:
                    if np.random.rand() < 0.20:
                        file_idx_to_remove.append(idx)
    for one_list_of_files in list_of_files:
        list_of_files_new.append([x for idx, x in enumerate(one_list_of_files) if idx not in file_idx_to_remove])
    return list_of_files_new

def build_list_dict(list_of_files):
    out = {}
    all_indices = list(range(len(list_of_files[0])))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=0)
    out['json_val'] = [list_of_files[0][i] for i in test_indices]
    out['png_val'] = [list_of_files[2][i] for i in test_indices]
    out['csv_val'] = [list_of_files[1][i] for i in test_indices]
    out['label_val'] = [list_of_files[3][i] for i in test_indices]
    out['json_train'] = [list_of_files[0][i] for i in train_indices]
    out['png_train'] = [list_of_files[2][i] for i in train_indices]
    out['csv_train'] = [list_of_files[1][i] for i in train_indices]
    out['label_train'] = [list_of_files[3][i] for i in train_indices]
    return out

def register_datasets_defilant(list_dict):
    DatasetCatalog.register('train_defilant', lambda : get_dataset_satellites(list_dict['json_train'], list_dict['png_train'], list_dict['label_train']))
    DatasetCatalog.register('val_defilant', lambda : get_dataset_satellites(list_dict['json_val'], list_dict['png_val'], list_dict['label_val']))
    DatasetCatalog.register('test_defilant', lambda : get_dataset_satellites(list_dict['json_test'], list_dict['png_test'], list_dict['label_test']))
    MetadataCatalog.get("train_defilant").set(thing_classes=["satellite"], thing_colors=[(255, 0, 0)])
    MetadataCatalog.get("val_defilant").set(thing_classes=["satellite"], thing_colors=[(255, 0, 0)])
    MetadataCatalog.get("test_defilant").set(thing_classes=["satellite"], thing_colors=[(255, 0, 0)])
    return MetadataCatalog.get("train_defilant")

def register_datasets_star(list_dict):
    DatasetCatalog.register('train_star', lambda : get_dataset_etoiles(list_dict['csv_train'], list_dict['png_train'], list_dict['label_train']))
    DatasetCatalog.register('val_star', lambda : get_dataset_etoiles(list_dict['csv_val'], list_dict['png_val'], list_dict['label_val']))
    DatasetCatalog.register('test_star', lambda : get_dataset_etoiles(list_dict['csv_test'], list_dict['png_test'], list_dict['label_test']))
    MetadataCatalog.get("train_star").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    MetadataCatalog.get("val_star").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    MetadataCatalog.get("test_star").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    return MetadataCatalog.get("train_star")

class MyVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.3
            
        labels = None
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 12, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

def display_classes(list_of_files_new, metadatas):
    
    random_index = np.random.randint(0, len(list_of_files_new[0]), 1)[0]
    data_satellite = get_gt_satellite(list_of_files_new[0][random_index], list_of_files_new[2][random_index], list_of_files_new[3][random_index])
    data_star = get_gt_star(list_of_files_new[1][random_index], list_of_files_new[2][random_index], list_of_files_new[3][random_index])
    
    print(data_satellite["file_name"])
    
    output = {'file_name': data_satellite["file_name"], 'height': 512, 'width': 512, 'image_id': 0}
    objs = []
    for obj_star in data_star['annotations']:
        obj_star['category_id'] = 1
        objs.append(obj_star)
    for obj_defilant in data_satellite['annotations']:
        obj_defilant['category_id'] = 0
        objs.append(obj_defilant)
    output['annotations'] = objs
    img = cv2.imread(data_satellite["file_name"], cv2.IMREAD_COLOR)
    visualizer = Visualizer(img, metadata=metadatas, scale=1.0, instance_mode=ColorMode.IMAGE)
    out = visualizer.draw_dataset_dict(output)
    cv2.imwrite('gt_image_example.png', out.get_image())


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self.trainer.model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

class MyTrainer(DefaultTrainer):
    #@classmethod
    #def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #    if output_folder is None:
    #        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #    return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True, augmentations=[]))))
        return hooks
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        rf = RandomFlip()
        if 'output_defilant' in cfg.OUTPUT_DIR:
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations=[]))
        else:
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations=[rf]))
        #rf = RandomFlip()
        #return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations=[]))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        #cfg_new = cfg.clone()
        #cfg_new.DATASETS.TRAIN = cfg.DATASETS.TEST
        #return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, True, augmentations=[]))
    
    #@classmethod
    #def build_optimizer(cls, cfg, model):
    #    """
    #    Returns:
    #        torch.optim.Optimizer:
    #    It now calls :func:`detectron2.solver.build_optimizer`.
    #    Overwrite it if you'd like a different optimizer.
    #    """
    #    return optim.Adam()
    #    #return build_optimizer(cfg, model)
    
    #@classmethod
    #def build_lr_scheduler(cls, cfg, optimizer):
    #    """
    #    It now calls :func:`detectron2.solver.build_lr_scheduler`.
    #    Overwrite it if you'd like a different scheduler.
    #    """
    #    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 100, )
    #    return build_lr_scheduler(cfg, optimizer)