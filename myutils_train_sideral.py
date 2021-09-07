# import some common libraries
import pandas as pd
import numpy as np
import os, json, cv2
from glob import glob
from math import sqrt, atan2, degrees
import json
from sklearn.model_selection import train_test_split
import shutil
import time
import torch
from torch.nn import functional as F
import torch.nn as nn
import pickle
import copy
from typing import Dict, List
from pathlib import Path
import math
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
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, DatasetMapper, build_detection_train_loader
from detectron2.structures import BoxMode, Boxes
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats
from detectron2.layers import cat, ShapeSpec, CNNBlockBase, Conv2d, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, PROPOSAL_GENERATOR_REGISTRY, StandardROIHeads, BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.backbone import BottleneckBlock, ResNet, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool,  _assert_strides_are_log2_contiguous
from detectron2.modeling.proposal_generator.rrpn import RRPN
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import _dense_box_regression_loss
import detectron2.utils.comm as comm
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import fvcore.nn.weight_init as weight_init

def get_gt_satellite(json_filepath, png_filepath):
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_defilants(json_filepath, image, 0)
    return image

def get_gt_star(csv_filepath, png_filepath):
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_stars(csv_filepath, image, 0)
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

def get_defilants(json_filepath, image, category_id, test=False):
    with open(json_filepath) as json_file:
        objs = []
        data = json.load(json_file)
        satellites = data['Objects']
        for satellite in satellites:
            if satellite['Type'] != 'defilant':
                continue
            obj = {"bbox_mode": BoxMode.XYWHA_ABS}
            obj["category_id"] = category_id
            if test:
                obj["snr"] = satellite["snr"]
            margin = -2 * satellite['mag'] + 28
            angle = get_angle(satellite)
            second_point_coords = (satellite['x0'] + satellite['dx'], satellite['y0'] + satellite['dy'])
            width = sqrt((satellite['x0'] - second_point_coords[0])**2 + (satellite['y0'] - second_point_coords[1])**2) #+ margin
            height = -1.8518 * satellite['mag'] + 29.555
            obj["bbox"] = [satellite['x'], satellite['y'], width, height, angle]
            objs.append(obj)
        image["annotations"].extend(objs)
    return image

def get_stars(csv_filepath, image, category_id):
    df = pd.read_csv(csv_filepath, sep=';', usecols=['X0', 'Y0', 'X1', 'Y1', 'Gmag'])
    df['bbox_mode'] = BoxMode.XYWH_ABS
    df['category_id'] = category_id
    margin = ((600 * np.exp(-0.4 * df['Gmag']) + 5) / 2).round().astype(int)
    width = 2 * margin
    height = 2 * margin
    X = (df['X1'] - margin)
    Y = (df['Y1'] - margin)
    bbox = pd.concat([X, Y, width, height], axis=1)
    df['bbox'] = bbox.values.tolist()
    df.drop(['X0','Y0', 'X1', 'Y1', 'Gmag'], axis=1, inplace=True)
    objs = df.to_dict(orient='records')
    image["annotations"].extend(objs)
    return image

def get_dataset_all_classes(json_list, png_list, csv_list):
    dataset = []
    for idx, (json_filepath, image_filepath, csv_filepath) in enumerate(tqdm(zip(json_list, png_list, csv_list), total=len(json_list), desc='Chargement des datasets')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(json_filepath, image, 0, True)
        #image = get_stars(csv_filepath, image, 1)
        dataset.append(image)
    return dataset

def get_dataset_satellites(json_list, png_list):
    dataset = []
    for idx, (json_filepath, image_filepath) in enumerate(tqdm(zip(json_list, png_list), total=len(json_list), desc='Chargement du dataset defilant')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(json_filepath, image, 0)
        dataset.append(image)
    return dataset

def get_dataset_etoiles(csv_list, png_list):
    dataset = []
    for idx, (csv_filepath, image_filepath) in enumerate(tqdm(zip(csv_list, png_list), total=len(csv_list), desc='Chargement du dataset etoile')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_stars(csv_filepath, image, 0)
        dataset.append(image)
    return dataset

def remove_mag(list_of_files):
    file_idx_to_remove = []
    list_of_files_new = []
    for idx, filepath in enumerate(list_of_files[0]):
        with open(filepath) as json_file:
            data = json.load(json_file)
            satellites = data['Objects']
            for satellite in satellites:
                if float(satellite['mag']) > 13 and float(satellite['mag']) < 13.5:
                    if np.random.rand() < 0.22:
                        file_idx_to_remove.append(idx)
    for one_list_of_files in list_of_files:
        list_of_files_new.append([x for idx, x in enumerate(one_list_of_files) if idx not in file_idx_to_remove])
    return list_of_files_new

def register_datasets_satellite(list_dict):
    DatasetCatalog.register('train_satellite', lambda : get_dataset_satellites(list_dict['json_train'], list_dict['png_train']))
    DatasetCatalog.register('val_satellite', lambda : get_dataset_satellites(list_dict['json_val'], list_dict['png_val']))
    DatasetCatalog.register('test_satellite', lambda : get_dataset_satellites(list_dict['json_test'], list_dict['png_test']))
    MetadataCatalog.get("train_satellite").set(thing_classes=["satellite"], thing_colors=[(255, 0, 0)])
    MetadataCatalog.get("val_satellite").set(thing_classes=["satellite"], thing_colors=[(255, 0, 0)])
    MetadataCatalog.get("test_satellite").set(thing_classes=["satellite"], thing_colors=[(255, 0, 0)])
    return MetadataCatalog.get("train_satellite")

def register_datasets_etoile(list_dict):
    DatasetCatalog.register('train_etoile', lambda : get_dataset_etoiles(list_dict['csv_train'], list_dict['png_train']))
    DatasetCatalog.register('val_etoile', lambda : get_dataset_etoiles(list_dict['csv_val'], list_dict['png_val']))
    DatasetCatalog.register('test_etoile', lambda : get_dataset_etoiles(list_dict['csv_test'], list_dict['png_test']))
    MetadataCatalog.get("train_etoile").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    MetadataCatalog.get("val_etoile").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    MetadataCatalog.get("test_etoile").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    return MetadataCatalog.get("train_etoile")

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):
    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

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
                [x / 255 for x in self.metadata.thing_colors[c]] for c in classes
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
    
    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.
        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.
        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    [x / 255 for x in self.metadata.thing_colors[c]]
                    for c in category_ids
                ]
            names = self.metadata.get("thing_classes", None)
            labels = _create_text_labels(
                category_ids,
                scores=None,
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
            )
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)

        pan_seg = dic.get("pan_seg", None)
        if pan_seg is None and "pan_seg_file_name" in dic:
            with PathManager.open(dic["pan_seg_file_name"], "rb") as f:
                pan_seg = Image.open(f)
                pan_seg = np.asarray(pan_seg)
                from panopticapi.utils import rgb2id

                pan_seg = rgb2id(pan_seg)
        if pan_seg is not None:
            segments_info = dic["segments_info"]
            pan_seg = torch.tensor(pan_seg)
            self.draw_panoptic_seg(pan_seg, segments_info, area_threshold=0, alpha=0.5)
        return self.output
    
    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None):
        """
        Draw a rotated box with label on its top-left corner.
        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.
        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            24 if area < 1000 * self.output.scale else 12
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[0]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

def display_classes(list_of_files_new, metadatas):
    
    random_index = np.random.randint(0, len(list_of_files_new[0]), 1)[0]
    data_satellite = get_gt_satellite(list_of_files_new[0][random_index], list_of_files_new[2][random_index])
    data_star = get_gt_star(list_of_files_new[1][random_index], list_of_files_new[2][random_index])
    print(data_satellite["file_name"])
    
    output = {'file_name': data_satellite["file_name"], 'height': 512, 'width': 512, 'image_id': 0}
    objs = []
    for obj_star in data_star['annotations']:
        obj_data_star = {}
        obj_data_star['bbox'] = BoxMode.convert(obj_star['bbox'], obj_star['bbox_mode'], BoxMode.XYWHA_ABS)
        obj_data_star['bbox_mode'] = BoxMode.XYWHA_ABS
        obj_data_star['category_id'] = 1
        objs.append(obj_data_star)
    for obj_ponctuel in data_satellite['annotations']:
        obj_data_ponctuel = {}
        obj_data_ponctuel['bbox'] = BoxMode.convert(obj_ponctuel['bbox'], obj_ponctuel['bbox_mode'], BoxMode.XYWHA_ABS)
        obj_data_ponctuel['bbox_mode'] = BoxMode.XYWHA_ABS
        obj_data_ponctuel['category_id'] = 0
        objs.append(obj_data_ponctuel)
    output['annotations'] = objs
    img = cv2.imread(data_satellite["file_name"], cv2.IMREAD_COLOR)
    visualizer = MyVisualizer(img, metadata=metadatas, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_dataset_dict(output)
    cv2.imwrite('gt_image_example.png', out.get_image())