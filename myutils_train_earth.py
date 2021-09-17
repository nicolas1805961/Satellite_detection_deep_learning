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
from detectron2.engine import DefaultTrainer, HookBase, TrainerBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
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

# Permet d'obtenir la vérité terrain pour les ponctuels d'une seule image
def get_gt_ponctuel(json_filepath, png_filepath):
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_ponctuels(json_filepath, image, 0)
    return image

# Permet d'obtenir la vérité terrain pour les defilants d'une seule image
def get_gt_defilant(json_filepath, png_filepath):
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_defilants(json_filepath, image, 0)
    return image

# Permet d'obtenir la vérité terrain pour les étoiles d'une seule image
def get_gt_star(csv_filepath, png_filepath):
    image = {'file_name': png_filepath, 'height': 512, 'width': 512, 'image_id': 0}
    image["annotations"] = []
    image = get_stars(csv_filepath, image, 0)
    return image

# crée la vérité terrain dans un format attendu par Detectron2 pour les ponctuels
def get_ponctuels(json_filepath, image, category_id, test=False):
    with open(json_filepath) as json_file:
        objs = []
        data = json.load(json_file)
        satellites = data['Objects']
        for satellite in satellites:
            if satellite['Type'] == 'defilant':
                continue
            obj = {"bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = category_id
            if test:
                obj["snr"] = satellite['snr']
            if satellite['mag'] < 16:
                margin = 4
            else:
                margin = 3
            #margin = -0.92 * satellite['mag'] + 20
            obj["bbox"] = [satellite['x0'] - margin, satellite['y0'] - margin, 2 * margin, 2 * margin]
            objs.append(obj)
        image["annotations"].extend(objs)
    return image

# Calcul de l'angle d'orientation pour les satellites défilants
def get_angle(satellite):
    delta_x = satellite['x0'] - satellite['x']
    delta_y = satellite['y'] - satellite['y0']
    return degrees(atan2(delta_y, delta_x))

# crée la vérité terrain dans un format attendu par Detectron2 pour un défilant (cf doc Detectron2)
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
                obj["snr"] = satellite['snr']
            margin = -2 * satellite['mag'] + 28
            angle = get_angle(satellite)
            second_point_coords = (satellite['x0'] + satellite['dx'], satellite['y0'] + satellite['dy'])
            width = sqrt((satellite['x0'] - second_point_coords[0])**2 + (satellite['y0'] - second_point_coords[1])**2) #+ margin
            height = -1.8518 * satellite['mag'] + 29.555
            obj["bbox"] = [satellite['x'], satellite['y'], width, height, angle]
            objs.append(obj)
        image["annotations"].extend(objs)
    return image

# crée la vérité terrain dans un format attendu par Detectron2 pour une étoile (cf doc Detectron2)
def get_stars(csv_filepath, image, category_id):
    df = pd.read_csv(csv_filepath, sep=';', usecols=['X0', 'Y0', 'X1', 'Y1', 'Gmag'])
    df['bbox_mode'] = BoxMode.XYWH_ABS
    df['category_id'] = category_id
    margin = (190 * np.exp(-0.5 * df['Gmag']) + 1)
    width = (df['X0'] - df['X1']) + 2 * margin
    height = (300 * np.exp(-0.4 * df['Gmag']) + 4)
    X = (df['X1'] - margin)
    Y = (df['Y1'] - (height / 2))
    bbox = pd.concat([X, Y, width, height], axis=1)
    df['bbox'] = bbox.values.tolist()
    df.drop(['X0','Y0', 'X1', 'Y1', 'Gmag'], axis=1, inplace=True)
    objs = df.to_dict(orient='records')
    image["annotations"].extend(objs)
    return image

# Crée le dataset pour l'ensemble des ponctuels dans le format attendu par Detectron2
def get_dataset_ponctuels(json_list, png_list):
    dataset = []
    for idx, (json_filepath, image_filepath) in enumerate(tqdm(zip(json_list, png_list), total=len(json_list), desc='Chargement du dataset ponctuel')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_ponctuels(json_filepath, image, 0)
        dataset.append(image)
    return dataset

# Crée le dataset pour l'ensemble des défilants dans le format attendu par Detectron2
def get_dataset_defilants(json_list, png_list):
    dataset = []
    for idx, (json_filepath, image_filepath) in enumerate(tqdm(zip(json_list, png_list), total=len(json_list), desc='Chargement du dataset defilant')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_defilants(json_filepath, image, 0)
        dataset.append(image)
    return dataset

# Crée le dataset pour l'ensemble des étoiles dans le format attendu par Detectron2
def get_dataset_stars(csv_list, png_list):
    dataset = []
    for idx, (csv_filepath, image_filepath) in enumerate(tqdm(zip(csv_list, png_list), total=len(csv_list), desc='Chargement du dataset etoile')):
        image = {'file_name': image_filepath, 'height': 512, 'width': 512, 'image_id': idx}
        image["annotations"] = []
        image = get_stars(csv_filepath, image, 0)
        dataset.append(image)
    return dataset

# Lecture des fichiers json, csv et png dans le dataset et stockage dans 3 listes distinctes
def build_lists(path):
    json_list = sorted(glob(os.path.join(path, 'json_data', '*.json')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    csv_list = sorted(glob(os.path.join(path, 'csv_data', '*.csv')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    png_list = sorted(glob(os.path.join(path, 'png_data', '*.png')), key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    print(len(json_list))
    print(len(csv_list))
    print(len(png_list))
    return [json_list, csv_list, png_list]

# Suppression des exemples trop nombreux dans une certaine plage de magnitudes (processus aléatoire)
def remove_small_mag(list_of_files):
    file_idx_to_remove = []
    list_of_files_new = []
    for idx, filepath in enumerate(list_of_files[0]):
        with open(filepath) as json_file:
            data = json.load(json_file)
            satellites = data['Objects']
            flag = False
            for satellite in satellites:
                if satellite['Type'] == 'defilant':
                    continue
                elif float(satellite['mag']) > 16.5:
                    flag = True
            if flag and np.random.rand() < 0.1:
                file_idx_to_remove.append(idx)
    for one_list_of_files in list_of_files:
        list_of_files_new.append([x for idx, x in enumerate(one_list_of_files) if idx not in file_idx_to_remove])
    return list_of_files_new

# Création d'un dictionnaire avec séparation entre données d'entrainement et de validation
def build_list_dict(list_of_files_new):
    out = {}
    all_indices = list(range(len(list_of_files_new[0])))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=0)
    out['json_val'] = [list_of_files_new[0][i] for i in test_indices]
    out['png_val'] = [list_of_files_new[2][i] for i in test_indices]
    out['csv_val'] = [list_of_files_new[1][i] for i in test_indices]
    out['json_train'] = [list_of_files_new[0][i] for i in train_indices]
    out['png_train'] = [list_of_files_new[2][i] for i in train_indices]
    out['csv_train'] = [list_of_files_new[1][i] for i in train_indices]
    return out

# Enregistrement du dataset des défilants auprès de Detectron2 (cf doc)
def register_datasets_defilant(list_dict):
    DatasetCatalog.register('train_defilant', lambda : get_dataset_defilants(list_dict['json_train'], list_dict['png_train']))
    DatasetCatalog.register('val_defilant', lambda : get_dataset_defilants(list_dict['json_val'], list_dict['png_val']))
    DatasetCatalog.register('test_defilant', lambda : get_dataset_defilants(list_dict['json_test'], list_dict['png_test']))
    MetadataCatalog.get("train_defilant").set(thing_classes=["defilant"], thing_colors=[(255, 0, 0)])
    MetadataCatalog.get("val_defilant").set(thing_classes=["defilant"], thing_colors=[(255, 0, 0)])
    MetadataCatalog.get("test_defilant").set(thing_classes=["defilant"], thing_colors=[(255, 0, 0)])
    return MetadataCatalog.get("train_defilant")

# Enregistrement du dataset des ponctuels auprès de Detectron2 (cf doc)
def register_datasets_ponctuel(list_dict):
    DatasetCatalog.register('train_ponctuel', lambda : get_dataset_ponctuels(list_dict['json_train'], list_dict['png_train']))
    DatasetCatalog.register('val_ponctuel', lambda : get_dataset_ponctuels(list_dict['json_val'], list_dict['png_val']))
    DatasetCatalog.register('test_ponctuel', lambda : get_dataset_ponctuels(list_dict['json_test'], list_dict['png_test']))
    MetadataCatalog.get("train_ponctuel").set(thing_classes=["ponctuel"], thing_colors=[(0, 255, 0)])
    MetadataCatalog.get("val_ponctuel").set(thing_classes=["ponctuel"], thing_colors=[(0, 255, 0)])
    MetadataCatalog.get("test_ponctuel").set(thing_classes=["ponctuel"], thing_colors=[(0, 255, 0)])
    return MetadataCatalog.get("train_ponctuel")

# Enregistrement du dataset des étoiles auprès de Detectron2 (cf doc)
def register_datasets_star(list_dict):
    DatasetCatalog.register('train_star', lambda : get_dataset_stars(list_dict['csv_train'], list_dict['png_train']))
    DatasetCatalog.register('val_star', lambda : get_dataset_stars(list_dict['csv_val'], list_dict['png_val']))
    DatasetCatalog.register('test_star', lambda : get_dataset_stars(list_dict['csv_test'], list_dict['png_test']))
    MetadataCatalog.get("train_star").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    MetadataCatalog.get("val_star").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    MetadataCatalog.get("test_star").set(thing_classes=["etoile"], thing_colors=[(0, 0, 255)])
    return MetadataCatalog.get("train_star")

# Classe de Detectron2 modifiée pour afficher les résultats différemment
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

# Fonction permettant d'afficher une image de vérité terrain avant l'entrainement pour controler
def display_classes(list_of_files_new, metadatas):
    
    random_index = np.random.randint(0, len(list_of_files_new[0]), 1)[0]
    data_defilant = get_gt_defilant(list_of_files_new[0][random_index], list_of_files_new[2][random_index])
    data_ponctuel = get_gt_ponctuel(list_of_files_new[0][random_index], list_of_files_new[2][random_index])
    data_star = get_gt_star(list_of_files_new[1][random_index], list_of_files_new[2][random_index])
    print(data_ponctuel["file_name"])
    
    output = {'file_name': data_defilant["file_name"], 'height': 512, 'width': 512, 'image_id': 0}
    objs = []
    for obj_star in data_star['annotations']:
        obj_data_star = {}
        obj_data_star['bbox'] = BoxMode.convert(obj_star['bbox'], obj_star['bbox_mode'], BoxMode.XYWHA_ABS)
        obj_data_star['bbox_mode'] = BoxMode.XYWHA_ABS
        obj_data_star['category_id'] = 2
        objs.append(obj_data_star)
    for obj_ponctuel in data_ponctuel['annotations']:
        obj_data_ponctuel = {}
        obj_data_ponctuel['bbox'] = BoxMode.convert(obj_ponctuel['bbox'], obj_ponctuel['bbox_mode'], BoxMode.XYWHA_ABS)
        obj_data_ponctuel['bbox_mode'] = BoxMode.XYWHA_ABS
        obj_data_ponctuel['category_id'] = 1
        objs.append(obj_data_ponctuel)
    for obj_defilant in data_defilant['annotations']:
        obj_data_defilant = {}
        obj_data_defilant['bbox'] = BoxMode.convert(obj_defilant['bbox'], obj_defilant['bbox_mode'], BoxMode.XYWHA_ABS)
        obj_data_defilant['bbox_mode'] = BoxMode.XYWHA_ABS
        obj_data_defilant['category_id'] = 0
        objs.append(obj_data_defilant)
    output['annotations'] = objs
    img = cv2.imread(data_ponctuel["file_name"], cv2.IMREAD_COLOR)
    visualizer = MyVisualizer(img, metadata=metadatas, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_dataset_dict(output)
    cv2.imwrite('gt_image_example.png', out.get_image())


# Remplacement de la cross entropy par la focal loss
def my_cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient

    nb_of_one = torch.tensor([(target == 1).sum()]) if (target == 1).sum() > 0 else torch.tensor([1])
    nb_of_zero = torch.tensor([(target == 0).sum()]) if (target == 0).sum() > 0 else torch.tensor([1])
    #weights = (1 / torch.cat((nb_of_zero, nb_of_one))).to('cuda:0')
    weights = (torch.cat((nb_of_one, nb_of_zero)) / (nb_of_one + nb_of_zero)).to('cuda:0')
    #return F.cross_entropy(input, target, weight=weights, **kwargs)
    #return F.cross_entropy(input, target, **kwargs)
    return focal_loss_multiclass(input, target, alphas=weights, gamma=2, reduction=reduction)
    #new_target = (~(target.type(torch.bool))).type(torch.float32)
    #return sigmoid_focal_loss(input[:, 0].reshape(target.shape), new_target, alpha=-1, gamma=1, reduction=reduction)

def focal_loss_multiclass(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alphas: torch.Tensor,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:

    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)
    return F.nll_loss(((1 - prob) ** gamma) * log_prob, targets, weight=alphas, reduction=reduction)

def focal_loss_multiclass2(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alphas: torch.Tensor,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:

    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    p_t = torch.exp(-ce_loss)

    loss = ce_loss * ((1 - p_t) ** gamma)

    if (~torch.all(alphas == alphas[0], dim=0)).item():
        alpha_t = torch.gather(alphas, 0, targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def focal_loss_binary(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

# Remplacement de la cross entropy par la focal loss
class MyRCNNOutput(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": my_cross_entropy(scores, gt_classes, reduction="mean"), # Seule ligne modifiée
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


# Block Squeeze and Excitation
class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


# Enregistrement d'un nouveau bout de code auprès de détectron pour remplacer la cross entropie par la focal loss
@ROI_HEADS_REGISTRY.register()
class MyStandardROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        my_box_predictor = MyRCNNOutput(cfg, box_head.output_shape) # Ma classe crée avec la focal loss
        super().__init__(cfg, input_shape, box_predictor=my_box_predictor)


@PROPOSAL_GENERATOR_REGISTRY.register()
class MyRPN(RPN):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.
        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0

        #target = gt_labels[valid_mask].to(torch.float32)

        #nb_of_one = torch.tensor([(target == 1.).sum()]) if (target == 1.).sum() > 0 else torch.tensor([1])
        #nb_of_zero = torch.tensor([(target == 0.).sum()]) if (target == 0.).sum() > 0 else torch.tensor([1])
        #weights = (1 / torch.cat((nb_of_zero, nb_of_one))).to('cuda:0')
        #weights = (torch.cat((nb_of_one, nb_of_zero)) / (nb_of_one + nb_of_zero)).to('cuda:0')

        # focal loss à la place de la cross entropie
        objectness_loss = focal_loss_binary(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32), alpha=0.25, gamma=2,
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

# Bottleneck block modifié en ajoutant le block Squeeze and Excitation
class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None
        
        self.se = SE_Block(out_channels)

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        
        out = self.se(out) # Ajout du block Squeeze and Excitation

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

# Classe FPN de Detectron2 modifiée pour pouvoir utiliser un autre backbone
class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]) + 1)
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s) + 1)): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** s
        
        print(self._out_feature_strides)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


# Basic stem du reseau resnet50. J'ai modifié la stride de la première couche pour eliminer une couche de sous échantillonage
class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=1, # stride=2
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

# Nouvelle fonction enregistrant un bout de code auprès de Detectron2 pour la construction du nouveau backbone
@BACKBONE_REGISTRY.register()
def my_build_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)

# Nouvelle fonction enregistrant un bout de code auprès de Detectron2 pour la construction du nouveau backbone
@BACKBONE_REGISTRY.register()
def my_build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = my_build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

# Fonction permettant de définir les paramètres de configuration pour chaque classe.
# Les paramètres sont ceux du fichier de config
def get_cfgs(config_dict, path_out, classes):
    x = Symbol('x', real=True, positive=True)
    cfgs = []
    for class_name in classes:
        config = config_dict[class_name.upper()]
        cfg = get_cfg()
        cfg.OUTPUT_DIR = os.path.join(path_out, 'output_' + class_name)
        # Mask-R-CNN
        if config_dict['TOOL'] == 'SEG_INSTANCE':
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # Faster-R-CNN
        else:
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.INPUT.MIN_SIZE_TRAIN = 512
        cfg.INPUT.MAX_SIZE_TRAIN = 512
        cfg.INPUT.MIN_SIZE_TEST = 512
        cfg.INPUT.MAX_SIZE_TEST = 512
        cfg.DATASETS.TRAIN = ("train_" + class_name,)
        cfg.DATASETS.TEST = ("val_" + class_name,)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.TEST.EVAL_PERIOD = 20
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.TEST.DETECTIONS_PER_IMAGE = 100000
        if class_name == 'ponctuel':
            cfg.SOLVER.MAX_ITER = config['SOLVER']['MAX_ITER']
            # Le learning rate augmente progressivement jusqu'à WARMUP_ITERS
            cfg.SOLVER.WARMUP_ITERS = cfg.SOLVER.MAX_ITER / 40
            cfg.SOLVER.BASE_LR = config['SOLVER']['BASE_LR']
            # Les deux lignes suivantes permettent de mettre en place une décroissance linéaire du learning rate
            # STEPS correspond aux itérations où le learning rate est multiplié par GAMMA.
            # Ici, STEPS correspond à toutes les itérations multiples de 20
            cfg.SOLVER.STEPS = np.arange(math.ceil(cfg.SOLVER.WARMUP_ITERS / 20) * 20, cfg.SOLVER.MAX_ITER, 20).tolist()
            # GAMMA est le flottant par lequel est multiplié le learning rate à chaque STEP.
            cfg.SOLVER.GAMMA = float(solve(cfg.SOLVER.BASE_LR * pow(x, len(cfg.SOLVER.STEPS)) - config['SOLVER']['FINAL_LR'], x)[0])
            # Utilisation de la focal loss
            cfg.MODEL.ROI_HEADS.NAME = 'MyStandardROIHeads'
            cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'MyRPN'
            # Utilisation du nouveau backbone
            if config['MODEL']['USE_BACKBONE'] and config_dict['MODE'] == 'EARTH' and config_dict['TOOL'] == 'BB_DETECTION':
                cfg.MODEL.BACKBONE.NAME = 'my_build_resnet_fpn_backbone'
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['MODEL']['ANCHOR_GENERATOR']['SIZES']
            cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
        elif class_name == 'defilant':
            cfg.SOLVER.MAX_ITER = config['SOLVER']['MAX_ITER']
            cfg.SOLVER.WARMUP_ITERS = cfg.SOLVER.MAX_ITER / 40
            cfg.SOLVER.BASE_LR = config['SOLVER']['BASE_LR']
            cfg.SOLVER.STEPS = np.arange(math.ceil(cfg.SOLVER.WARMUP_ITERS / 20) * 20, cfg.SOLVER.MAX_ITER, 20).tolist()
            cfg.SOLVER.GAMMA = float(solve(cfg.SOLVER.BASE_LR * pow(x, len(cfg.SOLVER.STEPS)) - config['SOLVER']['FINAL_LR'], x)[0])
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['MODEL']['ANCHOR_GENERATOR']['SIZES']
            cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
            if config_dict['TOOL'] == 'BB_DETECTION':
                cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
                cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
                cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
                cfg.MODEL.ANCHOR_GENERATOR.NAME = 'RotatedAnchorGenerator'
                cfg.MODEL.ANCHOR_GENERATOR.ANGLES = config['MODEL']['ANCHOR_GENERATOR']['ANGLES']
                cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = tuple(config['MODEL']['ROI_BOX_HEAD']['BBOX_REG_WEIGHTS'])
                cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1,1,1,1,1)
        elif class_name == 'star':
            if config_dict['TOOL'] == 'SEG_INSTANCE':
                cfg.MODEL.ROI_HEADS.NAME = 'MyStandardROIHeads'
                cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'MyRPN'
            cfg.SOLVER.MAX_ITER = config['SOLVER']['MAX_ITER']
            cfg.SOLVER.WARMUP_ITERS = cfg.SOLVER.MAX_ITER / 40
            cfg.SOLVER.BASE_LR = config['SOLVER']['BASE_LR']
            cfg.SOLVER.STEPS = np.arange(math.ceil(cfg.SOLVER.WARMUP_ITERS / 20) * 20, cfg.SOLVER.MAX_ITER, 20).tolist()
            cfg.SOLVER.GAMMA = float(solve(cfg.SOLVER.BASE_LR * pow(x, len(cfg.SOLVER.STEPS)) - config['SOLVER']['FINAL_LR'], x)[0])
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['MODEL']['ANCHOR_GENERATOR']['SIZES']
            cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN']
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = config['MODEL']['RPN']['POST_NMS_TOPK_TEST']
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = config['MODEL']['RPN']['PRE_NMS_TOPK_TEST']
            cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN']
            cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
            print(cfg)
        cfgs.append(cfg)
    return cfgs

# Suppression du precedent dossier se trouvant à PATH_OUT.
# Sauvegarde des configurations pour chaque classe dans un nouveau dossier crée à PATH_OUT.
def set_folder(cfgs, path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)    
    Path(path).mkdir(exist_ok=True)
    with open(os.path.join(path, 'cfgs.txt'), "wb") as fp:
        pickle.dump(cfgs, fp)

# Classe nécessaire pour avoir des détections avec des boites orientées
def my_transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
  if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
    annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
  else:
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

  return annotation

# Classe nécessaire pour avoir des détections avec des boites orientées
class My_Dataset_Mapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        #image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Implement additional transformations if you have other types of data
        annos = [
            my_transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

# Classe permettant d'obtenir une loss sur les données de validation au moment de l'entrainement
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, get_val_loss):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.final_loss = None
        self.get_val_loss = get_val_loss

    def _do_loss_eval(self, write_metric=True):
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
        if write_metric:
            self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return mean_loss
            
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
        if self.get_val_loss:
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                self.final_loss = self._do_loss_eval()
        else:
            if is_final:
                self.final_loss = self._do_loss_eval(False)
        self.trainer.storage.put_scalars(timetest=12)

# Classe permettant de faire l'entrainement.
# L'augmentation de données a été supprimée pour les satellites défilants.
class MyTrainer(DefaultTrainer):
    def __init__(self, cfg, get_val_loss):
        self.get_val_loss = get_val_loss
        super().__init__(cfg)
    #@classmethod
    #def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #    if output_folder is None:
    #        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #    return COCOEvaluator(dataset_name, cfg, True, output_folder)
    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        TrainerBase.train(self, self.start_iter, self.max_iter)
        if comm.is_main_process():
            if len(self.cfg.TEST.EXPECTED_RESULTS):
                assert hasattr(
                    self, "_last_eval_results"
                ), "No evaluation results obtained during training!"
                verify_results(self.cfg, self._last_eval_results)
                return self._last_eval_results
            else:
                return self.custom_hook.final_loss
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        if 'output_defilant' in self.cfg.OUTPUT_DIR:
            self.custom_hook = LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], My_Dataset_Mapper(self.cfg, True, augmentations=[])), self.get_val_loss)
        else:
            self.custom_hook = LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True, augmentations=[])), self.get_val_loss)
        hooks.insert(-1, self.custom_hook)
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
            return build_detection_train_loader(cfg, mapper=My_Dataset_Mapper(cfg, True, augmentations=[]))
        else:
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations=[rf]))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        #if cfg.OUTPUT_DIR == '/data_deep/SST_CNES_LOT2/output_earth/output_defilant':
        #    return build_detection_train_loader(cfg, mapper=My_Dataset_Mapper(cfg, True, augmentations=[]))
        #else:
        #    return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations=[]))
    
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

# Cette fonction est la fonction que Optuna va chercher à optimiser
# La fonction doit prendre le paramètre trial en argument
def objective(trial, cfg):
    # 2 paramètres vont être optimisés ici avec les fonctions "suggest_*"
    # On aurait pu en définir plus ou moins
    lr_start = trial.suggest_float("lr_start", 1e-4, 1e-2)
    lr_end = trial.suggest_float("lr_end", 1e-5, lr_start / 5)
    print(f'lr_start = {lr_start}')
    print(f'lr_end = {lr_end}')
    cfg.SOLVER.BASE_LR = lr_start
    x = Symbol('x', real=True, positive=True)
    cfg.SOLVER.GAMMA = float(solve(cfg.SOLVER.BASE_LR * pow(x, len(cfg.SOLVER.STEPS)) - lr_end, x)[0])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg, False)
    trainer.resume_or_load(resume=False)
    # L'entrainement est lancé normalement au sein de la fonction à optimiser
    final_loss = trainer.train()
    # La valeur retournée par cette fonction est la quantité qui doit être optimisée.
    # Ici il s'agit de la loss sur les données de validation
    return final_loss

# Fonction permettant de lancer l'optimisation
# On peut remplacer la partie responsable de l'entrainement dans le script
# train_sst.py par cette fonction
def optimize():
    study = optuna.create_study()
    # On indique la fonction que l'on cherche à optimiser grâce à une lambda fonction de manière
    # à pouvoir passer plusieurs paramètres
    # n_trials indique le nombre d'entrainements lancé pour optimiser les paramètres
    study.optimize(lambda trial: objective(trial, cfgs[1]), n_trials=100)
    # On récupère les meilleures paramètres à la fin de ces n_trials essais
    best_params = study.best_params
    found_lr_start = best_params["lr_start"]
    found_lr_end = best_params["lr_end"]