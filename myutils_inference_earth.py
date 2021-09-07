# import some common libraries
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
from glob import glob
import torch
from shapely.geometry import Polygon
import math
from tqdm import tqdm

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
from detectron2.structures import BoxMode, Instances, RotatedBoxes

from myutils_train_earth import my_build_resnet_fpn_backbone

# Cette fonction permet de charger les poids des modèles pour chaque classe et de définir des paramètres de test
def get_predictors(cfgs, weights, config_dict):
    predictors = []
    if config_dict['MODE'] == 'SIDERAL':
        classes = ['DEFILANT', 'STAR']
    else:
        classes = ['DEFILANT', 'PONCTUEL', 'STAR']
    if weights is None:
        for cfg_class, class_type in zip(cfgs, classes):
            # Inference should use the config with parameters that are used in training
            # cfg now already contains everything we've set previously. We changed it a little bit for inference:
            cfg_class.MODEL.WEIGHTS = os.path.join(cfg_class.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
            cfg_class.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config_dict[class_type]['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST']   # set a custom testing threshold
            cfg_class.MODEL.ROI_HEADS.NMS_THRESH_TEST = config_dict[class_type]['MODEL']['ROI_HEADS']['NMS_THRESH_TEST']   # set a custom testing threshold
            predictor = DefaultPredictor(cfg_class)
            predictors.append(predictor)
    else:
        for cfg_class, class_weights, class_type in zip(cfgs, weights, classes):
            # Inference should use the config with parameters that are used in training
            # cfg now already contains everything we've set previously. We changed it a little bit for inference:
            cfg_class.MODEL.WEIGHTS = class_weights  # path to the model
            cfg_class.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config_dict[class_type]['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST']   # set a custom testing threshold
            cfg_class.MODEL.ROI_HEADS.NMS_THRESH_TEST = config_dict[class_type]['MODEL']['ROI_HEADS']['NMS_THRESH_TEST']   # set a custom testing threshold
            predictor = DefaultPredictor(cfg_class)
            predictors.append(predictor)
    return predictors

# Cette fonction permet de découper les images en 25 patchs se recouvrant sur 128 pixels
def split_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    x = torch.tensor(img)
    res = x.unfold(1, 512, 384).unfold(0, 512, 384)
    image  ={'file_name': path, 'patches': torch.transpose(res, 2, -1)}
    return image

# Cette fonction permet de dessiner les lignes séparant les sous-images.
def draw_lines(image):
    for i in range(384, 2048 - 384, 384):
        image = cv2.line(image, (i, 0), (i, 2048), color=(255, 255, 255), thickness=1)
        image = cv2.line(image, (i + 128, 0), (i + 128, 2048), color=(255, 255, 255), thickness=1)

        image = cv2.line(image, (0, i), (2048, i), color=(255, 255, 255), thickness=1)
        image = cv2.line(image, (0, i + 128), (2048, i + 128), color=(255, 255, 255), thickness=1)
    return image

def convert_boxes(boxes):
    res = torch.clone(boxes)
    res[:, 2] = res[:, 2] - res[:, 0]
    res[:, 3] = res[:, 3] - res[:, 1]
    return res

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

# Création d'un objet polygon à partir des 4 coins de la boite englobante
def get_polygons(bev_corners):
    list_of_polygons = []
    for i in range(len(bev_corners)):
        list_of_polygons.append(Polygon([(bev_corners[i, j, 0], bev_corners[i, j, 1]) for j in range(4)]).buffer(0))
    return list_of_polygons

def filter_boxes(image_data):
    indices_to_remove_list = []
    boxes = image_data['instances'].pred_boxes.tensor
    classes = image_data['instances'].pred_classes
    print(len(boxes))
    areas = image_data['instances'].pred_boxes.area().reshape(-1, 1)

    boxes_rows = convert_boxes(boxes)
    boxes_cols = convert_boxes(boxes)
    ious = get_iou_not_same_shape(boxes_cols, boxes_rows)
    ious[torch.eye(len(ious), dtype=torch.bool)] = 0
    indices_pair = torch.nonzero(ious > 0.25)
    #mask_ponctuels = (classes == 1).reshape(-1, 1)
    mask_ious = torch.any(ious > 0.05, dim=1).reshape(-1, 1)
    #mask_ponctuels_remove = torch.all(torch.cat([mask_ponctuels, mask_ious], dim=1), dim=1).reshape(-1, 1)

    areas_row = areas[indices_pair[:, 0]]
    areas_col = areas[indices_pair[:, 1]]
    whole_areas = torch.cat([areas_row, areas_col], dim=1)
    indices_to_remove = torch.min(whole_areas, dim=1)[1]
    indices_to_remove = torch.gather(indices_pair, 1, indices_to_remove.reshape(-1, 1)).reshape(-1)

    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]

    for i in range(384, 2048 - 384, 384):
        col1 = ((x0 > i) & (x0 < i + 128)).reshape(-1, 1)
        col2 = ((x1 > i) & (x1 < i + 128)).reshape(-1, 1)
        col3 = ((y0 > i) & (y0 < i + 128)).reshape(-1, 1)
        col4 = ((y1 > i) & (y1 < i + 128)).reshape(-1, 1)
        mask = torch.cat([col1, col2, col3, col4], dim=1)
        mask = torch.any(mask, dim=1).reshape(-1, 1)
        #mask_ponctuels = torch.all(torch.cat([mask_ponctuels_remove, mask], dim=1), dim=1)
        #indices_filtered_ponctuels = torch.nonzero(mask_ponctuels)[:, 0]
        mask = mask[indices_pair[:, 0]].reshape(-1)
        indices_filtered = indices_to_remove[mask]
        #indices_filtered = torch.cat([indices_filtered, indices_filtered_ponctuels])
        indices_to_remove_list.append(indices_filtered)
    indices_filter = torch.cat(indices_to_remove_list)
    indices_filter = torch.unique(indices_filter)
    print(len(indices_filter))

    index = torch.ones(boxes.shape[0], dtype=torch.bool)
    index[indices_filter] = False

    image_data['instances'].pred_boxes.tensor = image_data['instances'].pred_boxes.tensor[index]
    image_data['instances'].scores = image_data['instances'].scores[index]
    image_data['instances'].pred_classes = image_data['instances'].pred_classes[index]
    
    return image_data

# Cette fonction calcul le pourcentage de superposition plutôt que l'IOU pour les post traitements.
def get_my_metric(polys_a, polys_b):
    table_metric = torch.zeros((len(polys_a), len(polys_b)), dtype=torch.float)
    for i, poly_row in enumerate(polys_a):
        for j, poly_col in enumerate(polys_b):
            intersection = poly_row.intersection(poly_col)
            intersection_area = intersection.area
            table_metric[i, j] = intersection_area / poly_row.area
    return table_metric

# Calcul de l'IOU pour des boites orientées.
def get_iou_rotated(polys_a, polys_b):
    table_iou = torch.zeros((len(polys_a), len(polys_b)), dtype=torch.float)
    for i, poly_row in enumerate(polys_a):
        for j, poly_col in enumerate(polys_b):
            intersection = poly_row.intersection(poly_col)
            intersection_area = intersection.area
            union = poly_col.area + poly_row.area - intersection_area
            table_iou[i, j] = intersection_area / (union + 1e-16)
    return table_iou

def filter_boxes_rotated(image_data, thresh):
    indices_to_remove_list = []
    boxes = image_data['instances'].pred_boxes.tensor
    classes = image_data['instances'].pred_classes
    areas = image_data['instances'].pred_boxes.area().reshape(-1, 1)
    corners = get_corners(boxes)
    
    x0 = corners[:, 0, 0]
    y0 = corners[:, 0, 1]
    x1 = corners[:, 2, 0]
    y1 = corners[:, 2, 1]

    for i in range(384, 2048 - 384, 384):
        col1 = ((x0 > i) & (x0 < i + 128)).reshape(-1, 1)
        col2 = ((x1 > i) & (x1 < i + 128)).reshape(-1, 1)
        col3 = ((y0 > i) & (y0 < i + 128)).reshape(-1, 1)
        col4 = ((y1 > i) & (y1 < i + 128)).reshape(-1, 1)
        mask = torch.cat([col1, col2, col3, col4], dim=1)
        mask = torch.any(mask, dim=1).reshape(-1,)
        true_indices = torch.nonzero(mask)
        
        filtered_boxes = boxes[mask]
        filtered_areas = areas[mask]
        corners_filtered = corners[mask]
        
        polys = get_polygons(corners_filtered)
        ious = get_iou_rotated(polys, polys)
        ious[torch.eye(len(ious), dtype=torch.bool)] = 0
        indices_pair = torch.nonzero(ious > thresh)
        if indices_pair.numel() == 0:
            continue
        areas_row = filtered_areas[indices_pair[:, 0]]
        areas_col = filtered_areas[indices_pair[:, 1]]
        whole_areas = torch.cat([areas_row, areas_col], dim=1)
        indices_to_remove_filtered = torch.min(whole_areas, dim=1)[1]
        indices_to_remove_filtered = torch.gather(indices_pair, 1, indices_to_remove_filtered.reshape(-1, 1)).reshape(-1)
        
        indices_filtered = true_indices[indices_to_remove_filtered]
        
        indices_to_remove_list.append(indices_filtered)
    indices_filter = torch.cat(indices_to_remove_list)
    indices_filter = torch.unique(indices_filter)

    index = torch.ones(boxes.shape[0], dtype=torch.bool)
    index[indices_filter] = False

    image_data['instances'].pred_boxes.tensor = image_data['instances'].pred_boxes.tensor[index]
    image_data['instances'].scores = image_data['instances'].scores[index]
    image_data['instances'].pred_classes = image_data['instances'].pred_classes[index]
    
    return image_data

# Cette fonction permet de supprimer les detections redondantes sur les zones de recouvrement
def filter_boxes_rotated2(image_data, thresh):
    classes_id = image_data['instances'].pred_classes
    boxes = image_data['instances'].pred_boxes.tensor
    indices_to_remove_list = []

    for i in range(3):
        current_mask = (classes_id == i)
        object_indices = torch.nonzero(current_mask)
        boxes_current = boxes[current_mask]
        areas = image_data['instances'].pred_boxes.area()[current_mask].reshape(-1, 1)
        corners = get_corners(boxes_current)
        
        x0 = corners[:, 0, 0]
        y0 = corners[:, 0, 1]
        x1 = corners[:, 2, 0]
        y1 = corners[:, 2, 1]

        for i in range(384, 2048 - 384, 384):
            col1 = ((x0 > i) & (x0 < i + 128)).reshape(-1, 1)
            col2 = ((x1 > i) & (x1 < i + 128)).reshape(-1, 1)
            col3 = ((y0 > i) & (y0 < i + 128)).reshape(-1, 1)
            col4 = ((y1 > i) & (y1 < i + 128)).reshape(-1, 1)
            mask = torch.cat([col1, col2, col3, col4], dim=1)
            mask = torch.any(mask, dim=1).reshape(-1,)
            true_indices = torch.nonzero(mask)
            
            filtered_areas = areas[mask]
            corners_filtered = corners[mask]
            
            polys = get_polygons(corners_filtered)
            metrics = get_my_metric(polys, polys)
            metrics[torch.eye(len(metrics), dtype=torch.bool)] = 0
            indices_pair = torch.nonzero(metrics > thresh)
            if indices_pair.numel() == 0:
                continue
            areas_row = filtered_areas[indices_pair[:, 0]]
            areas_col = filtered_areas[indices_pair[:, 1]]
            whole_areas = torch.cat([areas_row, areas_col], dim=1)
            indices_to_remove_filtered = torch.min(whole_areas, dim=1)[1]
            indices_to_remove_filtered = torch.gather(indices_pair, 1, indices_to_remove_filtered.reshape(-1, 1)).reshape(-1)
            
            indices_filtered = object_indices[true_indices[indices_to_remove_filtered]]
            
            indices_to_remove_list.append(indices_filtered)
    indices_filter = torch.cat(indices_to_remove_list)
    indices_filter = torch.unique(indices_filter)

    index = torch.ones(boxes.shape[0], dtype=torch.bool)
    index[indices_filter] = False

    image_data['instances'].pred_boxes.tensor = image_data['instances'].pred_boxes.tensor[index]
    image_data['instances'].scores = image_data['instances'].scores[index]
    image_data['instances'].pred_classes = image_data['instances'].pred_classes[index]
    
    return image_data

def get_means(boxes, img):
    patches_mean_list = []
    corners = get_corners(boxes)
    for i in range(len(boxes)):
        current_box = boxes[i].reshape(-1,)
        angle = current_box[4].item()
        center = (current_box[0].item(), current_box[1].item())
        M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1)
        img_rot = cv2.warpAffine(src=img, M=M, dsize=(img.shape[1], img.shape[0]))

        img_crop = cv2.getRectSubPix(img_rot, (int(current_box[2].item()), int(current_box[3].item())), center)
        patches_mean_list.append(img_crop.mean())
        plt.imshow(img_crop, cmap='gray')
        plt.show()
    means = torch.tensor(patches_mean_list)
    return means.reshape(-1, 1)

def get_means_shapely(boxes, img):
    patches_mean_list = []
    for i in range(len(boxes)):
        current_box = boxes[i].type(torch.int).reshape(-1,)
        img_crop = img[current_box[1]:current_box[3], current_box[0]:current_box[2]]
        patches_mean_list.append(img_crop.mean())
        plt.imshow(img_crop, cmap='gray')
        plt.show()
    means = torch.tensor(patches_mean_list)
    return means.reshape(-1, 1)

# Cette fonction permet de supprimer les fausses détections de ponctuels superposées aux autres types d'objets.
def filter_ponctuels(image_data, metric_thresh, score_thresh):
    # image_data: données après inférence
    # metric_thresh: seuil du pourcentage de recouvrement au dessus duquel les ponctuels sont considérés comme superposés
    # score_thresh: seuil du score de confiance en dessous duquel les ponctuels sont considérés comme pouvant être supprimés.
    boxes = image_data['instances'].pred_boxes.tensor
    classes = image_data['instances'].pred_classes
    scores = image_data['instances'].scores
    mask = (classes == 1).type(torch.bool)
    true_indices = torch.nonzero(mask)
    boxes_ponctuels = boxes[mask]
    scores_ponctuels = scores[mask]
    boxes_non_ponctuel = boxes[~mask]
    corners_ponctuels = get_corners(boxes_ponctuels)
    corners_non_ponctuel = get_corners(boxes_non_ponctuel)
    polys_ponctuels = get_polygons(corners_ponctuels)
    polys_non_ponctuels = get_polygons(corners_non_ponctuel)
    metrics = get_my_metric(polys_ponctuels, polys_non_ponctuels)
    indices_pair = torch.nonzero(metrics > metric_thresh)
    values, count = torch.unique(indices_pair[:, 0], return_counts=True)
    values = values[count > 1]
    ponctuel_overlap = boxes_ponctuels[indices_pair[:, 0]]
    ponctuel_overlap_scores = scores_ponctuels[indices_pair[:, 0]]
    other_overlap = boxes_non_ponctuel[indices_pair[:, 1]]
    score_mask = (ponctuel_overlap_scores < score_thresh)
    indices_to_remove = true_indices[indices_pair[:, 0][score_mask]]
    indices_to_remove = torch.unique(torch.cat([indices_to_remove, true_indices[values]], dim=0))
    
    index = torch.ones(boxes.shape[0], dtype=torch.bool)
    index[indices_to_remove] = False

    image_data['instances'].pred_boxes.tensor = image_data['instances'].pred_boxes.tensor[index]
    image_data['instances'].scores = image_data['instances'].scores[index]
    image_data['instances'].pred_classes = image_data['instances'].pred_classes[index]
    
    return image_data

def build_lists(path):
    png_list = glob(os.path.join(path, '*.png'))
    print(len(png_list))
    return png_list

# Classe de Detectron2 modifiée pour afficher les résultats différemment
class MyVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions, display_labels):
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

        # Modification faite ici pour ne pas afficher les labels
        if not display_labels:
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
            24 if area < 1000 * self.output.scale else 12 # 12/6 peut être modifié
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

# Cette fonction permet d'effectuer l'inference sur les images présentes dans un dossier
def get_infered_image(png_list, predictors, metadatas, path_out, display_lines, display_labels):
    for path in tqdm(png_list):
        image_data = split_image(path)
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
                        # Convertion des boites du format [x, y, x, y] au format [x, y, w, h, a] pour avoir des boites orientées
                        boxes = BoxMode.convert(out['instances'].pred_boxes.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).to('cpu')
                        pred_boxes_list.append(RotatedBoxes(BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)))
                        pred_classes_list.append(torch.full(out['instances'].pred_classes.shape, idx, device='cuda:0'))
                    else:
                        pred_boxes_list.append(out['instances'].pred_boxes.to('cpu'))
                        pred_classes_list.append(out['instances'].pred_classes)
                    scores_list.append(out['instances'].scores)

                # Fusion des résultats des 3 détecteurs
                output['instances'].pred_boxes = RotatedBoxes.cat(pred_boxes_list)
                output['instances'].scores = torch.cat(scores_list)
                output['instances'].pred_classes = torch.cat(pred_classes_list)
                
                # Mise à l'echelle des coordonnées des boites englobantes
                output['instances'].pred_boxes.tensor[:, 0] += (j * 384)
                output['instances'].pred_boxes.tensor[:, 1] += (i * 384)
                pred_boxes.append(output['instances'].pred_boxes)
                scores.append(output['instances'].scores)
                pred_classes.append(output['instances'].pred_classes)
        # Fusion des résultats des inférences sur chaque patch
        big_output['instances'].pred_boxes = RotatedBoxes.cat(pred_boxes)
        big_output['instances'].scores = torch.cat(scores)
        big_output['instances'].pred_classes = torch.cat(pred_classes)

        # Post traitements
        big_output = filter_boxes_rotated2(big_output, 0.5)
        big_output = filter_ponctuels(big_output, 0.5, 0.85)

        im = cv2.imread(path)
        #big_output = filter_visualization(big_output, 1)

        v = MyVisualizer(im[:, :, ::-1], metadata=metadatas, scale=1, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(big_output["instances"].to("cpu"), display_labels)
        out_im = out.get_image()[:, :, ::-1]
        if display_lines:
            out_im = draw_lines(out_im.astype(np.float32)).astype(np.uint8)
        converted = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
            #converted = np.flipud(converted)
        cv2.imwrite(os.path.join(path_out, path.split('/')[-1]), converted)