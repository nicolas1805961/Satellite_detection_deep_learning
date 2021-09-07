# import some common libraries
import numpy as np
import cv2
import torch
from tqdm import tqdm
import math
from shapely.geometry import Polygon
import matplotlib as mpl
import os

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, _create_text_labels

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

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

# Découpage des images en 25 patchs se recouvrant sur 128 pixels
def split_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    x = torch.tensor(img)
    res = x.unfold(1, 512, 384).unfold(0, 512, 384)
    image  ={'file_name': path, 'patches': torch.transpose(res, 2, -1)}
    return image

# Suppression d'un pourcentage d'exemples d'étoiles en segmentation d'instance du fait d'un problème mémoire
def drop_sample(out, nb_to_keep):
    new_out = {'instances': Instances((512, 512))}
    # Maintient des 'nb_to_keep' exemples ayant le score de confiance le plus élevé
    indices = torch.argsort(out['instances'].scores, descending=True)[0:nb_to_keep]
    mask = torch.zeros((out['instances'].pred_boxes.tensor.shape[0],), dtype=torch.bool)
    mask[indices] = True
    new_out['instances'].pred_boxes = out['instances'].pred_boxes[mask]
    new_out['instances'].scores = out['instances'].scores[mask]
    new_out['instances'].pred_masks = out['instances'].pred_masks[mask]
    new_out['instances'].pred_classes = out['instances'].pred_classes[mask]
    return new_out

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

def get_corners_aligned(boxes):
    device = boxes.device
    bev_corners = torch.zeros((len(boxes), 4, 2), dtype=torch.float, device=device)
    bev_corners[:, 0, 0] = boxes[:, 0]
    bev_corners[:, 0, 1] = boxes[:, 1]
    bev_corners[:, 1, 0] = boxes[:, 0]
    bev_corners[:, 1, 1] = boxes[:, 3]
    bev_corners[:, 2, 0] = boxes[:, 2]
    bev_corners[:, 2, 1] = boxes[:, 3]
    bev_corners[:, 3, 0] = boxes[:, 2]
    bev_corners[:, 3, 1] = boxes[:, 1]
    return bev_corners

# Création d'un objet polygon à partir des 4 coins de la boite englobante
def get_polygons(bev_corners):
    list_of_polygons = []
    for i in range(len(bev_corners)):
        list_of_polygons.append(Polygon([(bev_corners[i, j, 0], bev_corners[i, j, 1]) for j in range(4)]).buffer(0))
    return list_of_polygons

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
def filter_boxes_rotated2(image_data, thresh, aligned=False):
    classes_id = image_data['instances'].pred_classes
    boxes = image_data['instances'].pred_boxes.tensor
    indices_to_remove_list = []
    indices_filter = []

    for i in range(2):
        current_mask = (classes_id == i)
        object_indices = torch.nonzero(current_mask)
        boxes_current = boxes[current_mask]
        areas = image_data['instances'].pred_boxes.area()[current_mask].reshape(-1, 1)
        if aligned:
            corners = get_corners_aligned(boxes_current)
        else:
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
    if indices_to_remove_list:
        indices_filter = torch.cat(indices_to_remove_list)
        indices_filter = torch.unique(indices_filter)

    index = torch.ones(boxes.shape[0], dtype=torch.bool)
    index[indices_filter] = False

    image_data['instances'].pred_boxes.tensor = image_data['instances'].pred_boxes.tensor[index]
    image_data['instances'].scores = image_data['instances'].scores[index]
    image_data['instances'].pred_classes = image_data['instances'].pred_classes[index]
    image_data['instances'].pred_masks = image_data['instances'].pred_masks[index]
    
    return image_data

# Cette fonction permet d'effectuer l'inference sur les images présentes dans un dossier
def get_infered_image(png_list, predictors, metadatas, path_out, display_lines, display_labels, percent_to_keep):
    for path in tqdm(png_list):
        image_data = split_image(path)
        pred_boxes = []
        scores = []
        pred_classes = []
        seg = []
        for i in range(5):
            for j in range(5):
                output = {'instances': Instances((512, 512))}
                pred_boxes_list = []
                scores_list = []
                pred_classes_list = []
                seg_list = []
                for idx, class_predictor in enumerate(predictors):
                    out = class_predictor(image_data['patches'].numpy()[i, j, ...])
                    out['instances'].pred_classes = torch.full(out['instances'].pred_classes.shape, idx, device='cpu')
                    if idx == 1:
                        out = drop_sample(out, int(out['instances'].pred_classes.shape[0] * (percent_to_keep / 100)))
                    pred_boxes_list.append(out['instances'].pred_boxes.to('cpu'))
                    pred_classes_list.append(out['instances'].pred_classes.to('cpu'))
                    scores_list.append(out['instances'].scores.to('cpu'))
                    seg_list.append(out['instances'].pred_masks.to('cpu'))
                
                # Fusion des résultats des 3 détecteurs
                output['instances'].pred_boxes = Boxes.cat(pred_boxes_list)
                output['instances'].scores = torch.cat(scores_list)
                output['instances'].pred_classes = torch.cat(pred_classes_list)
                output['instances'].pred_masks = torch.cat(seg_list, dim=0)

                # Mise à l'echelle des coordonnées des boites englobantes
                output['instances'].pred_boxes.tensor[:, 0] += (j * 384)
                output['instances'].pred_boxes.tensor[:, 1] += (i * 384)
                output['instances'].pred_boxes.tensor[:, 2] += (j * 384)
                output['instances'].pred_boxes.tensor[:, 3] += (i * 384)
                # Création du masque pour un objet sous la forme d'une image de taille 2048
                pad = torch.nn.ZeroPad2d((384 * j, 384 * (4 - j), 384 * i, 384 * (4 - i)))
                output['instances'].pred_masks = torch.squeeze(pad(torch.unsqueeze(output['instances'].pred_masks, dim=1)), dim=1)
                pred_boxes.append(output['instances'].pred_boxes)
                scores.append(output['instances'].scores)
                pred_classes.append(output['instances'].pred_classes)
                seg.append(output['instances'].pred_masks)
        
        # Fusion des résultats des 25 patchs
        big_output = {'instances': Instances((2048, 2048))}
        big_output['instances'].pred_boxes = Boxes.cat(pred_boxes)
        big_output['instances'].scores = torch.cat(scores)
        big_output['instances'].pred_classes = torch.cat(pred_classes)
        big_output['instances'].pred_masks = torch.cat(seg)

        # Post traitements
        im = cv2.imread(path)
        big_output = filter_boxes_rotated2(big_output, 0.5, True)
        
        v = MyVisualizer(im[:, :, ::-1], metadata=metadatas, scale=1, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(big_output["instances"].to("cpu"), display_labels)
        out_im = out.get_image()[:, :, ::-1]
        if display_lines:
            out_im = draw_lines(out_im.astype(np.float32)).astype(np.uint8)
        converted = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
        #converted = np.flipud(converted)
        cv2.imwrite(os.path.join(path_out, path.split('/')[-1]), converted)