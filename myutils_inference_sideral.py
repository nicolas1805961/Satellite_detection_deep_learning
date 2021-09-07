# import some common libraries
import numpy as np
import cv2
import torch
from tqdm import tqdm
import os

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode, Instances, RotatedBoxes

from myutils_inference_earth import split_image, MyVisualizer, draw_lines, get_corners, get_my_metric, get_polygons

# Fonction permettant de supprimer les détections superflus dans les zones de recouvrement
def filter_boxes_rotated2(image_data, thresh, nb_classes=2):
    classes_id = image_data['instances'].pred_classes
    boxes = image_data['instances'].pred_boxes.tensor
    indices_to_remove_list = []

    for i in range(nb_classes):
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
    index = torch.ones(boxes.shape[0], dtype=torch.bool)
    if indices_to_remove_list:
        indices_filter = torch.cat(indices_to_remove_list)
        indices_filter = torch.unique(indices_filter)
        index[indices_filter] = False

    image_data['instances'].pred_boxes.tensor = image_data['instances'].pred_boxes.tensor[index]
    image_data['instances'].scores = image_data['instances'].scores[index]
    image_data['instances'].pred_classes = image_data['instances'].pred_classes[index]
    
    return image_data

# Cette fonction permet d'effectuer l'inférence sur les images comprises dans un dossier
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
                    if idx == 1:
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
        #print(len(big_output['instances'].pred_boxes.tensor[big_output['instances'].pred_classes == 0]))
        #big_output = filter_ponctuels(big_output, 0.5, 0.85)

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