# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import argparse
import json
from glob import glob
import os
import re

def ground_truth_parser(file, image_numbers):
    with open(file) as f:
        data = json.load(f)
        if int([x.split('_') for x in data['Image'].split('.')][0][-1]) not in image_numbers:
            objects = data['Objets']
            x = np.array([b['x'] for b in objects]).reshape(-1, 1)
            y = np.array([b['y'] for b in objects]).reshape(-1, 1)
            coords_gt = np.concatenate([x, y], axis=1)
            return coords_gt
    return None

def algo_parser(file, image_numbers):
    coords = []
    with open(file) as f:
        data = json.load(f)
        detections = data['Detections']
        algo_type = data['Type']
        number_of_objects = [x['Nombre_objet'] for x in detections if int(x['Image'].split('.')[0].split('_')[-1]) not in image_numbers]
        cumulative = np.insert(np.cumsum(number_of_objects), 0, 0)
        print(cumulative)
        objects = [x['Objects'] for x in detections if int(x['Image'].split('.')[0].split('_')[-1]) not in image_numbers]
        x = [a['x'] for b in objects for a in b]
        y = [a['y'] for b in objects for a in b]
        x = [np.array(x[n:t]).reshape(-1, 1) for (n, t) in zip(cumulative, cumulative[1:])]
        y = [np.array(y[n:t]).reshape(-1, 1) for (n, t) in zip(cumulative, cumulative[1:])]
        coords = [np.concatenate([a, b], axis=1) for (a, b) in zip(x, y)]
    return coords, algo_type

def euclidean_distance(arr):
    return np.sqrt(arr[0] + arr[1])

'''def get_metrics(coords, coords_gt_list, delta):
    tp = 0
    fp = 0
    fn = 0
    for image_nb in range(len(coords)):
        a = coords[image_nb]
        b = coords_gt_list[image_nb]
        detections_axis = 1
        gt_axis = 0
        a = np.expand_dims(a, axis=detections_axis)
        b = np.expand_dims(b, axis=gt_axis)
        new_shape = np.broadcast_shapes(a.shape, b.shape)
        a_broadcasted = np.broadcast_to(a, new_shape)
        b_broadcasted = np.broadcast_to(b, new_shape)
        res = np.square(a_broadcasted - b_broadcasted)
        res = np.apply_along_axis(euclidean_distance, 2, res)
        indices = np.unravel_index(np.argmin(res), shape=res.shape)
        while np.any(np.all(indices == np.argwhere(res < delta), axis=1)):
            tp += 1
            res = np.delete(res, indices[0], axis=0)
            res = np.delete(res, indices[1], axis=1)
            if res.size > 0:
                indices = np.unravel_index(np.argmin(res), shape=res.shape)
            else:
                break
        print(res)
        fp += res.shape[0]
        fn += res.shape[1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    return [precision, recall, f1_score]'''

def get_metrics(coords, coords_gt_list, delta):
    tp = 0
    fp = 0
    fn = 0
    for image_nb in range(len(coords)):
        detections = np.copy(coords[image_nb])
        gt = np.copy(coords_gt_list[image_nb])
        index = 0
        while index < len(coords[image_nb]) and len(gt) > 0:
            distances = np.sqrt(np.sum(np.square(coords[image_nb][index] - gt), axis=1))
            match_index = np.argmin(distances)
            if distances[match_index] < delta:
                tp += 1
                gt = np.delete(gt, match_index, axis=0)
                detections = np.delete(detections, index, axis=0)
            index += 1
        fp += len(detections)
        fn += len(gt)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    return [precision, recall, f1_score]


'''def get_metrics(coords, coords_gt_list, delta):
    detections_axis = -1
    gt_axis = -2
    coords = np.expand_dims(coords, axis=detections_axis)
    coords_gt_list = np.expand_dims(coords_gt_list, axis=gt_axis)
    new_shape = np.broadcast_shapes(coords.shape, coords_gt_list.shape)
    print(new_shape)
    values_broadcasted = np.broadcast_to(coords, new_shape)
    gt_broadcasted = np.broadcast_to(coords_gt_list, new_shape)
    res = np.abs(values_broadcasted - gt_broadcasted)
    res = np.all(res < delta, axis=1)
    print(res.shape)
    positive_axis = np.argmax(np.array(res.shape)[1:]) + 1
    tp = np.any(res, axis=positive_axis).sum()
    fp = np.all(~res, axis=detections_axis).sum()
    fn = np.all(~res, axis=gt_axis).sum()
    print(tp)

def resize(algo_data):
    algo_data = [np.expand_dims(x, 0) for x in algo_data]
    new_shape = np.broadcast_shapes(*[x.shape for x in algo_data])
    algo_data = [np.broadcast_to(x, new_shape) for x in algo_data]
    algo_data = np.concatenate(algo_data, 0)
    algo_data = np.swapaxes(algo_data, 1, 2)
    return algo_data'''

# %%
the_parser = argparse.ArgumentParser("Comparison between classic algorithm")
the_parser.add_argument('config_file', help="the directory where to find json files")
args = the_parser.parse_args()

# %%
with open(args.config_file) as f:
    data = json.load(f)
    algo_dic = {}
    gt_list = []
    files = glob(os.path.join(data['input_folder'], "*"))
    gt_files = [x for x in files if 'Objets_simules' in x]
    algo_files = [x for x in files if 'Objets_simules' not in x and x.split('.')[-1] == 'json']
    gt_files = sorted(gt_files, key=lambda x: int(re.split('[_.]', x)[-2]))
    for gt_file in gt_files:
        res = ground_truth_parser(gt_file, data['images_no_comparisons'])
        if res is not None:
            gt_list.append(res)
    for algo_file in algo_files:
        algo_data, algo_type = algo_parser(algo_file, data['images_no_comparisons'])
        algo_dic[algo_type] = algo_data
    with open(os.path.join(data['output_directory'], "comparisons.txt"), 'w') as out_file:
        for algo_type, algo_data in algo_dic.items():
            #print(algo_type)
            metrics = get_metrics(algo_data, gt_list, data['margin'])
            out_file.write(algo_type + '\n\n')
            out_file.write(f'Precision: {metrics[0]}\n')
            out_file.write(f'Recall: {metrics[1]}\n')
            out_file.write(f'F1 score: {metrics[2]}\n')
            out_file.write('\n\n')


# %%



# %%



# %%



