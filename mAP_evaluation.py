'''
Parallel 3D mAP calculation for the data in nuScenes format.

File formats:

`pred_file`: json file, predictions in global frame, in the format of:

predictions = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
    'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
    'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
    'name': 'car',
    'score': 0.3077029437237213
}]

`gt_file`: ground truth annotations in global frame, in the format of:


gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
}]

NOTICE both are lists of dicts (annotations).

'''

import json
import fire
from pathlib import Path
import numpy as np
from multiprocessing import Process

from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions


def save_AP(gt, predictions, class_names, iou_threshold, output_dir):
    ''' computes average precisions (AP) for a given threshold, and saves the metrics in a temp file '''
    # use lyft's provided function to compute AP
    AP = get_average_precisions(gt, predictions, class_names, iou_threshold)
    # create a dict with keys as class names and values as their respective APs
    metric = {c:AP[idx] for idx, c in enumerate(class_names)}

    # save the dict in a temp file
    summary_path = output_dir / f'metric_summary_{iou_threshold}.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f)


def get_metric_overall_AP(iou_th_range, output_dir, class_names):
    ''' reads temp files and calculates overall per class APs.

    returns:
        `metric`: a dict with key as iou thresholds and value as dicts of class and their respective APs,
        `overall_AP`: overall AP of each class
    '''

    metric = {}
    overall_AP = np.zeros(len(class_names))
    for iou_threshold in iou_th_range:
        summary_path = output_dir / f'metric_summary_{iou_threshold}.json'
        with open(str(summary_path), 'r') as f:
            data = json.load(f) # type(data): dict
            metric[iou_threshold] = data
            overall_AP += np.array([data[c] for c in class_names])
        summary_path.unlink() # delete this temp file
    overall_AP /= len(iou_th_range)
    return metric, overall_AP


def main(gt_file, pred_file, output_dir):
    '''
    Main function to compute mAP, metrics are saved in `metric_summary.json` file

    args:
    gt_file: json file path with ground truth annotations
    pred_file: json file path with predicted annotations
    output_dir: the final computed metrics are saved in this directory as a json file
    '''
    print('Starting mAP computation')

    gt_path = Path(gt_file)
    pred_path = Path(pred_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(pred_path)) as f:
        predictions = json.load(f)

    with open(str(gt_path)) as f:
        gt = json.load(f)

    class_names = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle',
                    'motorcycle', 'other_vehicle', 'pedestrian', 'truck']

    iou_th_range = np.linspace(0.5, 0.95, 10) # 0.5, 0.55, ..., 0.90, 0.95

    metric = {}

    # create and start parallel processes
    processes = []
    for iou_threshold in iou_th_range:
        process = Process(target=save_AP, args=(gt, predictions, class_names, iou_threshold, output_dir))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    # get overall metrics
    metric, overall_AP = get_metric_overall_AP(iou_th_range, output_dir, class_names)
    metric['overall'] = {c: overall_AP[idx] for idx, c in enumerate(class_names)}
    metric['mAP'] = np.mean(overall_AP)

    summary_path = Path(output_dir) / 'metric_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f, indent=4)

    print(f'Done!, Final metrics saved at {str(summary_path)}')

if __name__ == "__main__":
    fire.Fire(main)

