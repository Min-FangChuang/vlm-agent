import json
import numpy as np
import os
import torch


BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pc(scene_id):
    root_dir = BENCHMARK_DIR
    pcds, _, _, instance_labels = torch.load(
        os.path.join(root_dir, 'pcd_with_global_alignment', '%s.pth' % scene_id))
    inst_to_name = json.load(open(os.path.join(root_dir, 'instance_id_to_name', '%s.json' % scene_id)))

    obj_labels = []
    inst_locs = []
    obj_ids = []
    
    for i, obj_label in enumerate(inst_to_name):
        if obj_label in ['wall', 'floor', 'ceiling']:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, 'scene: %s, obj %d' % (scene_id, i)
        
        obj_pcd = pcds[mask]
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        obj_labels.append(obj_label)
        obj_ids.append(i)

    return obj_ids, obj_labels, inst_locs


def calc_iou(box_a, box_b):
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    
    return 1.0 * intersection / union
