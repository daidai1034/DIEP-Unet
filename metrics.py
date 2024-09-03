import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from scipy.spatial.distance import directed_hausdorff

def get_accuracy(output, target):
    output_binary = (output > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    correct_pixels = np.sum(output_binary == target_binary)
    total_pixels = output_binary.size

    acc = correct_pixels / total_pixels

    return acc

def get_sensitivity(output, target):
    # Sensitivity == Recall
    intersection = np.sum(output * target)
    # print('target:',np.max(target),np.min(target))
    # print('intersection:',np.max(intersection),np.min(intersection))
    SE = intersection/ (np.sum(target) + 1e-8)

    return SE


def get_specificity(output, target):
    output_binary = (output > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    true_negative = np.sum((output_binary == 0) & (target_binary == 0))
    false_positive = np.sum((output_binary == 1) & (target_binary == 0))

    try:
        SP = true_negative / (true_negative + false_positive)
    except ZeroDivisionError:
        SP = 0.0

    return SP


def get_precision(output, target):
    intersection = np.sum(output * target)
    PC = intersection / (np.sum(output) + 1e-8)

    return PC


def get_F1(output, target):
    # Sensitivity == Recall
    PC = get_precision(output, target)
    SE = get_sensitivity(output, target)
    F1 = F1 = 2 * SE * PC / (SE + PC + 1e-8)
    return F1



def get_JS(output, target):
    # JS : Jaccard similarity
    intersection = np.sum(output * target)
    union = np.sum(output) + np.sum(target) - intersection
    JS = intersection / (union + 1e-8)

    return JS

def get_DC(output, target):
    intersection = (output * target).sum()
    DC = (2. * intersection + 1e-8) / (output.sum() + target.sum() + 1e-8)
    return DC

def get_HD(output, target):
    batch_size = output.shape[0]
    
    # Reshape output and target to 2D matrices
    output_2d = output.reshape(batch_size, -1)
    target_2d = target.reshape(batch_size, -1)
    
    distances = []
    for i in range(batch_size):
        # Convert 1D vectors to 2D matrices with a single row
        o_matrix = np.expand_dims(output_2d[i], axis=0)
        t_matrix = np.expand_dims(target_2d[i], axis=0)
        
        distance_1 = directed_hausdorff(o_matrix, t_matrix)[0]
        distance_2 = directed_hausdorff(t_matrix, o_matrix)[0]
        distances.append(max(distance_1, distance_2))
    
    HD = np.max(distances)
    return HD

def get_HD95(output, target):
    distances = []
    for o in output:
        for t in target:
            distance = np.linalg.norm(o - t)  # 使用欧几里德距离计算距离
            distances.append(distance)
    
    distances.sort()
    threshold_index = int(len(distances) * 0.95)
    HD95 = distances[threshold_index]
    
    return HD95