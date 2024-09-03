import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def dice_coeff(pred, mask):
    # pred = torch.sigmoid(pred)
    # print("dice_pred:",torch.max(pred),torch.min(pred))
    smooth = 1.
    # y_pred = y_pred.float()
    # y_true = y_true.float()
    intersection = torch.sum(pred * mask)
    score = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(mask) + smooth)
    # print('score:',torch.max(score),torch.min(score))
    return score

def dice_loss(pred, mask):
    # print("dice loss_pred:",torch.max(pred),torch.min(pred))
    loss = 1 - dice_coeff(pred, mask)
    return loss

def bce_dice_loss(pred, mask):
   
    loss = 0.5 * nn.functional.binary_cross_entropy(pred,mask) + 0.5 * dice_loss(pred, mask)

    return loss

