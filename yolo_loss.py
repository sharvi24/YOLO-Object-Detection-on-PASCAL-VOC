#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:03:34 2022

@author: sharvitomar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here

        for i in range(boxes.size()[0]):
            w = boxes[i, 2]
            h = boxes[i, 3]
                     
            boxes[i,0] = boxes[i,0] / self.S - 0.5 * w
            boxes[i,1] = boxes[i,1] / self.S - 0.5 * h
            boxes[i,2] = boxes[i,0] / self.S + 0.5 * w
            boxes[i,3] = boxes[i,1] / self.S + 0.5 * h

        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 4) ...]
        box_target : (tensor)  size (-1, 5)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        
#         pred_box_list.shape:  torch.Size([2, 78, 5])
        # boxes1.shape:  torch.Size([78, 4])
        # boxes2.shape:  torch.Size([78, 4])
        # box_target.shape:  torch.Size([78, 4])
        # iou_1.shape:  torch.Size([78, 78]) # 
        # best_boxes.shape:  torch.Size([6084])
        # best_ious.shape:  torch.Size([6084])
        # best_boxes[0] :  tensor(0.2505) 
        #print("pred_box_list.shape: ", pred_box_list.shape)
        
        boxes1 = self.xywh2xyxy(pred_box_list[0][:,:4])
        boxes2 = self.xywh2xyxy(pred_box_list[1][:,:4])
        target_box = self.xywh2xyxy(box_target)

#        print("boxes1.shape: ", boxes1.shape)
#        print("boxes2.shape: ", boxes2.shape)
#        print("box_target.shape: ", box_target.shape)

        iou_1 = compute_iou(boxes1, target_box)
        #print("iou_1.shape: ", iou_1.shape)
        # print("iou_1", iou_1)
        # iou_1 = torch.diag(iou_1, 0)

        # print("after iou_1", iou_1)
        iou_2 = compute_iou(boxes2, target_box)
        # iou_2 = torch.diag(iou_2, 0)

        #print("iou_1.shape: ", iou_1.shape)
        best_ious = []

        best_boxes = torch.zeros((target_box.size()[0], 5))
        #print("before best_boxes.shape: ", best_boxes.shape)

        for i in range(iou_1.shape[0]):
            if max(iou_1[i]) > max(iou_2[i]):
                best_boxes[i,:4] = boxes1[i]
                best_ious.append(max(iou_1[i]))
                #best_boxes[i][4] = max(iou_1[i])
                best_boxes[i][4] = pred_box_list[0][i,4]
            else:
                best_boxes[i,:4] = boxes2[i]
                best_ious.append(max(iou_2[i]))
                #best_boxes[i][4] = max(iou_2[i])
                best_boxes[i][4] = pred_box_list[1][i,4]

#        print("after best_boxes.shape: ", best_boxes.shape)
#        print("\n\n best_boxes : ", best_boxes)
#        print("\n\n")
        best_ious = torch.Tensor(best_ious)
        
        # best_boxes = torch.cat((best_boxes, best_ious), 0)
        # best_boxes = torch.Tensor(best_boxes)

#        print("best_boxes.shape: ", best_boxes.shape) # 78 x 5
#        print("best_ious.shape: ", best_ious.shape) # 78 x 1
#        print("best_boxes[0] : ", best_boxes[0])
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
      
        #classes_pred = classes_pred[has_object_map==True]
        # classes_target = classes_target
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss = F.mse_loss(classes_pred, classes_target, reduction = 'sum').to(device)
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n, p = pred_boxes_list[0][has_object_map == False].shape
#
#        has_no_object_map = (~has_object_map).to(torch.int)
#
#        print("pred_boxes_list[0].shape ", pred_boxes_list[0].shape)
#        print("prob dimension ", pred_boxes_list[0][has_object_map == False].size())
#        print("prob dimension exact", pred_boxes_list[0][has_object_map == False][:,4].size())
#        print("dimension of having object ", pred_boxes_list[0][has_object_map == False].size())
        
        l1 = F.mse_loss(pred_boxes_list[0][has_object_map == False][:,4], torch.zeros(n), reduction = 'sum')
        l2 = F.mse_loss(pred_boxes_list[1][has_object_map == False][:,4], torch.zeros(n), reduction = 'sum')
        
        final_loss = (l1 + l2).to(device)
     

        return final_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss = F.mse_loss(box_pred_conf, box_target_conf, reduction = 'sum').to(device)
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        pred_xy = box_pred_response[:,:2]
        pred_wh = torch.sqrt(box_pred_response[:,2:])
        #print("pred_xy size ",pred_xy.size())
        #print("pred_wh size ",pred_wh.size())
        
        
        target_xy = box_target_response[:,:2]
        target_wh = torch.sqrt(box_target_response[:,2:])
        #print("target_xy size ",target_xy.size())
        #print("target_wh size ",target_wh.size())
        
        loss_xy = F.mse_loss(pred_xy, target_xy, reduction = 'sum')
        loss_wh = F.mse_loss(pred_wh, target_wh, reduction = 'sum') 

        loss =  (loss_xy + loss_wh).to(device)       
        return loss

    

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        
        pred_boxes_list = torch.stack((pred_tensor[:,:,:,:5], pred_tensor[:,:,:,5:10]))
        pred_cls = pred_tensor[:,:,:,10:]
        

        # compute classification loss
        pred_cls_with_object = pred_tensor[has_object_map==True][:,10:]
        target_cls_with_object = target_cls[has_object_map==True][:,:]
        
        class_loss = self.get_class_prediction_loss(pred_cls_with_object, target_cls_with_object, has_object_map)
        #print("the class loss is",class_loss )
        
        
        # compute no-object loss
        no_object_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)
        #print("the no_object_loss is", no_object_loss)
        #        print("pred_boxes_list size ", pred_boxes_list.size())
#        print("pred_boxes_list 0 shape ", pred_boxes_list[0].shape)


        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        
        # Only the one with object map value True
        pred_boxes = torch.stack((pred_boxes_list[0][has_object_map==True][:,:5], pred_boxes_list[1][has_object_map==True][:,:5]))
        target_boxes = target_boxes[has_object_map==True][:,:]
        


        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes, target_boxes)

        
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        #         box_pred_response : (tensor) size (-1, 4)
        #         box_target_response : (tensor) size (-1, 4)
        regression_loss = self.get_regression_loss(best_boxes[:,:4], target_boxes)
        #print(regression_loss)

        # compute contain_object_loss
#        both_box_target_conf = torch.stack((pred_boxes_list[0][has_object_map==True][:,4], pred_boxes_list[1][has_object_map==True][:,4]))
#        max_box_target_conf= torch.max(both_box_target_conf, 0)[0]
        contain_obj_loss = self.get_contain_conf_loss(best_ious, best_boxes[:,4])
        #print(contain_obj_loss)

        # compute final loss
        total_loss = (self.l_coord * regression_loss + contain_obj_loss + self.l_noobj * no_object_loss + class_loss) / N  
        #print(total_loss)
        # construct return loss_dict
        loss_dict = dict(
            total_loss = total_loss,
            reg_loss = regression_loss/ N,
            containing_obj_loss = contain_obj_loss/ N,
            no_obj_loss = no_object_loss/ N,
            cls_loss = class_loss/ N
        )
        return loss_dict