#System
import numpy as np
import sys
import os
import cv2
from PIL import Image
from glob import glob
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import torchvision.models
import torch
from roi_feature_model import Resnet152

INSTRUMENT_CLASSES = (
'', 'kidney', 'bipolar_forceps', 'fenestrated_bipolar', 'prograsp_forceps', 'large_needle_driver', 'vessel_sealer',
'grasping_retractor', 'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier', 'stapler')

ACTION_CLASSES = (
'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 'Tool_Manipulation', 'Cutting', 'Cauterization'
, 'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')    
    
mlist = [1,2,3,4,5,6,7,9,10,11,12,14,15,16]
dir_root_gt = 'instruments18/seq_'
xml_dir_list = []

for i in mlist:
    xml_dir_temp = dir_root_gt + str(i) + '/xml/'
    seq_list_each = glob(xml_dir_temp + '/*.xml')
    xml_dir_list = xml_dir_list + seq_list_each

    
class_to_ind = dict(zip(INSTRUMENT_CLASSES, range(len(INSTRUMENT_CLASSES))))

for index, _xml_dir in  enumerate(xml_dir_list):
    _xml_dir = 'instruments18/seq_5/xml/frame040.xml'
    #print(index, _xml_dir)
    img_name = os.path.basename(xml_dir_list[index][:-4])
    _img_dir = os.path.dirname(os.path.dirname(xml_dir_list[index])) + '/left_frames/' + img_name + '.png'
    save_data_path = os.path.join(os.path.dirname(os.path.dirname(xml_dir_list[index])),'roi_features')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    
    
    _img = Image.open(_img_dir).convert('RGB')
    _xml = ET.parse(_xml_dir).getroot()
    
    det_classes = []
    act_classes = []
    node_bbox = []
    det_boxes_all = []
    c_flag = False
    for obj in _xml.iter('objects'):
#         try:
#             name = obj.find('name').text.strip()
#             bbox = obj.find('bndbox')
#             interact = obj.find('interaction').text.strip()
#             act_classes.append(ACTION_CLASSES.index(str(interact)))
#             det_classes.append(INSTRUMENT_CLASSES.index(str(name)))
#         except:
#             print(_xml_dir)
#             c_flag = True
#             break
            
            
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        interact = obj.find('interaction').text.strip()
        act_classes.append(ACTION_CLASSES.index(str(interact)))
        det_classes.append(INSTRUMENT_CLASSES.index(str(name)))
        
        bbox_col = INSTRUMENT_CLASSES.index(str(name)) - 1;
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        label_idx = class_to_ind[name]

        for i, pt in enumerate(pts):         
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        node_bbox += [bndbox]
        det_boxes_all.append(np.array(bndbox))
    
    if c_flag:
        continue
    node_num = len(act_classes)
    instrument_num = node_num - 1
    adj_mat = np.zeros((node_num, node_num))
    adj_mat[0, :] = act_classes
    adj_mat[:, 0] = act_classes
    adj_mat = adj_mat.astype(int)
    adj_mat[adj_mat > 0] = 1
    
    node_labels = np.zeros((node_num, len(ACTION_CLASSES)))
    for edge_idx in range(node_num):
        if act_classes[edge_idx] > 0:
            node_labels[0, act_classes[edge_idx]] = 1
            node_labels[edge_idx,act_classes[edge_idx]] = 1
            bndbox = np.hstack((np.minimum(node_bbox[0][:2], node_bbox[edge_idx][:2]),
                            np.maximum(node_bbox[0][2:], node_bbox[edge_idx][2:])))
            
            det_boxes_all.append(bndbox)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    
    input_h, input_w = 224, 224
    feature_network = Resnet152(num_classes=13).cuda()
    
    # roi features extraction
    node_features = np.zeros((np.array(node_bbox).shape[0], 200))
    edge_features = np.zeros((node_num, node_num, 200))
    roi_idx = 0
    adj_idx = np.where(adj_mat[0, :] == 1)[0]
    edge_idx = 0
    _img = np.array(_img)
    for bndbox in det_boxes_all:
        roi = np.array(bndbox).astype(int)
        roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
        # plt.imshow(roi_image)
        # plt.show()
        roi_image = transform(cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))

        roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
        feature = feature_network(roi_image)
        if roi_idx < node_num:
            node_features[roi_idx, ...] = feature.data.cpu().numpy()
        else:
            edge_features[0, adj_idx[edge_idx]] = feature.data.cpu().numpy()
            edge_features[adj_idx[edge_idx], 0] = feature.data.cpu().numpy()
            edge_idx += 1
        roi_idx += 1
    #print(save_data_path)
    np.save(os.path.join(save_data_path, '{}_edge_features'.format(img_name)), edge_features)
    np.save(os.path.join(save_data_path, '{}_node_features'.format(img_name)), node_features)