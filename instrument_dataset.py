
import os
import sys
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


INSTRUMENT_CLASSES = (
    '', 'kidney', 'bipolar_forceps', 'fenestrated_bipolar', 'prograsp_forceps', 'large_needle_driver', 'vessel_sealer',
    'grasping_retractor', 'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier', 'stapler')

ACTION_CLASSES = (
    'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 'Tool_Manipulation', 'Cutting', 'Cauterization'
    , 'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')


class SurgicalDataset18(Dataset):
    def __init__(self, seq_set, clip_length, seq_len, is_train=None):
        self.is_train = is_train
        self.list = seq_set
        self.dir_root_gt = 'instruments18/seq_'
        self.xml_dir_list = []

        for i in self.list:
            xml_dir_temp = self.dir_root_gt + str(i) + '/xml/'
            self.xml_dir_list = self.xml_dir_list + glob(xml_dir_temp + '/*.xml')
            random.shuffle(self.xml_dir_list)

    def __len__(self):
        return len(self.xml_dir_list)

    def __getitem__(self, index):
        file_name = os.path.splitext(os.path.basename(self.xml_dir_list[index]))[0]
        file_root = os.path.dirname(os.path.dirname(self.xml_dir_list[index]))
        _xml = ET.parse(self.xml_dir_list[index]).getroot()
        interaction_to_ind = dict(zip(INSTRUMENT_CLASSES, range(len(INSTRUMENT_CLASSES))))
        _img_dir = os.path.join(file_root, '/left_frames/', file_name + '.png')
        #         _img_orig = Image.open(_img_dir).convert('RGB')
        #         _img = _img_orig.resize((1024, 1024), Image.BILINEAR)
        #         _img_shape = np.array(_img_orig).shape

        class_to_ind = dict(zip(INSTRUMENT_CLASSES, range(len(INSTRUMENT_CLASSES))))
        node_bbox = []
        det_classes = []
        act_classes = []
        for obj in _xml.iter('objects'):
            name = obj.find('name').text.strip()
            interact = obj.find('interaction').text.strip()
            act_classes.append(ACTION_CLASSES.index(str(interact)))
            det_classes.append(INSTRUMENT_CLASSES.index(str(name)))
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            label_idx = class_to_ind[name]
            # interaction_idx = interaction_to_ind[interact]
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                # cur_pt = cur_pt / _img_shape[1] if i % 2 == 0 else cur_pt / _img_shape[0]
                bndbox.append(cur_pt)
            bndbox.append(label_idx)
            node_bbox += [bndbox]

        node_num = len(det_classes)
        human_num = 1
        obj_num = node_num - 1
        adj_mat = np.zeros((node_num, node_num))
        adj_mat[0, :] = act_classes
        adj_mat[:, 0] = act_classes
        adj_mat = adj_mat.astype(int)
        adj_mat[adj_mat > 0] = 1
        node_labels = np.zeros((node_num, len(ACTION_CLASSES)))
        for edge_idx in range(node_num):
            if act_classes[edge_idx] > 0:
                node_labels[0, act_classes[edge_idx]] = 1
                node_labels[edge_idx, act_classes[edge_idx]] = 1
                bndbox = np.hstack((np.minimum(node_bbox[0][:2], node_bbox[edge_idx][:2]),
                                    np.maximum(node_bbox[0][2:], node_bbox[edge_idx][2:])))

        edge_features = np.load(os.path.join(file_root, 'roi_features_ap-mtl', '{}_edge_features.npy').format(file_name))
        node_features = np.load(os.path.join(file_root, 'roi_features_ap-mtl', '{}_node_features.npy').format(file_name))
        #         _bbox = torch.from_numpy(np.asarray(_bbox, np.float32)).float()
        #         _img = np.asarray(_img, np.float32)/255
        #         _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()

        return edge_features, node_features, adj_mat, node_labels, file_name, human_num, obj_num