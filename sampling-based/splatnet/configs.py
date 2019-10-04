"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys

# project root
ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..'))

# required for custom layers
sys.path.append(os.path.join(ROOT_DIR, 'splatnet'))
sys.path.append(os.path.join(ROOT_DIR, 'splatnet', 'dataset'))

# modify these if you put data in non-default locations
FACADE_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'ruemonge428')
SHAPENET3D_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'shapenet_ericyi_ply')
SHAPENET2D3D_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'shapenet_2d3d_h5')
LIDAR3D_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'stanford_indoor3d')
MODELNET_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'modelnet40_ply_hdf5_2048')
# facade global variables

FACADE_CATEGORIES = ('wall', 'sky', 'balcony', 'window', 'door', 'shop', 'roof')

FACADE_CMAP = ((255, 255, 0),
               (128, 255, 255),
               (128, 0, 255),
               (255, 0, 0),
               (255, 128, 0),
               (0, 255, 0),
               (0, 0, 255))

# shapenet global variables

SN_CATEGORIES = ('03642806', '02958343', '04225987', '03001627', '03261776', '03790512', '03797390', '02691156',
                 '03948459', '03636649', '02773838', '02954340', '04099429', '03467517', '04379243', '03624134')

SN_CATEGORY_NAMES = ('laptop', 'car', 'skateboard', 'chair', 'earphone', 'motorbike', 'mug', 'airplane',
                     'pistol', 'lamp', 'bag', 'cap', 'rocket', 'guitar', 'table', 'knife')

SN_NUM_PART_CATEGORIES = (2, 4, 3, 4, 3, 6, 2, 4,
                          3, 4, 2, 2, 3, 3, 3, 2)

SN_CMAP = ((255, 255, 0),
           (128, 255, 255),
           (128, 0, 255),
           (255, 0, 0),
           (255, 128, 0),
           (0, 255, 0),
           (0, 0, 255))

# stanford indoor 3d variables

g_classes = [x.rstrip() for x in open(os.path.join(LIDAR3D_DATA_DIR, 'class_names.txt'))]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {'ceiling': [0,255,0],
         'floor': [0,0,255],
         'wall':  [0,255,255],
         'beam':        [255,255,0],
         'column':      [255,0,255],
         'window':      [100,100,255],
         'door':        [200,200,100],
         'table':       [170,120,200],
         'chair':       [255,0,0],
         'sofa':        [200,100,100],
         'bookcase':    [10,200,100],
         'board':       [200,200,200],
         'clutter':     [50,50,50]} 
g_easy_view_labels = [7,8,9,10,11,1]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

# LIDAR_CATEGORIES