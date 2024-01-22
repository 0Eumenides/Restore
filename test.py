import torch
import smplx
from pycocotools.coco import COCO
import json
import numpy as np

target_subject = 1
target_action = 2
target_subaction = 1
target_frame = 0
target_cam = 1

db = COCO('/data/dth/h3.6m/annotations/Human36M_subject' + str(target_subject) + '_data.json')
# camera load
with open('/data/dth/h3.6m/annotations/Human36M_subject' + str(target_subject) + '_camera.json', 'r') as f:
    cameras = json.load(f)
# joint coordinate load
with open('/data/dth/h3.6m/annotations/Human36M_subject' + str(target_subject) + '_joint_3d.json', 'r') as f:
    joints = json.load(f)
# smpl parameter load
with open('/data/dth/h3.6m/SMPL/Human36M_subject' + str(target_subject) + '_SMPL_NeuralAnnot.json', 'r') as f:
    smpl_params = json.load(f)

smpl_path = '/data/dth/h3.6m/SMPL/smpl_neutral_lbs_10_207_0_v1.0.0.pkl'

smpl_layer = smplx.create(smpl_path, 'smpl')

smpl_param = smpl_params[str(target_action)][str(target_subaction)][str(target_frame)]
pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
pose = torch.FloatTensor(pose).view(-1, 3)  # (24,3)
root_pose = pose[0, None, :]
body_pose = pose[1:, :]
shape = torch.FloatTensor(shape).view(1, -1)  # SMPL shape parameter
trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

