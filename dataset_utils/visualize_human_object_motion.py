import os
from os.path import join, isfile
import sys
sys.path.append("..")
import numpy as np
import pickle
import torch
from torch import nn
import pytorch3d
import pytorch3d.io as IO
import trimesh
from smplx import smplx
import cv2
import imageio
from utils.txt2intrinsic import txt2intrinsic
from utils.pyt3d_wrapper import Pyt3DWrapper
from utils.avi2depth import avi2depth
from utils.time_align import time_align
from utils.process_timestamps import txt_to_paried_frameids, paired_frameids_to_txt
from utils.contact import compute_contact
from utils.VTS_object import get_obj_info
from utils.visualization import save_mesh
from utils.load_smplx_params import load_multiperson_smplx_params
from utils.object_retargeting import obj_retargeting
from utils.contact import compute_contact_and_closest_point
from smplx.smplx.lbs import batch_rodrigues
from transforms3d.axangles import mat2axangle
import open3d as o3d
from optimization.bvh2smplx import Simple_SMPLX, create_SMPLX_model
from tqdm import tqdm
import torchvision.io as io
from moviepy.editor import VideoFileClip, clips_array
from utils.simplify_mesh import simplify_mesh
from tqdm import tqdm



