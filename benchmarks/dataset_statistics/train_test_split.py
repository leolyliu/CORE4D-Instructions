import os
from os.path import join, isfile, isdir, dirname, abspath
import sys
import numpy as np
import json
sys.path.insert(0, join(dirname(abspath(__file__)), ".."))  # HHO-dataset
from data_processing.utils.VTS_object import get_obj_info
# from data_processing.utils.load_smplx_params import load_multiperson_smplx_params


TRAIN_OBJECTS = {
    "chair006", "chair021", "chair022",
    "desk001", "Desk020", "Desk021", "Desk023",
    "box001", "Box001", "box004", "Box004", "Box021", "Box022", "Box023", "Box025",
    "Board001", "board005", "Board020",
    "bucket001", "bucket003", "bucket004", "bucket005", "bucket007", "Bucket007", "bucket010",
    "stick006", "stick008",
}
TEST_OBJECTS = {
    "chair005", "chair020",
    "desk005", "desk007",
    "Box020", "Box024", "Box026",
    "board007", "board021",
    "bucket006", "Bucket006", "bucket008", "bucket009",
    "stick001", "stick003",
}


def train_test_split(dataset_root, obj_dataset_dir, clip_names):
    
    train_sequence_names, test_sequence_names = [], []
    
    obj_dict = {}
    
    for clip_name in clip_names:
        clip_dir = join(dataset_root, clip_name)
        for seq_name in os.listdir(clip_dir):
            if len(seq_name) != 3:
                continue
            seq_dir = join(clip_dir, seq_name)
            if not isdir(seq_dir):
                continue
            if not isfile(join(seq_dir, "VTS_data.npz")):
                continue
            if isfile(join(seq_dir, "deleted")):
                continue
            
            obj_name, obj_model_path = get_obj_info(seq_dir, obj_dataset_dir)
            json.dump({"obj_name": obj_name, "obj_model_path": obj_model_path}, open(join(seq_dir, "object_metadata.json"), "w"))
            print("ok {} {} ...".format(seq_dir, obj_name))

            if not obj_name in obj_dict:
                obj_dict[obj_name] = 0
            obj_dict[obj_name] += 1
            
            if obj_name in TRAIN_OBJECTS:
                train_sequence_names.append(clip_name + "." + seq_name)
            if obj_name in TEST_OBJECTS:
                test_sequence_names.append(clip_name + "." + seq_name)
    
    print(obj_dict)
    
    print("training sequence number = ", len(train_sequence_names))
    print("test sequence number = ", len(test_sequence_names))
    
    return train_sequence_names, test_sequence_names


def load_train_test_split():
    train_sequence_names = json.load(open(join(dirname(abspath(__file__)), "train_sequence_names.json"), "r"))
    test_sequence_names_seen_obj = json.load(open(join(dirname(abspath(__file__)), "test_sequence_names_seen_obj.json"), "r"))
    test_sequence_names_unseen_obj = json.load(open(join(dirname(abspath(__file__)), "test_sequence_names_unseen_obj.json"), "r"))
    return train_sequence_names, test_sequence_names_seen_obj, test_sequence_names_unseen_obj


def load_train_test_split_retargeted():
    train_sequence_names = []  # TODO: add sequence names from CORE4D_Synthetic
    return train_sequence_names
