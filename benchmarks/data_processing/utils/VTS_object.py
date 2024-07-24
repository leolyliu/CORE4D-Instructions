import os
from os.path import join, isfile
import sys
sys.path.append("..")
import numpy as np


# def get_obj_name_correspondance():
#     corr = {
#         "chair": "椅子",
#         "desk": "桌子",
#         "box": "箱子",
#         "board": "板子",
#         "bucket": "bucket",
#         "stick": "stick",
#         "Desk": "桌子",
#         "Box" :"箱子",
#         "Board": "板子",
#         "Broad" : "板子",
#         "Bucket": "bucket",
#     }
#     return corr


def get_obj_info(data_dir, obj_dataset_dir):
    VTS_path = join(data_dir, "VTS_data.npz")
    assert isfile(VTS_path), "VTS_data.npz not found in {}".format(data_dir)
    VTS_data = np.load(join(data_dir, "VTS_data.npz"), allow_pickle=True)["data"].item()
    if (not "/labels" in VTS_data) or (not "/rigid" in VTS_data):  # no interacted object
        return None, None
    labels = VTS_data["/labels"]

    # get object name
    N_obj = {}
    for device_names in labels:
        for device_name in device_names:
            if not device_name in N_obj:
                N_obj[device_name] = 0
            N_obj[device_name] += 1
    mx, obj_name = 0, None
    
    N_obj["action1"] = 0
    
    for name in N_obj:
        if mx < N_obj[name]:
            mx = N_obj[name]
            obj_name = name
            
    if obj_name is None:  # no interacted object
        return None, None
    
    obj_name = obj_name.replace("road", "oard")
    # corr = get_obj_name_correspondance()
    # obj_model_path = join(obj_dataset_dir, obj_name[:-3], corr[obj_name[:-3]] + str(int(obj_name[-3:])), obj_name[:-3] + str(int(obj_name[-3:])) + "_m.obj")
   
    # if not isfile(obj_model_path):
    #     obj_model_path = join(obj_dataset_dir, corr[obj_name[:-3]], corr[obj_name[:-3]] + str(int(obj_name[-3:])), obj_name[:-3] + str(int(obj_name[-3:])).zfill(3) + "_m.obj")
    # if not isfile(obj_model_path):
    #     obj_model_path = join(obj_dataset_dir, corr[obj_name[:-3]], corr[obj_name[:-3]] + str(int(obj_name[-3:])).zfill(3), obj_name[:-3] + str(int(obj_name[-3:])) + "_m.obj")
    # if not isfile(obj_model_path):
    #     obj_model_path = join(obj_dataset_dir, corr[obj_name[:-3]], corr[obj_name[:-3]] + str(int(obj_name[-3:])).zfill(3), obj_name[:-3] + str(int(obj_name[-3:])).zfill(3) + "_m.obj")
    # if not isfile(obj_model_path):
    obj_model_path = join(obj_dataset_dir, obj_name[:-3].lower(), obj_name[:-3].lower() + str(int(obj_name[-3:])).zfill(3), obj_name[:-3].lower() + str(int(obj_name[-3:])).zfill(3) + "_m.obj")
    # if not isfile(obj_model_path):
    #     obj_model_path = join(obj_dataset_dir, corr[obj_name[:-3]], obj_name[:-3] + str(int(obj_name[-3:])).zfill(3), (obj_name[:-3] + str(int(obj_name[-3:])).zfill(3) + "_m.obj").capitalize())

    return obj_name, obj_model_path


if __name__=="__main__":
    for i in range(100):
        n, p = get_obj_info("/share/datasets/hhodataset/VTS/20231011/"+str(i).zfill(3),"/data3/datasets/HHO_object_dataset_final")
        print(n, p)
        assert(os.path.isfile(p))