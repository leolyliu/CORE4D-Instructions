import numpy as np
import json
import os
from os.path import isfile
import math
import argparse

import bpy

if __name__ == "__main__":
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--")+1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    print("scene name:{0}".format(scene_name))
   
    obj_folder = args.folder
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare ply paths 
    ori_obj_files = os.listdir(obj_folder)
    ori_obj_files.sort()
    obj_files = []
    for tmp_name in ori_obj_files:
        if "_person1.ply" in tmp_name:
            obj_files.append(tmp_name)

    for frame_idx in range(len(obj_files)):
        file_name = obj_files[frame_idx]

        # Iterate folder to process all model
        p1_path_to_file = os.path.join(obj_folder, file_name)
        p2_path_to_file = p1_path_to_file.replace("_person1.ply", "_person2.ply").replace("_person1.obj", "_person2.obj")
        object_path_to_file = p1_path_to_file.replace("_person1.ply", "_object.ply").replace("_person1.obj", "_object.obj")

        # Load human mesh and set material 
        if ".obj" in p1_path_to_file:
            p1_new_obj = bpy.ops.import_scene.obj(filepath=p1_path_to_file, split_mode ="OFF")
            p2_new_obj = bpy.ops.import_scene.obj(filepath=p2_path_to_file, split_mode ="OFF")
        elif ".ply" in p1_path_to_file:
            p1_new_obj = bpy.ops.import_mesh.ply(filepath=p1_path_to_file)
            p2_new_obj = bpy.ops.import_mesh.ply(filepath=p2_path_to_file)
        p1_obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", ""))]
        p2_obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", "").replace("_person1", "_person2"))]
        p1_mesh = p1_obj_object.data
        p2_mesh = p2_obj_object.data
        for f in p1_mesh.polygons:
            f.use_smooth = True
        for f in p2_mesh.polygons:
            f.use_smooth = True
        
        p1_obj_object.rotation_euler = (math.radians(90), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
        p2_obj_object.rotation_euler = (math.radians(90), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 

        p1_mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        p1_obj_object.data.materials.append(p1_mat)
        p1_mat.use_nodes = True
        principled_bsdf = p1_mat.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            principled_bsdf.inputs[0].default_value = (10/255.0, 30/255.0, 225/255.0, 1)  # person1: Light Blue, used for floor scene
        p1_obj_object.active_material = p1_mat
        
        p2_mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        p2_obj_object.data.materials.append(p2_mat)
        p2_mat.use_nodes = True
        principled_bsdf = p2_mat.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            principled_bsdf.inputs[0].default_value = (10/255.0, 225/255.0, 30/255.0, 1)  # person2: Light Green, used for floor scene
        p2_obj_object.active_material = p2_mat

        
        if isfile(object_path_to_file):
            # Load object mesh and set material 
            if ".obj" in object_path_to_file:
                new_obj = bpy.ops.import_scene.obj(filepath=object_path_to_file, split_mode ="OFF")
            elif ".ply" in object_path_to_file:
                new_obj = bpy.ops.import_mesh.ply(filepath=object_path_to_file)
            # obj_object = bpy.context.selected_objects[0]
            obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", "").replace("_person1", "_object"))]
            # obj_object.scale = (0.3, 0.3, 0.3)
            mesh = obj_object.data
            for f in mesh.polygons:
                f.use_smooth = True
            
            obj_object.rotation_euler = (math.radians(90), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 

            mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
            obj_object.data.materials.append(mat)
            mat.use_nodes = True
            principled_bsdf = mat.node_tree.nodes['Principled BSDF']
            if principled_bsdf is not None:
                # principled_bsdf.inputs[0].default_value = (220/255.0, 220/255.0, 220/255.0, 1) # Gray, close to white after rendering 
                # principled_bsdf.inputs[0].default_value = (10/255.0, 30/255.0, 225/255.0, 1) # Light Blue, used for floor scene 
                principled_bsdf.inputs[0].default_value = (153/255.0, 51/255.0, 255/255.0, 1) # Light Purple

            obj_object.active_material = mat

        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
        bpy.ops.render.render(write_still=True)

        # Delete materials
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        bpy.data.objects.remove(p1_obj_object, do_unlink=True)
        bpy.data.objects.remove(p2_obj_object, do_unlink=True)
        if isfile(object_path_to_file):
            bpy.data.objects.remove(obj_object, do_unlink=True)

    bpy.ops.wm.quit_blender()
