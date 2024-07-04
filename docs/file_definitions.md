# Definitions of CORE4D Data File

## Data Organization

The data is organized as follows:

```
|--CORE4D_Real
    |--object_models
        |-- <object category 1>
            |-- <object name 1>_m.obj
            |-- <object name 2>_m.obj
            ...
        |-- <object category 2>
            ...
        ...
    |--human_object_motions
        |-- <date 1>
            |-- <sequence 1>
                |-- smooth_objposes.npy
                |-- person1_poses.npz
                |-- person2_poses.npz
                |-- object_metadata.json
                |-- aligned_frame_ids.txt
            |-- <sequence 2>
                ...
            ...
        |-- <date 2>
            ...
        ...
    |--allocentric_RGBD_videos
        |-- <date 1>
            |-- <sequence 1>
                |-- <camera view 1>
                    |-- config.json
                    |-- intrinsic.json
                    |-- timestamp.txt
                    |-- color.mp4
                |-- <camera view 2>
                ...
            |-- <sequence 2>
                ...
            ...
        |-- <date 2>
            ...
        ...
    |--egocentric_RGB_videos
        |-- video
            |-- <video 1>.mp4
            |-- <video 2>.mp4
            ...
        |-- audio
            |-- <audio 1>.mp3 # provide timestamp
            |-- <audio 2>.mp3
            ...
            
    |--human_object_segmentations
        |-- <date 1>
            |-- <sequence 1>
                |-- <camera view 1>_mask.npz
                |-- <camera view 2>_mask.npz
                ...
            |-- <sequence 2>
                ...
            ...
        |-- <date 2>
            ...
        ...
    |--camera_parameters
        |-- <date 1>
            |-- <camera view 1>_intrinsic.json
            |-- <camera view 2>_intrinsic.json
            ...
            |-- <camera view 1>_extrinsic.txt
            |-- <camera view 2>_extrinsic.txt
            ...
        |-- <date 2>
            ...
        ...
    |--action_labels.json
|--CORE4D_Synthetic
    |-- <motion sequence 1>
        |-- human_poses.npy
        |-- object_mesh.obj
        |-- object_poses.npy
    |-- <motion sequence 2>
        ...
    ...
```

## Object Models

Each object model is represented as an ```obj``` file in the canonical space of the object category. The unit is meter.

## Human-object Motions

The object motion in the world coordinate system is defined as an ```numpy.float64``` array with shape (N_frame, 4, 4), where ```N_frame``` is the frame number. For each frame, the object pose is defined as a 4x4 transformation matrix of the object model.

The motion for each person in the world coordinate system is defined as an ```npz``` file. Please use the following command to parse the file:

```x
human_motion = np.load(<human pose file path>, allow_pickle=True)["arr_0"].item()
```

After that, the ```human_motion``` comprises the following information:

* ```betas```: An ```numpy.float32``` array with shape (N_frame, 10), denotes the human shape parameter in each frame. In fact, the shape parameters are exactly the same among all the frames.
* ```global_orient```: An ```numpy.float32``` array with shape (N_frame, 3), denotes the human root's orientation in each frame. The orientation is represented as axis-angle.
* ```transl```: An ```numpy.float32``` array with shape (N_frame, 3), denotes the human root's position in each frame.
* ```body_pose```: An ```numpy.float32``` array with shape (N_frame, 21, 3), denotes the human's body pose in each frame. The local orientation of each joint is represented as axis-angle.
* ```left_hand_pose```: An ```numpy.float32``` array with shape (N_frame, 12), denotes the human's left hand pose in each frame. The hand pose is defined in PCA space with 12 DoF.
* ```right_hand_pose```: An ```numpy.float32``` array with shape (N_frame, 12), denotes the human's right hand pose in each frame. The hand pose is defined in PCA space with 12 DoF.
* ```joints```: An ```numpy.float32``` array with shape (N_frame, 127, 3), denotes the human's SMPLX joint positions in each frame.
* ```vertices```: An ```numpy.float32``` array with shape (N_frame, 10475, 3), denotes the human's SMPLX vertex positions in each frame.

To feed ```betas```, ```global_orient```, ```transl```, ```body_pose```, ```left_hand_pose```, and ```right_hand_pose``` into the SMPLX model. Please first download [SMPLX models](https://smpl-x.is.tue.mpg.de/index.html), and then use the following python codes in the ```dataset_utils``` folder:

```x
from smplx import smplx

device = "cuda:0"

# load data
human_motion = np.load(<human pose file path>, allow_pickle=True)["arr_0"].item()
N_frame = human_motion["betas"].shape[0]

# create SMPLX model
smplx_model = smplx.create(<SMPLX model directory>, model_type="smplx", gender="neutral", batch_size=N_frame, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
smplx_model.to(device)

# prepare SMPLX parameters
SMPLX_params = {
    "betas": torch.from_numpy(human_motion["betas"]).to(device),
    "global_orient": torch.from_numpy(human_motion["global_orient"]).to(device),
    "transl": torch.from_numpy(human_motion["transl"]).to(device),
    "body_pose": torch.from_numpy(human_motion["body_pose"]).to(device),
    "left_hand_pose": torch.from_numpy(human_motion["left_hand_pose"]).to(device),
    "right_hand_pose": torch.from_numpy(human_motion["right_hand_pose"]).to(device),
}

# SMPLX forward
results = smplx_model(betas=SMPLX_params["betas"], body_pose=SMPLX_params["body_pose"], global_orient=SMPLX_params["global_orient"], transl=SMPLX_params["transl"], left_hand_pose=SMPLX_params["left_hand_pose"], right_hand_pose=SMPLX_params["right_hand_pose"])
print(results)

```
