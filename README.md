# :sparkles: CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement :sparkles:

Official repository of "CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement".

### :file_folder:[Dataset](https://1drv.ms/f/s!Ap-t7dLl7BFUmHl9Une1E6FLsS4J?e=RLt0Fk)

#### Authors

Chengwen Zhang*, Yun Liu*, Ruofan Xing, Bingda Tang, Li Yi

## Data Organization

The data is organized as follows:

```
|--CORE4D_Real
    |--object_models
    |--human_object_motions
    |--allocentric_RGBD_videos
    |--egocentric_RGB_videos
    |--camera_parameters
    |--action_labels.json
|--CORE4D_Synthetic
    |--object_models
    |--human_object_motions
```

## File Definitions

Please refer to ```docs/file_definitions.md``` for details of our dataset.

## Data Visualization

[1] Environment setup

Our code is tested on Ubuntu 20.04 with one NVIDIA GeForce RTX 3090 GPU. The Driver version is 535.129.03. The CUDA version is 12.2.

Please use the following command to set up the environment:

```x
conda create -n core4d python=3.9
conda activate core4d
<install PyTorch >= 1.7.1>
<install PyTorch3D >= 0.6.1>
cd dataset_utils
pip install -r requirements.txt
```

[2] Visualize human-object motions

```x
cd dataset_utils
python visualize_human_object_motion.py --dataset_root <dataset root directory> --object_model_root <object model root directory> --sequence_name <sequence name> --save_path <path to save the visualization result> --device <device for the rendering process>
```

For example, if you select the following data sequence:

```x
python visualize_human_object_motion.py --dataset_root <dataset root directory> --object_model_root <object model root directory> --sequence_name "20231002/004" --save_path "./example.gif" --device "cuda:0"
```

You can obtain the following visualization result:

<img src="https://raw.githubusercontent.com/leolyliu/CORE4D-Instructions/main/assets/example.gif" width="1920"/>

## Email

If you have any questions, please feel free to contact ```yun-liu22@mails.tsinghua.edu.cn```.
