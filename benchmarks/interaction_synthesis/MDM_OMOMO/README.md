The code is developed based on [Object Motion Guided Human Motion Synthesis](https://github.com/lijiaman/omomo_release), and is tested on Ubuntu 20.04 with NVIDIA Geforce RTX 3090.

## Environment Setup

> Note: This code was developed on Ubuntu 20.04 with Python 3.8, CUDA 11.3 and PyTorch 1.11.0.

Create a virtual environment using Conda and activate the environment. 
```
conda create -n omomo_env python=3.8
conda activate omomo_env 
```
Install PyTorch. 
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install PyTorch3D. 
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```
Install human_body_prior. 
```
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
```
Install BPS.
```
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
```
Install other dependencies. 
```
pip install -r requirements.txt 
```

## Baseline Methods

### [1] MDM

#### Training on CORE4D

```x
sh scripts/train_singlestage_hho.sh
```

#### Inference on CORE4D

```x
sh scripts/test_singlestage_hho.sh
```

#### Evaluation on CORE4D

```x
<set "results_dir" in "evaluation_hho.py" as the directory of your synthesized results>
python evaluation_hho.py
```

### [2] OMOMO

#### Training on CORE4D

```x
sh scripts/train_stage1_hho.sh

sh scripts/train_stage2_hho.sh
```

#### Inference on CORE4D

```x
sh scripts/test_omomo_hho.sh
```

#### Evaluation on CORE4D

```x
<set "results_dir" in "evaluation_hho.py" as the directory of your synthesized results>
python evaluation_hho.py
```

## Citation

If you find this code helpful, please cite the original project:

```
@article{li2023object,
  title={Object Motion Guided Human Motion Synthesis},
  author={Li, Jiaman and Wu, Jiajun and Liu, C Karen},
  journal={ACM Trans. Graph.},
  volume={42},
  number={6},
  year={2023}
}
```
