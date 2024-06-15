The code is developed based on [InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion](https://github.com/Sirui-Xu/InterDiff)

## Environment Setup

To create the environment, you can check and build according to the requirement file [requirements.txt](requirements.txt), which is based on Python 3.7. 

> [!NOTE]
> For specific packages such as [psbody-mesh](https://github.com/MPI-IS/mesh.git) and [human-body-prior](https://github.com/nghorbani/human_body_prior.git), you may need to build from their sources.

You may also build from a detailed requirement file based on Python 3.8, which might contain redundancies,
```
conda env create -f environment.yml
```

## Baseline Methods

### [1] MDM

#### Training on CORE4D

```x
cd interdiff
python train_diffusion_hho.py --seed <random_seed>
```

#### Evaluation on CORE4D

```x
cd interdiff
python eval_hho_short.py --resume_checkpoint <checkpoint_path> --resume_checkpoint_obj "" --mode no_correction
```

### [2] InterDiff

#### Training on CORE4D

```x
cd interdiff
python train_diffusion_hho.py --seed <random_seed>

cd interdiff
python train_correction_hho.py --seed <random_seed>
```

#### Evaluation on CORE4D

```x
cd interdiff
python eval_hho_short.py --resume_checkpoint <diffusion_checkpoint_path> --resume_checkpoint_obj <correction_checkpoint_path>
```

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, Hugging Face, Hydra, and uses datasets which each have their own respective licenses that must also be followed.

## Citation

If you find this code helpful, please cite the original project:

```bibtex
@inproceedings{
   xu2023interdiff,
   title={{InterDiff}: Generating 3D Human-Object Interactions with Physics-Informed Diffusion},
   author={Xu, Sirui and Li, Zhengyuan and Wang, Yu-Xiong and Gui, Liang-Yan},
   booktitle={ICCV},
   year={2023},
}
```
