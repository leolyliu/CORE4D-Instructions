CUDA_VISIBLE_DEVICES=3 nohup python3 -u trainer_hand_foot_manip_diffusion.py --window=120 --batch_size=32 --project="/data2/datasets/omomo_runs" --exp_name="stage1_manip_set1" --wandb_pj_name="omomo_release_stage1" --entity="leoly" --data_root_folder="/data2/datasets/OMOMO_data" >train_stage1.out 2>&1 &