import os
from os.path import join, dirname, abspath
import sys
from datetime import datetime
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from tools import rotvec_to_rotmat
from model.diffusion_hho import create_model_and_diffusion
from data.dataset_hho import Dataset, MODEL_PATH
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
import functools
from diffusion.resample import LossAwareSampler
from diffusion.resample import create_named_schedule_sampler
from render.mesh_viz import visualize_body_obj_hho
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))
from data_processing.smplx import smplx


class LitInteraction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")

        self.model, self.diffusion = create_model_and_diffusion(args)
        self.use_ddp = False
        self.ddp_model = self.model
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

    def on_train_start(self) -> None:
        #     backup trainer and model
        shutil.copy('./train_diffusion_hho.py', str(save_dir / 'train_diffusion_hho.py'))
        shutil.copy('./model/diffusion_hho.py', str(save_dir / 'diffusion_hho.py'))
        shutil.copy('./data/dataset_hho.py', str(save_dir / 'dataset_hho.py'))
        shutil.copy('./diffusion/gaussian_diffusion.py', str(save_dir / 'gaussian_diffusion.py'))
        return

    def l2(self, a, b):
        # assuming a.shape == b.shape == seqlen, bs, N
        loss = torch.nn.MSELoss(reduction='none')(a, b)
        loss = loss.mean(dim=[0, 2]) 
        return loss

    def forward_backward(self, batch, cond):
        t, weights = self.schedule_sampler.sample(batch.shape[0], device)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,  # [bs, ch, image_size, image_size]
            t,  # [bs](int) sampled timesteps
            model_kwargs=cond,
        )

        pred, gt = compute_losses()
        body1_pred, body2_pred, obj_pred = torch.split(pred.squeeze(1).permute(2, 0, 1).contiguous(), [self.args.smpl_dim+3, self.args.smpl_dim+3, 9], dim=2)
        body1_gt, body2_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), [self.args.smpl_dim+3, self.args.smpl_dim+3, 9], dim=2)
        
        loss_dict = dict()
        weighted_loss_dict = dict()

        T, B, nJ = body1_pred[:, :, :-3].shape
        nJ = nJ // 6
        body1_rot = body1_pred[:, :, :-3]
        body1_rot_gt = body1_gt[:, :, :-3]
        body2_rot = body2_pred[:, :, :-3]
        body2_rot_gt = body2_gt[:, :, :-3]
        obj_rot = obj_pred[:, :, :-3]
        obj_rot_gt = obj_gt[:, :, :-3]

        loss_body1_rot_past = self.l2(body1_rot[:self.args.past_len], body1_rot_gt[:self.args.past_len])
        loss_body1_nonrot_past = self.l2(body1_pred[:self.args.past_len, :, -3:], body1_gt[:self.args.past_len, :, -3:])
        loss_body2_rot_past = self.l2(body2_rot[:self.args.past_len], body2_rot_gt[:self.args.past_len])
        loss_body2_nonrot_past = self.l2(body2_pred[:self.args.past_len, :, -3:], body2_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_past = self.l2(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = self.l2(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_body1_rot_v_past = self.l2(body1_rot[1:self.args.past_len+1]-body1_rot[:self.args.past_len], body1_rot_gt[1:self.args.past_len+1]-body1_rot_gt[1:self.args.past_len+1]) +\
                               self.l2(body1_rot[1:self.args.past_len]-body1_rot[:self.args.past_len-1], body1_rot[2:self.args.past_len+1]-body1_rot[1:self.args.past_len])
        loss_body1_nonrot_v_past = self.l2(body1_pred[1:self.args.past_len+1, :, -3:]-body1_pred[:self.args.past_len, :, -3:], body1_gt[1:self.args.past_len+1, :, -3:]-body1_gt[1:self.args.past_len+1, :, -3:]) +\
                                  self.l2(body1_pred[1:self.args.past_len, :, -3:]-body1_pred[:self.args.past_len-1, :, -3:], body1_pred[2:self.args.past_len+1, :, -3:]-body1_pred[1:self.args.past_len, :, -3:])
        loss_body2_rot_v_past = self.l2(body2_rot[1:self.args.past_len+1]-body2_rot[:self.args.past_len], body2_rot_gt[1:self.args.past_len+1]-body2_rot_gt[1:self.args.past_len+1]) +\
                               self.l2(body2_rot[1:self.args.past_len]-body2_rot[:self.args.past_len-1], body2_rot[2:self.args.past_len+1]-body2_rot[1:self.args.past_len])
        loss_body2_nonrot_v_past = self.l2(body2_pred[1:self.args.past_len+1, :, -3:]-body2_pred[:self.args.past_len, :, -3:], body2_gt[1:self.args.past_len+1, :, -3:]-body2_gt[1:self.args.past_len+1, :, -3:]) +\
                                  self.l2(body2_pred[1:self.args.past_len, :, -3:]-body2_pred[:self.args.past_len-1, :, -3:], body2_pred[2:self.args.past_len+1, :, -3:]-body2_pred[1:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = self.l2(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[1:self.args.past_len+1]) +\
                              self.l2(obj_rot[1:self.args.past_len]-obj_rot[:self.args.past_len-1], obj_rot[2:self.args.past_len+1]-obj_rot[1:self.args.past_len])
        loss_obj_nonrot_v_past = self.l2(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[1:self.args.past_len+1, :, -3:]) +\
                                 self.l2(obj_pred[1:self.args.past_len, :, -3:]-obj_pred[:self.args.past_len-1, :, -3:], obj_pred[2:self.args.past_len+1, :, -3:]-obj_pred[1:self.args.past_len, :, -3:])

        loss_body1_rot_future = self.l2(body1_rot[self.args.past_len:], body1_rot_gt[self.args.past_len:])
        loss_body1_nonrot_future = self.l2(body1_pred[self.args.past_len:, :, -3:], body1_gt[self.args.past_len:, :, -3:])
        loss_body2_rot_future = self.l2(body2_rot[self.args.past_len:], body2_rot_gt[self.args.past_len:])
        loss_body2_nonrot_future = self.l2(body2_pred[self.args.past_len:, :, -3:], body2_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_future = self.l2(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = self.l2(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_body1_rot_v_future = self.l2(body1_rot[self.args.past_len:]-body1_rot[self.args.past_len-1:-1], body1_rot_gt[self.args.past_len:]-body1_rot_gt[self.args.past_len:]) +\
                                 self.l2(body1_rot[self.args.past_len-1:-2]-body1_rot[self.args.past_len:-1], body1_rot[self.args.past_len:-1]-body1_rot[self.args.past_len+1:])
        loss_body1_nonrot_v_future = self.l2(body1_pred[self.args.past_len:, :, -3:]-body1_pred[self.args.past_len-1:-1, :, -3:], body1_gt[self.args.past_len:, :, -3:]-body1_gt[self.args.past_len:, :, -3:]) +\
                                    self.l2(body1_pred[self.args.past_len-1:-2, :, -3:]-body1_pred[self.args.past_len:-1, :, -3:], body1_pred[self.args.past_len:-1, :, -3:]-body1_pred[self.args.past_len+1:, :, -3:])
        loss_body2_rot_v_future = self.l2(body2_rot[self.args.past_len:]-body2_rot[self.args.past_len-1:-1], body2_rot_gt[self.args.past_len:]-body2_rot_gt[self.args.past_len:]) +\
                                 self.l2(body2_rot[self.args.past_len-1:-2]-body2_rot[self.args.past_len:-1], body2_rot[self.args.past_len:-1]-body2_rot[self.args.past_len+1:])
        loss_body2_nonrot_v_future = self.l2(body2_pred[self.args.past_len:, :, -3:]-body2_pred[self.args.past_len-1:-1, :, -3:], body2_gt[self.args.past_len:, :, -3:]-body2_gt[self.args.past_len:, :, -3:]) +\
                                    self.l2(body2_pred[self.args.past_len-1:-2, :, -3:]-body2_pred[self.args.past_len:-1, :, -3:], body2_pred[self.args.past_len:-1, :, -3:]-body2_pred[self.args.past_len+1:, :, -3:])

        loss_obj_rot_v_future = self.l2(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len:]) +\
                                self.l2(obj_rot[self.args.past_len-1:-2]-obj_rot[self.args.past_len:-1], obj_rot[self.args.past_len:-1]-obj_rot[self.args.past_len+1:])
        loss_obj_nonrot_v_future = self.l2(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len:, :, -3:]) +\
                                   self.l2(obj_pred[self.args.past_len-1:-2, :, -3:]-obj_pred[self.args.past_len:-1, :, -3:], obj_pred[self.args.past_len:-1, :, -3:]-obj_pred[self.args.past_len+1:, :, -3:])

        loss_dict.update(dict(
                        body1_rot_past=loss_body1_rot_past,
                        body1_nonrot_past=loss_body1_nonrot_past,
                        body2_rot_past=loss_body2_rot_past,
                        body2_nonrot_past=loss_body2_nonrot_past,
                        obj_rot_past=loss_obj_rot_past,
                        obj_nonrot_past=loss_obj_nonrot_past,
                        body1_rot_v_past=loss_body1_rot_v_past,
                        body1_nonrot_v_past=loss_body1_nonrot_v_past,
                        body2_rot_v_past=loss_body2_rot_v_past,
                        body2_nonrot_v_past=loss_body2_nonrot_v_past,
                        obj_rot_v_past=loss_obj_rot_v_past,
                        obj_nonrot_v_past=loss_obj_nonrot_v_past,
                        body1_rot_future=loss_body1_rot_future,
                        body1_nonrot_future=loss_body1_nonrot_future,
                        body2_rot_future=loss_body2_rot_future,
                        body2_nonrot_future=loss_body2_nonrot_future,
                        obj_rot_future=loss_obj_rot_future,
                        obj_nonrot_future=loss_obj_nonrot_future,
                        body1_rot_v_future=loss_body1_rot_v_future,
                        body1_nonrot_v_future=loss_body1_nonrot_v_future,
                        body2_rot_v_future=loss_body2_rot_v_future,
                        body2_nonrot_v_future=loss_body2_nonrot_v_future,
                        obj_rot_v_future=loss_obj_rot_v_future,
                        obj_nonrot_v_future=loss_obj_nonrot_v_future,
                        ))

        weighted_loss_dict.update(dict(
                                body1_rot_past=loss_body1_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body1_nonrot_past=loss_body1_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body2_rot_past=loss_body2_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body2_nonrot_past=loss_body2_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                body1_rot_v_past=loss_body1_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body1_nonrot_v_past=loss_body1_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body2_rot_v_past=loss_body2_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body2_nonrot_v_past=loss_body2_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                body1_rot_future=loss_body1_rot_future * self.args.weight_smplx_rot,
                                body1_nonrot_future=loss_body1_nonrot_future * self.args.weight_smplx_nonrot,
                                body2_rot_future=loss_body2_rot_future * self.args.weight_smplx_rot,
                                body2_nonrot_future=loss_body2_nonrot_future * self.args.weight_smplx_nonrot,
                                obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                body1_rot_v_future=loss_body1_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body1_nonrot_v_future=loss_body1_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                body2_rot_v_future=loss_body2_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body2_nonrot_v_future=loss_body2_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                ))

        loss = sum(list(weighted_loss_dict.values()))

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, loss.detach()
            )

        loss = (loss * weights).mean()
        self.log_loss_dict(
            self.diffusion, t, weighted_loss_dict, loss
        )
        return loss, body1_pred, body2_pred, obj_pred, body1_gt, body2_gt, obj_gt

    def log_loss_dict(self, diffusion, ts, losses, loss):
        self.log('train_loss', loss, prog_bar=False)
        for key, values in losses.items():
            self.log(key, values.mean().item(), prog_bar=True)
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{key}_q{quartile}", sub_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        return ({'optimizer': optimizer,
                 })

    def calc_val_loss(self, body1_pred, body1_gt, body2_pred, body2_gt, obj_pred, obj_gt, batch):
        loss_dict = dict()
        weighted_loss_dict = dict()

        T, B, nJ = body1_pred[:, :, :-3].shape
        nJ = nJ // 3
        body1_rot = rotvec_to_rotmat(body1_pred[:, :, :-3]).view(T, B, nJ * 9)
        body1_rot_gt = rotvec_to_rotmat(body1_gt[:, :, :-3]).view(T, B, nJ * 9)
        body2_rot = rotvec_to_rotmat(body2_pred[:, :, :-3]).view(T, B, nJ * 9)
        body2_rot_gt = rotvec_to_rotmat(body2_gt[:, :, :-3]).view(T, B, nJ * 9)
        obj_rot = rotvec_to_rotmat(obj_pred[:, :, :-3]).view(T, B, 9)
        obj_rot_gt = rotvec_to_rotmat(obj_gt[:, :, :-3]).view(T, B, 9)

        loss_body1_rot_past = torch.nn.MSELoss(reduction='mean')(body1_rot[:self.args.past_len], body1_rot_gt[:self.args.past_len])
        loss_body1_nonrot_past = torch.nn.MSELoss(reduction='mean')(body1_pred[:self.args.past_len, :, -3:], body1_gt[:self.args.past_len, :, -3:])
        loss_body2_rot_past = torch.nn.MSELoss(reduction='mean')(body2_rot[:self.args.past_len], body2_rot_gt[:self.args.past_len])
        loss_body2_nonrot_past = torch.nn.MSELoss(reduction='mean')(body2_pred[:self.args.past_len, :, -3:], body2_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_body1_rot_v_past = torch.nn.MSELoss(reduction='mean')(body1_rot[1:self.args.past_len+1]-body1_rot[:self.args.past_len], body1_rot_gt[1:self.args.past_len+1]-body1_rot_gt[:self.args.past_len])
        loss_body1_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(body1_pred[1:self.args.past_len+1, :, -3:]-body1_pred[:self.args.past_len, :, -3:], body1_gt[1:self.args.past_len+1, :, -3:]-body1_gt[:self.args.past_len, :, -3:])
        loss_body2_rot_v_past = torch.nn.MSELoss(reduction='mean')(body2_rot[1:self.args.past_len+1]-body2_rot[:self.args.past_len], body2_rot_gt[1:self.args.past_len+1]-body2_rot_gt[:self.args.past_len])
        loss_body2_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(body2_pred[1:self.args.past_len+1, :, -3:]-body2_pred[:self.args.past_len, :, -3:], body2_gt[1:self.args.past_len+1, :, -3:]-body2_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[:self.args.past_len, :, -3:])

        loss_body1_rot_future = torch.nn.MSELoss(reduction='mean')(body1_rot[self.args.past_len:], body1_rot_gt[self.args.past_len:])
        loss_body1_nonrot_future = torch.nn.MSELoss(reduction='mean')(body1_pred[self.args.past_len:, :, -3:], body1_gt[self.args.past_len:, :, -3:])
        loss_body2_rot_future = torch.nn.MSELoss(reduction='mean')(body2_rot[self.args.past_len:], body2_rot_gt[self.args.past_len:])
        loss_body2_nonrot_future = torch.nn.MSELoss(reduction='mean')(body2_pred[self.args.past_len:, :, -3:], body2_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_body1_rot_v_future = torch.nn.MSELoss(reduction='mean')(body1_rot[self.args.past_len:]-body1_rot[self.args.past_len-1:-1], body1_rot_gt[self.args.past_len:]-body1_rot_gt[self.args.past_len-1:-1])
        loss_body1_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(body1_pred[self.args.past_len:, :, -3:]-body1_pred[self.args.past_len-1:-1, :, -3:], body1_gt[self.args.past_len:, :, -3:]-body1_gt[self.args.past_len-1:-1, :, -3:])
        loss_body2_rot_v_future = torch.nn.MSELoss(reduction='mean')(body2_rot[self.args.past_len:]-body2_rot[self.args.past_len-1:-1], body2_rot_gt[self.args.past_len:]-body2_rot_gt[self.args.past_len-1:-1])
        loss_body2_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(body2_pred[self.args.past_len:, :, -3:]-body2_pred[self.args.past_len-1:-1, :, -3:], body2_gt[self.args.past_len:, :, -3:]-body2_gt[self.args.past_len-1:-1, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len-1:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len-1:-1, :, -3:])

        loss_dict.update(dict(
                        body1_rot_past=loss_body1_rot_past,
                        body1_nonrot_past=loss_body1_nonrot_past,
                        body2_rot_past=loss_body2_rot_past,
                        body2_nonrot_past=loss_body2_nonrot_past,
                        obj_rot_past=loss_obj_rot_past,
                        obj_nonrot_past=loss_obj_nonrot_past,
                        body1_rot_v_past=loss_body1_rot_v_past,
                        body1_nonrot_v_past=loss_body1_nonrot_v_past,
                        body2_rot_v_past=loss_body2_rot_v_past,
                        body2_nonrot_v_past=loss_body2_nonrot_v_past,
                        obj_rot_v_past=loss_obj_rot_v_past,
                        obj_nonrot_v_past=loss_obj_nonrot_v_past,
                        body1_rot_future=loss_body1_rot_future,
                        body1_nonrot_future=loss_body1_nonrot_future,
                        body2_rot_future=loss_body2_rot_future,
                        body2_nonrot_future=loss_body2_nonrot_future,
                        obj_rot_future=loss_obj_rot_future,
                        obj_nonrot_future=loss_obj_nonrot_future,
                        body1_rot_v_future=loss_body1_rot_v_future,
                        body1_nonrot_v_future=loss_body1_nonrot_v_future,
                        body2_rot_v_future=loss_body2_rot_v_future,
                        body2_nonrot_v_future=loss_body2_nonrot_v_future,
                        obj_rot_v_future=loss_obj_rot_v_future,
                        obj_nonrot_v_future=loss_obj_nonrot_v_future,
                        ))

        weighted_loss_dict.update(dict(
                                body1_rot_past=loss_body1_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body1_nonrot_past=loss_body1_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body2_rot_past=loss_body2_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body2_nonrot_past=loss_body2_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                body1_rot_v_past=loss_body1_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body1_nonrot_v_past=loss_body1_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body2_rot_v_past=loss_body2_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body2_nonrot_v_past=loss_body2_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                body1_rot_future=loss_body1_rot_future * self.args.weight_smplx_rot,
                                body1_nonrot_future=loss_body1_nonrot_future * self.args.weight_smplx_nonrot,
                                body2_rot_future=loss_body2_rot_future * self.args.weight_smplx_rot,
                                body2_nonrot_future=loss_body2_nonrot_future * self.args.weight_smplx_nonrot,
                                obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                body1_rot_v_future=loss_body1_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body1_nonrot_v_future=loss_body1_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                body2_rot_v_future=loss_body2_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body2_nonrot_v_future=loss_body2_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def calc_loss(self, body1_pred, body1_gt, body2_pred, body2_gt, obj_pred, obj_gt, batch):
        loss_dict = dict()
        weighted_loss_dict = dict()

        T, B, nJ = body1_gt[:, :, :-3].shape
        nJ = nJ // 3
        body1_rot = rotvec_to_rotmat(body1_pred[:, :, :, :-3]).view(self.args.diverse_samples, T, B, nJ * 9)
        body1_rot_gt = rotvec_to_rotmat(body1_gt[:, :, :-3]).view(T, B, nJ * 9).unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        body2_rot = rotvec_to_rotmat(body2_pred[:, :, :, :-3]).view(self.args.diverse_samples, T, B, nJ * 9)
        body2_rot_gt = rotvec_to_rotmat(body2_gt[:, :, :-3]).view(T, B, nJ * 9).unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        obj_rot = rotvec_to_rotmat(obj_pred[:, :, :, :-3]).view(self.args.diverse_samples, T, B, 9)
        obj_rot_gt = rotvec_to_rotmat(obj_gt[:, :, :-3]).view(T, B, 9).unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        body_gt = body_gt.unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        obj_gt = obj_gt.unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        
        loss_body1_rot_past = torch.nn.MSELoss(reduction='mean')(body1_rot[:, :self.args.past_len], body1_rot_gt[:, :self.args.past_len])
        loss_body1_nonrot_past = torch.nn.MSELoss(reduction='mean')(body1_pred[:, :self.args.past_len, :, -3:], body1_gt[:, :self.args.past_len, :, -3:])
        loss_body2_rot_past = torch.nn.MSELoss(reduction='mean')(body2_rot[:, :self.args.past_len], body2_rot_gt[:, :self.args.past_len])
        loss_body2_nonrot_past = torch.nn.MSELoss(reduction='mean')(body2_pred[:, :self.args.past_len, :, -3:], body2_gt[:, :self.args.past_len, :, -3:])

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:, :self.args.past_len], obj_rot_gt[:, :self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, :self.args.past_len, :, -3:])

        loss_body1_rot_v_past = torch.nn.MSELoss(reduction='mean')(body1_rot[:, 1:self.args.past_len+1]-body1_rot[:, :self.args.past_len], body1_rot_gt[:, 1:self.args.past_len+1]-body1_rot_gt[:, :self.args.past_len])
        loss_body1_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(body1_pred[:, 1:self.args.past_len+1, :, -3:]-body1_pred[:, :self.args.past_len, :, -3:], body1_gt[:, 1:self.args.past_len+1, :, -3:]-body1_gt[:, :self.args.past_len, :, -3:])
        loss_body2_rot_v_past = torch.nn.MSELoss(reduction='mean')(body2_rot[:, 1:self.args.past_len+1]-body2_rot[:, :self.args.past_len], body2_rot_gt[:, 1:self.args.past_len+1]-body2_rot_gt[:, :self.args.past_len])
        loss_body2_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(body2_pred[:, 1:self.args.past_len+1, :, -3:]-body2_pred[:, :self.args.past_len, :, -3:], body2_gt[:, 1:self.args.past_len+1, :, -3:]-body2_gt[:, :self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:, 1:self.args.past_len+1]-obj_rot[:, :self.args.past_len], obj_rot_gt[:, 1:self.args.past_len+1]-obj_rot_gt[:, :self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:, 1:self.args.past_len+1, :, -3:]-obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, 1:self.args.past_len+1, :, -3:]-obj_gt[:, :self.args.past_len, :, -3:])

        loss_body1_rot_future = torch.nn.MSELoss(reduction='mean')(body1_rot[:, self.args.past_len:], body1_rot_gt[:, self.args.past_len:])
        loss_body1_nonrot_future = torch.nn.MSELoss(reduction='mean')(body1_pred[:, self.args.past_len:, :, -3:], body1_gt[:, self.args.past_len:, :, -3:])
        loss_body2_rot_future = torch.nn.MSELoss(reduction='mean')(body2_rot[:, self.args.past_len:], body2_rot_gt[:, self.args.past_len:])
        loss_body2_nonrot_future = torch.nn.MSELoss(reduction='mean')(body2_pred[:, self.args.past_len:, :, -3:], body2_gt[:, self.args.past_len:, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[:, self.args.past_len:], obj_rot_gt[:, self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[:, self.args.past_len:, :, -3:], obj_gt[:, self.args.past_len:, :, -3:])

        loss_body1_rot_v_future = torch.nn.MSELoss(reduction='mean')(body1_rot[:, self.args.past_len+1:]-body1_rot[:, self.args.past_len:-1], body1_rot_gt[:, self.args.past_len+1:]-body1_rot_gt[:, self.args.past_len:-1])
        loss_body1_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(body1_pred[:, self.args.past_len+1:, :, -3:]-body1_pred[:, self.args.past_len:-1, :, -3:], body1_gt[:, self.args.past_len+1:, :, -3:]-body1_gt[:, self.args.past_len:-1, :, -3:])
        loss_body2_rot_v_future = torch.nn.MSELoss(reduction='mean')(body2_rot[:, self.args.past_len+1:]-body2_rot[:, self.args.past_len:-1], body2_rot_gt[:, self.args.past_len+1:]-body2_rot_gt[:, self.args.past_len:-1])
        loss_body2_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(body2_pred[:, self.args.past_len+1:, :, -3:]-body2_pred[:, self.args.past_len:-1, :, -3:], body2_gt[:, self.args.past_len+1:, :, -3:]-body2_gt[:, self.args.past_len:-1, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[:, self.args.past_len+1:]-obj_rot[:, self.args.past_len:-1], obj_rot_gt[:, self.args.past_len+1:]-obj_rot_gt[:, self.args.past_len:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[:, self.args.past_len+1:, :, -3:]-obj_pred[:, self.args.past_len:-1, :, -3:], obj_gt[:, self.args.past_len+1:, :, -3:]-obj_gt[:, self.args.past_len:-1, :, -3:])

        loss_body1_rot_past_min = torch.nn.MSELoss(reduction='none')(body1_rot[:, :self.args.past_len], body1_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body1_nonrot_past_min = torch.nn.MSELoss(reduction='none')(body1_pred[:, :self.args.past_len, :, -3:], body1_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_rot_past_min = torch.nn.MSELoss(reduction='none')(body2_rot[:, :self.args.past_len], body2_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_nonrot_past_min = torch.nn.MSELoss(reduction='none')(body2_pred[:, :self.args.past_len, :, -3:], body2_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_past_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, :self.args.past_len], obj_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_past_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_body1_rot_v_past_min = torch.nn.MSELoss(reduction='none')(body1_rot[:, 1:self.args.past_len+1]-body1_rot[:, :self.args.past_len], body1_rot_gt[:, 1:self.args.past_len+1]-body1_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body1_nonrot_v_past_min = torch.nn.MSELoss(reduction='none')(body1_pred[:, 1:self.args.past_len+1, :, -3:]-body1_pred[:, :self.args.past_len, :, -3:], body1_gt[:, 1:self.args.past_len+1, :, -3:]-body1_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_rot_v_past_min = torch.nn.MSELoss(reduction='none')(body2_rot[:, 1:self.args.past_len+1]-body2_rot[:, :self.args.past_len], body2_rot_gt[:, 1:self.args.past_len+1]-body2_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_nonrot_v_past_min = torch.nn.MSELoss(reduction='none')(body2_pred[:, 1:self.args.past_len+1, :, -3:]-body2_pred[:, :self.args.past_len, :, -3:], body2_gt[:, 1:self.args.past_len+1, :, -3:]-body2_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_v_past_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, 1:self.args.past_len+1]-obj_rot[:, :self.args.past_len], obj_rot_gt[:, 1:self.args.past_len+1]-obj_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_v_past_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, 1:self.args.past_len+1, :, -3:]-obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, 1:self.args.past_len+1, :, -3:]-obj_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_body1_rot_future_min = torch.nn.MSELoss(reduction='none')(body1_rot[:, self.args.past_len:], body1_rot_gt[:, self.args.past_len:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body1_nonrot_future_min = torch.nn.MSELoss(reduction='none')(body1_pred[:, self.args.past_len:, :, -3:], body1_gt[:, self.args.past_len:, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_rot_future_min = torch.nn.MSELoss(reduction='none')(body2_rot[:, self.args.past_len:], body2_rot_gt[:, self.args.past_len:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_nonrot_future_min = torch.nn.MSELoss(reduction='none')(body2_pred[:, self.args.past_len:, :, -3:], body2_gt[:, self.args.past_len:, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_future_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, self.args.past_len:], obj_rot_gt[:, self.args.past_len:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_future_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, self.args.past_len:, :, -3:], obj_gt[:, self.args.past_len:, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_body1_rot_v_future_min = torch.nn.MSELoss(reduction='none')(body1_rot[:, self.args.past_len+1:]-body1_rot[:, self.args.past_len:-1], body1_rot_gt[:, self.args.past_len+1:]-body1_rot_gt[:, self.args.past_len:-1]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body1_nonrot_v_future_min = torch.nn.MSELoss(reduction='none')(body1_pred[:, self.args.past_len+1:, :, -3:]-body1_pred[:, self.args.past_len:-1, :, -3:], body1_gt[:, self.args.past_len+1:, :, -3:]-body1_gt[:, self.args.past_len:-1, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_rot_v_future_min = torch.nn.MSELoss(reduction='none')(body2_rot[:, self.args.past_len+1:]-body2_rot[:, self.args.past_len:-1], body2_rot_gt[:, self.args.past_len+1:]-body2_rot_gt[:, self.args.past_len:-1]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body2_nonrot_v_future_min = torch.nn.MSELoss(reduction='none')(body2_pred[:, self.args.past_len+1:, :, -3:]-body2_pred[:, self.args.past_len:-1, :, -3:], body2_gt[:, self.args.past_len+1:, :, -3:]-body2_gt[:, self.args.past_len:-1, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_v_future_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, self.args.past_len+1:]-obj_rot[:, self.args.past_len:-1], obj_rot_gt[:, self.args.past_len+1:]-obj_rot_gt[:, self.args.past_len:-1]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_v_future_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, self.args.past_len+1:, :, -3:]-obj_pred[:, self.args.past_len:-1, :, -3:], obj_gt[:, self.args.past_len+1:, :, -3:]-obj_gt[:, self.args.past_len:-1, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_dict.update(dict(
                        body1_rot_past=loss_body1_rot_past,
                        body1_nonrot_past=loss_body1_nonrot_past,
                        body2_rot_past=loss_body2_rot_past,
                        body2_nonrot_past=loss_body2_nonrot_past,
                        obj_rot_past=loss_obj_rot_past,
                        obj_nonrot_past=loss_obj_nonrot_past,
                        body1_rot_v_past=loss_body1_rot_v_past,
                        body1_nonrot_v_past=loss_body1_nonrot_v_past,
                        body2_rot_v_past=loss_body2_rot_v_past,
                        body2_nonrot_v_past=loss_body2_nonrot_v_past,
                        obj_rot_v_past=loss_obj_rot_v_past,
                        obj_nonrot_v_past=loss_obj_nonrot_v_past,
                        body1_rot_future=loss_body1_rot_future,
                        body1_nonrot_future=loss_body1_nonrot_future,
                        body2_rot_future=loss_body2_rot_future,
                        body2_nonrot_future=loss_body2_nonrot_future,
                        obj_rot_future=loss_obj_rot_future,
                        obj_nonrot_future=loss_obj_nonrot_future,
                        body1_rot_v_future=loss_body1_rot_v_future,
                        body1_nonrot_v_future=loss_body1_nonrot_v_future,
                        body2_rot_v_future=loss_body2_rot_v_future,
                        body2_nonrot_v_future=loss_body2_nonrot_v_future,
                        obj_rot_v_future=loss_obj_rot_v_future,
                        obj_nonrot_v_future=loss_obj_nonrot_v_future,
                        body1_rot_past_min=loss_body1_rot_past_min,
                        body1_nonrot_past_min=loss_body1_nonrot_past_min,
                        body2_rot_past_min=loss_body2_rot_past_min,
                        body2_nonrot_past_min=loss_body2_nonrot_past_min,
                        obj_rot_past_min=loss_obj_rot_past_min,
                        obj_nonrot_past_min=loss_obj_nonrot_past_min,
                        body1_rot_v_past_min=loss_body1_rot_v_past_min,
                        body1_nonrot_v_past_min=loss_body1_nonrot_v_past_min,
                        body2_rot_v_past_min=loss_body2_rot_v_past_min,
                        body2_nonrot_v_past_min=loss_body2_nonrot_v_past_min,
                        obj_rot_v_past_min=loss_obj_rot_v_past_min,
                        obj_nonrot_v_past_min=loss_obj_nonrot_v_past_min,
                        body1_rot_future_min=loss_body1_rot_future_min,
                        body1_nonrot_future_min=loss_body1_nonrot_future_min,
                        body2_rot_future_min=loss_body2_rot_future_min,
                        body2_nonrot_future_min=loss_body2_nonrot_future_min,
                        obj_rot_future_min=loss_obj_rot_future_min,
                        obj_nonrot_future_min=loss_obj_nonrot_future_min,
                        body1_rot_v_future_min=loss_body1_rot_v_future_min,
                        body1_nonrot_v_future_min=loss_body1_nonrot_v_future_min,
                        body2_rot_v_future_min=loss_body2_rot_v_future_min,
                        body2_nonrot_v_future_min=loss_body2_nonrot_v_future_min,
                        obj_rot_v_future_min=loss_obj_rot_v_future_min,
                        obj_nonrot_v_future_min=loss_obj_nonrot_v_future_min,
                        ))

        weighted_loss_dict.update(dict(
                                body1_rot_past=loss_body1_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body1_nonrot_past=loss_body1_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body2_rot_past=loss_body2_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body2_nonrot_past=loss_body2_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                body1_rot_v_past=loss_body1_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body1_nonrot_v_past=loss_body1_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body2_rot_v_past=loss_body2_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body2_nonrot_v_past=loss_body2_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                body1_rot_future=loss_body1_rot_future * self.args.weight_smplx_rot,
                                body1_nonrot_future=loss_body1_nonrot_future * self.args.weight_smplx_nonrot,
                                body2_rot_future=loss_body2_rot_future * self.args.weight_smplx_rot,
                                body2_nonrot_future=loss_body2_nonrot_future * self.args.weight_smplx_nonrot,
                                obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                body1_rot_v_future=loss_body1_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body1_nonrot_v_future=loss_body1_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                body2_rot_v_future=loss_body2_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body2_nonrot_v_future=loss_body2_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def _common_step(self, batch, batch_idx, mode):
        embedding, gt = self.model._get_embeddings(batch)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        if mode == 'train':
            loss, body1_pred, body2_pred, obj_pred, body1_gt, body2_gt, obj_gt = self.forward_backward(gt, model_kwargs)
            return loss
        elif mode == 'valid':
            model_kwargs['y']['inpainted_motion'] = gt
            model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                    device=device)  # True means use gt motion
            model_kwargs['y']['inpainting_mask'][:, :, :, self.args.past_len:] = False  # do inpainting in those frames
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(self.model, gt.shape, clip_denoised=False, model_kwargs=model_kwargs)
            body1_pred, body2_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), [self.args.smpl_dim+3, self.args.smpl_dim+3, 9], dim=2)
            body1_gt, body2_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), [self.args.smpl_dim+3, self.args.smpl_dim+3, 9], dim=2)
            T, B, _ = body1_pred[:, :, :-3].shape
            body1_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body1_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body1_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body1_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body2_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body2_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body2_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body2_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len
            # hand1_pose = torch.cat([frame['person1_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float()
            body1_pred = torch.cat([body1_rot, body1_pred[:, :, -3:]], dim=2)
            body1_gt = torch.cat([body1_rot_gt, body1_gt[:, :, -3:]], dim=2)
            # hand2_pose = torch.cat([frame['person2_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float()
            body2_pred = torch.cat([body2_rot, body2_pred[:, :, -3:]], dim=2)
            body2_gt = torch.cat([body2_rot_gt, body2_gt[:, :, -3:]], dim=2)
            obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)
            obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)
            loss, loss_dict, weighted_loss_dict = self.calc_val_loss(body1_pred, body1_gt, body2_pred, body2_gt, obj_pred, obj_gt, batch=batch)

            render_interval = 100
            if (batch_idx % render_interval == 0) and (((self.current_epoch % self.args.render_epoch) == self.args.render_epoch - 1) or self.args.debug):
                self.visualize(body1_pred, body2_pred, obj_pred, body1_gt, body2_gt, obj_gt, batch, batch_idx, mode, 0)
            return loss, loss_dict, weighted_loss_dict

        elif mode == 'test':
            model_kwargs['y']['inpainted_motion'] = gt
            model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                    device=device)  # True means use gt motion
            model_kwargs['y']['inpainting_mask'][:, :, :, self.args.past_len:] = False  # do inpainting in those frames
            sample_fn = self.diffusion.p_sample_loop
            body1_gt, body2_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), [self.args.smpl_dim+3, self.args.smpl_dim+3, 9], dim=2)
            idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len
            hand1_pose = torch.cat([frame['person1_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float()
            hand2_pose = torch.cat([frame['person2_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float()
            body1_preds, body2_preds, obj_preds = [], [], []
            T, B, _ = body1_gt[:, :, :-3].shape
            body1_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body1_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body2_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body2_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body1_gt = torch.cat([body1_rot_gt, body1_gt[:, :, -3:]], dim=2)  # TODO: check it
            body2_gt = torch.cat([body2_rot_gt, body2_gt[:, :, -3:]], dim=2)  # TODO: check it
            obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)
            for idx in range(self.args.diverse_samples):
                sample = sample_fn(self.model, gt.shape, clip_denoised=False, model_kwargs=model_kwargs)
                body1_pred, body2_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), [self.args.smpl_dim+3, self.args.smpl_dim+3, 9], dim=2)
                body1_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body1_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
                body2_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body2_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
                obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
                body1_pred = torch.cat([body1_rot, body1_pred[:, :, -3:]], dim=2)  # TODO: check it
                body2_pred = torch.cat([body2_rot, body2_pred[:, :, -3:]], dim=2)  # TODO: check it
                obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)
                body1_preds.append(body1_pred.unsqueeze(0))
                body2_preds.append(body2_pred.unsqueeze(0))
                obj_preds.append(obj_pred.unsqueeze(0))
            body1_preds = torch.cat(body1_preds, dim=0)
            body2_preds = torch.cat(body2_preds, dim=0)
            obj_preds = torch.cat(obj_preds, dim=0)
            loss, loss_dict, weighted_loss_dict = self.calc_loss(body1_preds, body1_gt, body2_preds, body2_gt, obj_preds, obj_gt, batch=batch)

            render_interval = 100
            if (batch_idx % render_interval == 0):
                for idx in range(self.args.diverse_samples):
                    if idx == 0:
                        self.visualize(body1_preds[idx], body2_preds[idx], obj_preds[idx], body1_gt, body2_gt, obj_gt, batch, batch_idx, mode, idx)
                    else:
                        self.visualize(body1_preds[idx], body2_preds[idx], obj_preds[idx], None, None, None, batch, batch_idx, mode, idx)
            return loss, loss_dict, weighted_loss_dict

    def visualize(self, body1_pred, body2_pred, obj_pred, body1_gt, body2_gt, obj_gt, batch, batch_idx, mode, idx):
        with torch.no_grad():
            body1 = body1_pred.detach().cpu().clone()
            body2 = body2_pred.detach().cpu().clone()
            obj = obj_pred.detach().cpu().clone()

            # visualize
            export_file = Path.joinpath(save_dir, 'render')
            export_file.mkdir(exist_ok=True, parents=True)
            rend_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_{}.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate, idx))
            
            T, B, _ = body1_pred.shape
            smplx_model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral", batch_size=T, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
            
            person1_pts = smplx_model(betas=torch.cat([record['person1_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body1[:, 0, 3:-3], global_orient=body1[:, 0, :3], transl=body1[:, 0, -3:])
            verts1, jtr1 = person1_pts.vertices, person1_pts.joints
            jtr1 = jtr1.detach().cpu().numpy()
            verts1 = verts1.detach().cpu().numpy()
            person2_pts = smplx_model(betas=torch.cat([record['person2_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body2[:, 0, 3:-3], global_orient=body2[:, 0, :3], transl=body2[:, 0, -3:])
            verts2, jtr2 = person2_pts.vertices, person2_pts.joints
            jtr2 = jtr2.detach().cpu().numpy()
            verts2 = verts2.detach().cpu().numpy()
            
            faces = smplx_model.faces_tensor.detach().cpu().numpy()
            obj_verts = []

            if (body1_gt is not None) and (body2_gt is not None) and (obj_gt is not None):
                body1_gt = body1_gt.detach().cpu().clone()
                body2_gt = body2_gt.detach().cpu().clone()
                obj_gt = obj_gt.detach().clone()
                rend_gt_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_gt.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))
                person1_gt_pts = smplx_model(betas=torch.cat([record['person1_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body1_gt[:, 0, 3:-3], global_orient=body1_gt[:, 0, :3], transl=body1_gt[:, 0, -3:])
                verts1_gt, jtr1_gt = person1_gt_pts.vertices, person1_gt_pts.joints
                jtr1_gt = jtr1_gt.detach().cpu().numpy()
                verts1_gt = verts1_gt.detach().cpu().numpy()
                person2_gt_pts = smplx_model(betas=torch.cat([record['person2_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body2_gt[:, 0, 3:-3], global_orient=body2_gt[:, 0, :3], transl=body2_gt[:, 0, -3:])
                verts2_gt, jtr2_gt = person2_gt_pts.vertices, person2_gt_pts.joints
                jtr2_gt = jtr2_gt.detach().cpu().numpy()
                verts2_gt = verts2_gt.detach().cpu().numpy()
                obj_verts_gt = []
                
            mesh_obj = Mesh()
            mesh_obj.load_from_file(batch['obj_model_path'][0])

            for t, record in (enumerate(batch['frames'])):
                mesh_obj_v = mesh_obj.v.copy()
                angle, trans = obj[t][0][:-3].detach().cpu().numpy(), obj[t][0][-3:].detach().cpu().numpy()
                rot = Rotation.from_rotvec(angle).as_matrix()
                # transform canonical mesh to fitting
                mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
                obj_verts.append(mesh_obj_v)

                if (body1_gt is not None) and (body2_gt is not None) and (obj_gt is not None):
                    mesh_obj_v = mesh_obj.v.copy()
                    angle, trans = obj_gt[t][0][:-3].detach().cpu().numpy(), obj_gt[t][0][-3:].detach().cpu().numpy()
                    rot = Rotation.from_rotvec(angle).as_matrix()
                    # transform canonical mesh to fitting
                    mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
                    obj_verts_gt.append(mesh_obj_v)
    
            m1 = visualize_body_obj_hho(verts1, verts2, faces, np.array(obj_verts), mesh_obj.f, past_len=self.args.past_len, save_path=rend_video_path, sample_rate=self.args.sample_rate)
            if (body1_gt is not None) and (body2_gt is not None) and (obj_gt is not None):
                m2 = visualize_body_obj_hho(verts1_gt, verts2_gt, faces, np.array(obj_verts_gt), mesh_obj.f, past_len=self.args.past_len, save_path=rend_gt_video_path, sample_rate=self.args.sample_rate)

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'test')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--model", type=str, default='Diffusion')
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--num_obj_keypoints", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=1)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_contact", type=float, default=0)
    parser.add_argument("--weight_dist", type=float, default=1)
    parser.add_argument("--weight_penetration", type=float, default=0)  #10

    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_obj_rot", type=float, default=0.1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_past", type=float, default=1)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=0.2)

    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=0)

    # dataset
    parser.add_argument("--past_len", type=int, default=15)
    parser.add_argument("--future_len", type=int, default=15)

    # train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)

    # diffusion
    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    parser.add_argument("--diverse_samples", type=int, default=10)
    
    # seed
    parser.add_argument("--seed", type=int, default=42)  # 233, 42, 0
    
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(args.seed, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    train_dataset = Dataset(mode = 'train', past_len=args.past_len, future_len=args.future_len)
    test_dataset = Dataset(mode = 'test', past_len=args.past_len, future_len=args.future_len)

    args.smpl_dim = 66 * 2
    args.num_obj_points = train_dataset.num_obj_points
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=False)  #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    if args.resume_checkpoint is not None:
        print('resume training')
        model = LitInteraction.load_from_checkpoint(args.resume_checkpoint, args=args)
    else:
        print('start training from scratch')
        model = LitInteraction(args)

    if args.mode == "train":
        # callback
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/interaction_hho'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                        mode="min")
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback, early_stop_callback],
                                                check_val_every_n_epoch=10,
                                                )
        trainer.fit(model, train_loader, val_loader)

    elif args.mode == "test" and args.resume_checkpoint is not None:
        # callback
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/sample_hho'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                        mode="min")
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback, early_stop_callback],
                                                )
        trainer.test(model, val_loader)
