import os
from os.path import join, dirname, abspath
import sys
from datetime import datetime
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from tools import point2point_signed
from model.correction_hho import ObjProjector
from data.dataset_hho import Dataset, MODEL_PATH
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
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

        self.model = ObjProjector(args)
        self.model.to(device=device, dtype=torch.float)

    def on_train_start(self) -> None:
        #     backup trainer and model
        shutil.copy('./train_correction_hho.py', str(save_dir / 'train_correction_hho.py'))
        shutil.copy('./model/correction_hho.py', str(save_dir / 'correction_hho.py'))
        shutil.copy('./data/dataset_hho.py', str(save_dir / 'dataset_hho.py'))
        return

    def forward(self, x, initialize=False):
        return self.model(x, initialize)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 })
    
    def calc_loss(self, obj_pred, obj_gt, batch):

        loss_dict = dict()
        weighted_loss_dict = {}

        obj_rot = obj_pred[:, :, :-3]
        obj_rot_gt = obj_gt[:, :, :-3]

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len-1:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len-1:-1, :, -3:])

        loss_dict.update(dict(obj_rot_past=loss_obj_rot_past,
                              obj_nonrot_past=loss_obj_nonrot_past,
                              obj_rot_future=loss_obj_rot_future,
                              obj_nonrot_future=loss_obj_nonrot_future,
                              obj_rot_v_past=loss_obj_rot_v_past,
                              obj_nonrot_v_past=loss_obj_nonrot_v_past,
                              obj_rot_v_future=loss_obj_rot_v_future,
                              obj_nonrot_v_future=loss_obj_nonrot_v_future,
                             ))
        weighted_loss_dict.update(dict(obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                       obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                       obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                       obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                       obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                       obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                       obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                       obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                      ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def calc_loss_contact(self, obj_pred, obj_gt, batch):
        T, B, _ = obj_gt.shape

        obj_rot = obj_pred[:, :, :-3]
        obj_rot_gt = obj_gt[:, :, :-3]

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len-1:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len-1:-1, :, -3:])

        obj_rot = rotation_6d_to_matrix(obj_pred[:, :, :-3]).permute(0, 1, 3, 2)
        obj_rot_gt = rotation_6d_to_matrix(obj_gt[:, :, :-3]).permute(0, 1, 3, 2)


        obj_points = batch['obj_points'].float() # (T)xBxPx6
        obj_points_pred = torch.matmul(obj_points.unsqueeze(0)[:, :, :, :3], obj_rot) + obj_pred[:, :, -3:].unsqueeze(2)

        p1_verts = torch.cat([frame['p1_verts'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBxPx7
        p2_verts = torch.cat([frame['p2_verts'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBxPx7
        p1_verts, p1_verts_normals, p1_contact_label = p1_verts[:, :, :, :3], p1_verts[:, :, :, 3:6], p1_verts[:, :, :, 6:]
        p2_verts, p2_verts_normals, p2_contact_label = p2_verts[:, :, :, :3], p2_verts[:, :, :, 3:6], p2_verts[:, :, :, 6:]

        o2p1_signed, p12o_signed, o2p1_idx, p12o_idx, o2p1, p12o = point2point_signed(p1_verts.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=p1_verts_normals.view(T * B, -1, 3), return_vector=True)
        p12o = p12o.view(T, B, -1, 3)
        o2p2_signed, p22o_signed, o2p2_idx, p22o_idx, o2p2, p22o = point2point_signed(p2_verts.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=p2_verts_normals.view(T * B, -1, 3), return_vector=True)
        p22o = p22o.view(T, B, -1, 3)

        v_contact_p1 = torch.zeros([T * B, p12o_signed.size(1)]).to(p12o_signed.device)
        v_contact_p2 = torch.zeros([T * B, p22o_signed.size(1)]).to(p22o_signed.device)
        # v_collision_p1 = torch.zeros([T * B, p12o_signed.size(1)]).to(p12o_signed.device)
        # v_collision_p2 = torch.zeros([T * B, p22o_signed.size(1)]).to(p22o_signed.device)
        v_dist_p1 = (torch.abs(p12o_signed) > 0.02) * (p1_contact_label.view(T * B, -1) > 0.5)
        v_dist_p2 = (torch.abs(p22o_signed) > 0.02) * (p2_contact_label.view(T * B, -1) > 0.5)

        v_contact_p1[v_dist_p1] = 1
        v_contact_p2[v_dist_p2] = 1

        w_p1 = torch.zeros([T * B, o2p1_signed.size(1)]).to(self.device)
        w_dist_p1 = (o2p1_signed < 0.01) * (o2p1_signed > 0)
        w_dist_neg_p1 = o2p1_signed < 0
        w_p1[w_dist_p1] = 0 
        w_p1[w_dist_neg_p1] = 20
        w_p2 = torch.zeros([T * B, o2p2_signed.size(1)]).to(self.device)
        w_dist_p2 = (o2p2_signed < 0.01) * (o2p2_signed > 0)
        w_dist_neg_p2 = o2p2_signed < 0
        w_p1[w_dist_p2] = 0 
        w_p1[w_dist_neg_p2] = 20
       
        # f = torch.nn.ReLU()

        loss_contact_p1 = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(p12o_signed), v_contact_p1))
        loss_contact_p2 = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(p22o_signed), v_contact_p2))
        loss_dist_o_p1 = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2p1_signed), w_p1))
        loss_dist_o_p2 = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2p2_signed), w_p2))

        loss_contact = loss_contact_p1 + loss_contact_p2
    
        loss_penetration = loss_dist_o_p1 + loss_dist_o_p2

        loss_dict = dict(penetration=loss_penetration,
                         contact=loss_contact,
                         )
        annealing_factor = min(1.0, max(float(self.current_epoch) / (self.args.second_stage), 0)) if self.args.use_annealing else 1
        weighted_loss_dict = {
            'contact': max(annealing_factor ** 2, 0) * loss_contact * self.args.weight_contact,
            'penetration': max(annealing_factor ** 2, 0) * loss_penetration * self.args.weight_penetration,
        }

        loss_dict.update(dict(obj_rot_past=loss_obj_rot_past,
                              obj_nonrot_past=loss_obj_nonrot_past,
                              obj_rot_future=loss_obj_rot_future,
                              obj_nonrot_future=loss_obj_nonrot_future,
                              obj_rot_v_past=loss_obj_rot_v_past,
                              obj_nonrot_v_past=loss_obj_nonrot_v_past,
                              obj_rot_v_future=loss_obj_rot_v_future,
                              obj_nonrot_v_future=loss_obj_nonrot_v_future,
                             ))
        weighted_loss_dict.update(dict(obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                       obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                       obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                       obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                       obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                       obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                       obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                       obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                      ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def _common_step(self, batch, batch_idx, mode):
        obj_pred, obj_gt = self(batch, self.current_epoch < 10)
        loss, loss_dict, weighted_loss_dict = self.calc_loss_contact(obj_pred, obj_gt, batch=batch)

        render_interval = 50 if mode == 'valid' else 200
        if mode != 'train' and (batch_idx % render_interval == 0) and (((self.current_epoch+1) % self.args.render_epoch == 0) or self.args.debug):
            body1_pose = torch.cat([frame['person1_params']['pose'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBxDb
            body2_pose = torch.cat([frame['person2_params']['pose'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBxDb
            body1_trans = torch.cat([frame['person1_params']['trans'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBx3
            body2_trans = torch.cat([frame['person2_params']['trans'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBx3
            body1_gt = torch.cat([body1_pose[:, :, :3+21*3], body1_trans], dim=2)
            body2_gt = torch.cat([body2_pose[:, :, :3+21*3], body2_trans], dim=2)
            T, B, _ = body1_gt[:, :, :-3].shape
            obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, 6))).view(T, B, -1)
            obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, 6))).view(T, B, -1)
            obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)
            obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)

            with torch.no_grad():
                body1 = body1_gt.detach().cpu().clone()
                body2 = body2_gt.detach().cpu().clone()
                obj = obj_pred.detach().cpu().clone()
                body1_gt = body1_gt.detach().cpu().clone()
                body2_gt = body2_gt.detach().cpu().clone()
                obj_gt = obj_gt.detach().cpu().clone()
                
                # visualize
                export_file = Path.joinpath(save_dir, 'render')
                export_file.mkdir(exist_ok=True, parents=True)
                # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
                rend_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))
                rend_gt_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_gt.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))
                rend_p_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_p.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))

                T, B, _ = body1.shape
                smplx_model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral", batch_size=T, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
                
                person1_pts = smplx_model(betas=torch.cat([record['person1_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body1[:, 0, 3:-3], global_orient=body1[:, 0, :3], transl=body1[:, 0, -3:])
                verts1, jtr1 = person1_pts.vertices, person1_pts.joints
                jtr1 = jtr1.detach().cpu().numpy()
                verts1 = verts1.detach().cpu().numpy()
                person2_pts = smplx_model(betas=torch.cat([record['person2_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body2[:, 0, 3:-3], global_orient=body2[:, 0, :3], transl=body2[:, 0, -3:])
                verts2, jtr2 = person2_pts.vertices, person2_pts.joints
                jtr2 = jtr2.detach().cpu().numpy()
                verts2 = verts2.detach().cpu().numpy()
                person1_gt_pts = smplx_model(betas=torch.cat([record['person1_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body1_gt[:, 0, 3:-3], global_orient=body1_gt[:, 0, :3], transl=body1_gt[:, 0, -3:])
                verts1_gt, jtr1_gt = person1_gt_pts.vertices, person1_gt_pts.joints
                jtr1_gt = jtr1_gt.detach().cpu().numpy()
                verts1_gt = verts1_gt.detach().cpu().numpy()
                person2_gt_pts = smplx_model(betas=torch.cat([record['person2_params']['betas'][0:1] for record in batch['frames']], dim=0).detach().cpu(), body_pose=body2_gt[:, 0, 3:-3], global_orient=body2_gt[:, 0, :3], transl=body2_gt[:, 0, -3:])
                verts2_gt, jtr2_gt = person2_gt_pts.vertices, person2_gt_pts.joints
                jtr2_gt = jtr2_gt.detach().cpu().numpy()
                verts2_gt = verts2_gt.detach().cpu().numpy()
                
                faces = smplx_model.faces_tensor.detach().cpu().numpy()
                obj_verts = []
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

                    mesh_obj_v = mesh_obj.v.copy()
                    angle, trans = obj_gt[t][0][:-3].detach().cpu().numpy(), obj_gt[t][0][-3:].detach().cpu().numpy()
                    rot = Rotation.from_rotvec(angle).as_matrix()
                    # transform canonical mesh to fitting
                    mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
                    obj_verts_gt.append(mesh_obj_v)

                # m1 = visualize_body_obj_hho(np.concatenate((verts1_gt[:args.past_len], verts1[args.past_len:]), axis=0), np.concatenate((verts2_gt[:args.past_len], verts2[args.past_len:]), axis=0), faces, np.array(obj_verts_gt[:args.past_len] + obj_verts[args.past_len:]), mesh_obj.f, past_len=self.args.past_len, save_path=rend_video_path)
                # m2 = visualize_body_obj_hho(verts1_gt, verts2_gt, faces, np.array(obj_verts_gt), mesh_obj.f, past_len=self.args.past_len, save_path=rend_gt_video_path, sample_rate=self.args.sample_rate)
                # m3 = visualize_body_obj_hho(verts1, verts2, faces, np.array(obj_verts), mesh_obj.f, past_len=self.args.past_len, save_path=rend_p_video_path, sample_rate=self.args.sample_rate)

        return loss, loss_dict, weighted_loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'train')

        self.log('train_loss', loss, prog_bar=False)
        for key in loss_dict:
            self.log(key, loss_dict[key], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='ObjProjector')
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--num_obj_keypoints", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=1)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=256)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dct", type=int, default=10)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_contact", type=float, default=1)
    parser.add_argument("--weight_dist", type=float, default=0.1)
    parser.add_argument("--weight_penetration", type=float, default=0.1)  #10

    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.1)
    parser.add_argument("--weight_obj_rot", type=float, default=0.1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=0.1)
    parser.add_argument("--weight_past", type=float, default=0.5)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=1)
    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=1)

    # dataset
    parser.add_argument("--past_len", type=int, default=15)
    parser.add_argument("--future_len", type=int, default=15)

    # train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(42, workers=True)  # 233, 42, 0
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    train_dataset = Dataset(mode = 'train', past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate)
    test_dataset = Dataset(mode = 'test', past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate)

    args.smpl_dim = train_dataset.smpl_dim
    args.num_obj_points = train_dataset.num_obj_points
    args.num_verts = train_dataset.num_markers
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

    # callback
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/interaction_hho'), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  # for this version
    print(save_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                       monitor="val_loss",
                                                       save_weights_only=True, save_last=True, save_top_k=-1, every_n_epochs=10)
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



