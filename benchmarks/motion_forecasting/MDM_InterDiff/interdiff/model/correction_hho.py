import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from data.utils import marker2bodypart
from model.layers import ST_GCNN_layer

class ObjProjector(nn.Module):
    def __init__(self, args):
        super(ObjProjector, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.n_pre = args.dct
        self.st_gcnns_relative=nn.ModuleList()
        self.st_gcnns_relative.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns=nn.ModuleList()
        self.st_gcnns.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns_all=nn.ModuleList()
        self.st_gcnns_all.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               args.num_verts*2+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               args.num_verts*2+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               args.num_verts*2+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               args.num_verts*2+1,args.dropout,version=2))

        self.dct_m, self.idct_m = self.get_dct_matrix(args.past_len + args.future_len)

    def get_dct_matrix(self, N, is_torch=True):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        if is_torch:
            dct_m = torch.from_numpy(dct_m)
            idct_m = torch.from_numpy(idct_m)
        return dct_m, idct_m     

    def forward(self, data, initialize=False):
        obj_angles = torch.cat([frame['objfit_params']['angle'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
        obj_angles = matrix_to_rotation_6d(axis_angle_to_matrix(obj_angles))
        obj_trans = torch.cat([frame['objfit_params']['trans'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
        p1_verts = torch.cat([frame['p1_markers'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBxPx7
        p2_verts = torch.cat([frame['p2_markers'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBxPx7
        p1_contact = p1_verts[self.args.past_len:, :, :, -1].sum(dim=0) # B P
        p2_contact = p2_verts[self.args.past_len:, :, :, -1].sum(dim=0) # B P
        final_results = self.sample(obj_angles, obj_trans, p1_verts, p1_contact, p2_verts, p2_contact, initialize)
        obj_gt = torch.cat([obj_angles, obj_trans], dim=2)
        return final_results, obj_gt

    def sample(self, obj_angles, obj_trans, p1_verts, p1_contact, p2_verts, p2_contact, initialize=False):
        p1_verts = p1_verts[:, :, :, :3]
        p2_verts = p2_verts[:, :, :, :3]
        dct_m = self.dct_m.to(obj_angles.device).float()
        idct_m = self.idct_m.to(obj_angles.device).float()

        idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len

        # 在p1的67个vertex frame下的result, shape = (B, 9, T, 67)
        obj_trans_relative_to_p1 = obj_trans.unsqueeze(2) - p1_verts[:, :, :, :3]  # (T, B, 67, 3)
        obj_relative_to_p1 = torch.cat([obj_angles.unsqueeze(2).repeat(1, 1, obj_trans_relative_to_p1.shape[2], 1), obj_trans_relative_to_p1], dim=3)[idx_pad]  # (T, B, 67, 9)
        T, B, P, C = obj_relative_to_p1.shape
        obj_relative_to_p1 = obj_relative_to_p1.permute(1, 0, 3, 2).contiguous().view(B, T, C * P)
        obj_relative_to_p1 = torch.matmul(dct_m[:self.n_pre], obj_relative_to_p1).view(B, -1, C, P).permute(0, 2, 1, 3).contiguous() # B C T P
        x = obj_relative_to_p1.clone()
        for gcn in (self.st_gcnns_relative):
            x = gcn(x)
        obj_relative_to_p1 = obj_relative_to_p1 + x
        p1_trans = p1_verts[:, :, :, :3].permute(1, 0, 3, 2).contiguous().view(B, T, -1)
        p1_trans = torch.matmul(dct_m[:self.n_pre], p1_trans).view(B, -1, 3, P).permute(0, 2, 1, 3).contiguous() # B C T P
        obj_multi_p1 = torch.cat([obj_relative_to_p1[:, :6, :, :], obj_relative_to_p1[:, 6:9, :, :] + p1_trans], dim=1)
        
        # 在p2的67个vertex frame下的result, shape = (B, 9, T, 67)
        obj_trans_relative_to_p2 = obj_trans.unsqueeze(2) - p2_verts[:, :, :, :3]  # (T, B, 67, 3)
        obj_relative_to_p2 = torch.cat([obj_angles.unsqueeze(2).repeat(1, 1, obj_trans_relative_to_p2.shape[2], 1), obj_trans_relative_to_p2], dim=3)[idx_pad]  # (T, B, 67, 9)
        T, B, P, C = obj_relative_to_p2.shape
        obj_relative_to_p2 = obj_relative_to_p2.permute(1, 0, 3, 2).contiguous().view(B, T, C * P)
        obj_relative_to_p2 = torch.matmul(dct_m[:self.n_pre], obj_relative_to_p2).view(B, -1, C, P).permute(0, 2, 1, 3).contiguous() # B C T P
        x = obj_relative_to_p2.clone()
        for gcn in (self.st_gcnns_relative):
            x = gcn(x)
        obj_relative_to_p2 = obj_relative_to_p2 + x
        p2_trans = p2_verts[:, :, :, :3].permute(1, 0, 3, 2).contiguous().view(B, T, -1)
        p2_trans = torch.matmul(dct_m[:self.n_pre], p2_trans).view(B, -1, 3, P).permute(0, 2, 1, 3).contiguous() # B C T P
        obj_multi_p2 = torch.cat([obj_relative_to_p2[:, :6, :, :], obj_relative_to_p2[:, 6:9, :, :] + p2_trans], dim=1)

        # 在object自己的frame下的result, shape = (B, 9, T, 1)
        obj_gt = torch.cat([obj_angles, obj_trans], dim=2)
        obj = obj_gt[idx_pad].unsqueeze(2)  # (T, B, 1, 9)
        obj = obj.permute(1, 0, 3, 2).contiguous().view(B, T, C * 1)
        obj = torch.matmul(dct_m[:self.n_pre], obj).view(B, -1, C, 1).permute(0, 2, 1, 3).contiguous() # B C T P
        x = obj.clone()
        for gcn in (self.st_gcnns):
            x = gcn(x)
        obj = obj + x

        obj = torch.cat([obj, obj_multi_p1, obj_multi_p2], dim=3)  # (B, 9, T, 1+67*2)

        x = obj.clone()
        for gcn in (self.st_gcnns_all):
            x = gcn(x)

        obj = obj + x
        obj = obj.permute(0, 2, 1, 3).contiguous().view(B, -1, C * (1+P*2))
        results = torch.matmul(idct_m[:, :self.n_pre], obj).view(B, T, C, 1+P*2).permute(1, 0, 3, 2)[:, :, :, :9]  # (T, B, 1+67*2, 9)

        if initialize:
            final_results = results.mean(dim=2)
        else:
            final_results = torch.zeros((T, B, 9)).to(results.device)
            # 和p1、p2都不接触的motion clip: 使用object frame下的result (1+67*2中的那个1)
            final_results[:, (p1_contact.sum(dim=1) == 0) & (p2_contact.sum(dim=1) == 0)] = results[:, (p1_contact.sum(dim=1) == 0) & (p2_contact.sum(dim=1) == 0), 0, :]
            # 和p2接触的motion clip: 使用p2的某个vertex的frame下的result (1+67*2中的第二个67中的某一个), 如果和p1也接触则会在下面被p1的结果覆盖
            contact_happen_p2 = p2_contact[p2_contact.sum(dim=1) > 0].float()
            hand_marker = marker2bodypart["left_hand_ids"] + marker2bodypart["right_hand_ids"]
            contact_happen_p2[:, hand_marker] = contact_happen_p2[:, hand_marker] + 0.5  # (0~B, 67)
            results_contact_happen_p2 = results[:, p2_contact.sum(dim=1) > 0, 1+P:, :]  # (T, 0~B, 67, 9)
            if self.training:
                idx = torch.multinomial(contact_happen_p2, 1).unsqueeze(0).unsqueeze(3).repeat(T, 1, 1, 9)  # (T, 0~B, 1, 9)
            else:
                idx = torch.argmax(contact_happen_p2, dim=1, keepdim=True).unsqueeze(0).unsqueeze(3).repeat(T, 1, 1, 9)  # (T, 0~B, 1, 9)
            final_results[:, p2_contact.sum(dim=1) > 0] = torch.gather(results_contact_happen_p2, 2, idx).squeeze(2)
            # 和p1接触的motion clip: 使用p1的某个vertex的frame下的result (1+67*2中的第一个67中的某一个)  # TODO: 和p1、p2都接触时考虑在67*2里选
            contact_happen_p1 = p1_contact[p1_contact.sum(dim=1) > 0].float()
            hand_marker = marker2bodypart["left_hand_ids"] + marker2bodypart["right_hand_ids"]
            contact_happen_p1[:, hand_marker] = contact_happen_p1[:, hand_marker] + 0.5  # (0~B, 67)
            results_contact_happen_p1 = results[:, p1_contact.sum(dim=1) > 0, 1:1+P, :]  # (T, 0~B, 67, 9)
            if self.training:
                idx = torch.multinomial(contact_happen_p1, 1).unsqueeze(0).unsqueeze(3).repeat(T, 1, 1, 9)  # (T, 0~B, 1, 9)
            else:
                idx = torch.argmax(contact_happen_p1, dim=1, keepdim=True).unsqueeze(0).unsqueeze(3).repeat(T, 1, 1, 9)  # (T, 0~B, 1, 9)
            final_results[:, p1_contact.sum(dim=1) > 0] = torch.gather(results_contact_happen_p1, 2, idx).squeeze(2)

        return final_results