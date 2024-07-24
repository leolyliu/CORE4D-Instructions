import os
from os.path import join, dirname
import sys
sys.path.append("..")
import numpy as np
import pickle
import torch
import trimesh
from smplx import smplx
from smplx.smplx.lbs import batch_rodrigues
import cv2
import imageio
from utils.txt2intrinsic import txt2intrinsic
from utils.pyt3d_wrapper import Pyt3DWrapper
from utils.avi2depth import avi2depth
from utils.time_align import time_align
from utils.process_timestamps import txt_to_paried_frameids, paired_frameids_to_txt
from utils.contact import compute_contact
import open3d as o3d


def save_pcd(save_path, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)


def save_mesh(save_path, tri_mesh):
    mesh_txt = trimesh.exchange.obj.export_obj(tri_mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    os.makedirs(dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        fp.write(mesh_txt)


def draw_head_orientation(img, head_pos, head_rot, camera_intrinsic, camera_extrinsic):
    p = np.float32([  # coordinate frame, (4, 3)
        [0, 0, 0],
        [0.2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.2],
    ])
    p = (p @ head_rot.T) + head_pos.reshape(3)  # (4, 3), in world space
    p = np.concatenate((p, np.ones((p.shape[0], 1))), axis=-1)  # (4, 4), in world space
    p = p @ camera_extrinsic.transpose(1, 0)  # (4, 4), in camera space
    p = p[:, :3]  # (4, 3), in camera space
    uv = p @ camera_intrinsic.transpose(1, 0)
    uv = (uv[:, :2] / uv[:, 2:]).astype(np.int32)  # (4, 2), in image space
    
    # draw lines
    cv2.line(img, tuple(uv[0]), tuple(uv[1]), (0, 0, 255), 3)
    cv2.line(img, tuple(uv[0]), tuple(uv[2]), (0, 255, 0), 3)
    cv2.line(img, tuple(uv[0]), tuple(uv[3]), (255, 0, 0), 3)
    return img


def read_data_from_SMPLX(smplx_path):
    rawdata = np.load(smplx_path, allow_pickle=True)
    data = {}
    for key, val in rawdata.items():
        data[key] = val
    return data


def render_SMPLX_TPOSE(pyt3d_wrapper, betas=np.zeros(10)):

    # SMPLX构建时会额外构建face, 而把hand和body的构建交给其继承的SMPL+H来做
    # use_pca: True: 每个手的theta是num_pca_comps DoF, 决定各个PCA分量的权重; False: 每个手的theta是3*15 DoF的
    # flat_hand_mean: True: 把hand pose设为0, False: 把hand pose设为MANO默认值
    smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
    smplx_model.to("cuda:0")

    # get mesh faces
    model_data = read_data_from_SMPLX("/share/human_model/models/smplx/SMPLX_NEUTRAL.npz")
    faces = model_data["f"]
    print(faces.shape, faces.min(), faces.max())

    # prepare betas
    betas = torch.from_numpy(betas).unsqueeze(0).to(torch.float32).to("cuda:0")

    # init values
    expression = torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to("cuda:0")
    global_orient = torch.zeros([1, 3], dtype=torch.float32).to("cuda:0")
    transl = torch.zeros([1, 3], dtype=torch.float32).to("cuda:0")
    body_pose = torch.zeros([1, smplx_model.NUM_BODY_JOINTS, 3]).to("cuda:0")

    result_model = smplx_model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, return_verts=True)
    result_vertices = result_model.vertices.detach().cpu().numpy()[0]
    result_joints = result_model.joints.detach().cpu().numpy()[0]
    print(result_vertices.shape)

    # output SMPLX mesh
    mesh = trimesh.Trimesh(vertices=result_vertices, faces=faces)
    save_mesh("./SMPLX_TPOSE.obj", mesh)
    
    # output SMPLX joint pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result_joints)
    o3d.io.write_point_cloud("./smplx_TPOSE.ply", pcd)

    # pytorch3d rendering
    render_meshes = [mesh]
    render_result = pyt3d_wrapper.render_meshes(render_meshes)
    imageio.mimsave('./smplx_T_POSE.gif', [(img*255).astype(np.uint8) for img in render_result])


def overlay(img, rgb_img):
    overlay = rgb_img.copy()
    mask_idx = np.where(img.sum(axis=-1) < 255*3)
    overlay[mask_idx[0], mask_idx[1]] = ((rgb_img[mask_idx[0], mask_idx[1]] * 0.15) + (img[mask_idx[0], mask_idx[1]] * 0.85)).astype(np.uint8)

    return overlay


def render_SMPLX(pyt3d_wrapper, smplx_model, betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose, rgb_img=None, frame_idx=None, suffix="", save=True):
    """
    betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose: torch.float, shape = (1, ...)
    rgb_img: 如果不是None, 则把SMPLX模型贴到原图上输出 (仅此时需用camera_intrin)
    """

    # init values
    expression = torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to("cuda:0")

    result_model = smplx_model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
    result_vertices = result_model.vertices.detach().cpu().numpy()[0]
    result_joints = result_model.joints.detach().cpu().numpy()[0]
    faces = result_model.faces.detach().cpu().numpy()
    # print(faces.shape)

    mesh = trimesh.Trimesh(vertices=result_vertices, faces=faces)

    # # output SMPLX mesh
    # mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    # fn = "./vis_smplx" + suffix + ".obj"
    # if not frame_idx is None:
    #     fn = "./vis_smplx_frame_" + str(frame_idx) + suffix + ".obj"
    # with open(fn, "w") as fp:
    #     fp.write(mesh_txt)
    
    # # output SMPLX joint pcd
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(result_joints)
    # fn = "./vis_smplx" + suffix + ".ply"
    # if not frame_idx is None:
    #     fn = "./vis_smplx_frame_" + str(frame_idx) + suffix + ".ply"
    # o3d.io.write_point_cloud(fn, pcd)

    # pytorch3d rendering
    render_meshes = [mesh]
    render_result = pyt3d_wrapper.render_meshes(render_meshes)
    # imageio.mimsave('./vis_smplx.gif', [(img*255).astype(np.uint8) for img in render_result])
    img = (render_result[0]*255).astype(np.uint8)

    # # 从pytorch3d的相机系转到opencv的相机系, 需使用intrin的光心
    # M = np.float32([
    #     [-1, 0, 2 * camera_intrin[0, 2]],
    #     [0, -1, 2 * camera_intrin[1, 2]],
    # ])
    # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    if not rgb_img is None:
        img = overlay(img, rgb_img)
        
        # result_joints = result_joints @ camera_extrin[:3, :3].T + camera_extrin[:3, 3].reshape(1, 3)
        # result_joints = result_joints @ camera_intrin.T
        # result_joints = result_joints[:, :2] / result_joints[:, 2:]
        # for p in result_joints:
        #     cv2.circle(img, p.astype(np.int32), 5, (255, 0, 0), -1)
    
    if save:
        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./vis_smplx.png', img1)
    
    return img


def render_HHO(pyt3d_wrapper, smplx_model, data, rgb_img=None, frame_idx=None, suffix="", save=True, device="cuda:0", render_contact=False):
    """
    data = {
        "person1": {"betas", "body_pose", "transl", "global_orient", "left_hand_pose", "right_hand_pose"} / None
        "person2": {"betas", "body_pose", "transl", "global_orient", "left_hand_pose", "right_hand_pose"} / None
        "object": {"mesh", "obj2world"} / None
    }
    render_contact = True: return the original img and contact imgs
    """
    if render_contact:
        if (data["person1"] is None) or (data["person2"] is None) or (data["object"] is None):
            raise NotImplementedError
    
    render_meshes = []
    
    # init values
    expression = torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to(device)

    if ("person1" in data) and (not data["person1"] is None):
        betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose = data["person1"]["betas"], data["person1"]["body_pose"], data["person1"]["transl"], data["person1"]["global_orient"], data["person1"]["left_hand_pose"], data["person1"]["right_hand_pose"]
        result_model = smplx_model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
        result_vertices = result_model.vertices.detach().cpu().numpy()[0]
        result_joints = result_model.joints.detach().cpu().numpy()[0]
        faces = result_model.faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=result_vertices, faces=faces)
        render_meshes.append(mesh)
    
    if ("person2" in data) and (not data["person2"] is None):
        betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose = data["person2"]["betas"], data["person2"]["body_pose"], data["person2"]["transl"], data["person2"]["global_orient"], data["person2"]["left_hand_pose"], data["person2"]["right_hand_pose"]
        result_model = smplx_model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
        result_vertices = result_model.vertices.detach().cpu().numpy()[0]
        result_joints = result_model.joints.detach().cpu().numpy()[0]
        faces = result_model.faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=result_vertices, faces=faces)
        render_meshes.append(mesh)

    
    if ("object" in data) and (not data["object"] is None):
        ori_mesh, obj2world = data["object"]["mesh"], data["object"]["obj2world"]
        vertices, faces = ori_mesh.vertices, ori_mesh.faces
        vertices = vertices @ obj2world[:3, :3].T + obj2world[:3, 3]  # object coord -> world coord
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        render_meshes.append(mesh)
        
    
    
    # pytorch3d render
    render_result = pyt3d_wrapper.render_meshes(render_meshes)
    img = (render_result[0]*255).astype(np.uint8)
    
    if not rgb_img is None:
        img = overlay(img, rgb_img)
    
    if save:
        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./vis_smplx.png', img1)
    
    if not render_contact:
        return img
    else:
        person1_v, person2_v, obj_v = render_meshes[0].vertices, render_meshes[1].vertices, render_meshes[2].vertices
        person1_contact = compute_contact(person1_v, np.concatenate((person2_v, obj_v), axis=0), threshould=0.03, device=device)
        person2_contact = compute_contact(person2_v, np.concatenate((person1_v, obj_v), axis=0), threshould=0.03, device=device)
        obj_contact = compute_contact(obj_v, np.concatenate((person1_v, person2_v), axis=0), threshould=0.03, device=device)
        
        person_contact_pyt3d_wrapper = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[-0.1, -2.0, -0.1]], device=device)
        obj_contact_pyt3d_wrapper = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[2.0, 0.5, 0.0]], device=device)
        
        # human TPOSE
        global_orient = torch.zeros([1, 3], dtype=torch.float32).to(device)
        transl = torch.zeros([1, 3], dtype=torch.float32).to(device)
        body_pose = torch.zeros([1, smplx_model.NUM_BODY_JOINTS, 3]).to(device)
        SMPLX_TPOSE1 = smplx_model(betas=data["person1"]["betas"], expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, return_verts=True)
        SMPLX_TPOSE2 = smplx_model(betas=data["person2"]["betas"], expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, return_verts=True)
        SMPLX_faces = SMPLX_TPOSE1.faces.detach().cpu().numpy()
        
        specified_vertices = [torch.where(person1_contact)[0].detach().cpu().numpy()]
        meshes = [trimesh.Trimesh(vertices=SMPLX_TPOSE1.vertices.detach().cpu().numpy()[0], faces=SMPLX_faces)]
        render_results = person_contact_pyt3d_wrapper.render_meshes(meshes, specified_vertices=specified_vertices)
        person1_contact_img = (render_results[0]*255).astype(np.uint8)
        
        specified_vertices = [torch.where(person2_contact)[0].detach().cpu().numpy()]
        meshes = [trimesh.Trimesh(vertices=SMPLX_TPOSE2.vertices.detach().cpu().numpy()[0], faces=SMPLX_faces)]
        render_results = person_contact_pyt3d_wrapper.render_meshes(meshes, specified_vertices=specified_vertices)
        person2_contact_img = (render_results[0]*255).astype(np.uint8)
        
        specified_vertices = [torch.where(obj_contact)[0].detach().cpu().numpy()]
        meshes = [data["object"]["mesh"]]
        render_results = obj_contact_pyt3d_wrapper.render_meshes(meshes, specified_vertices=specified_vertices)
        obj_contact_img = (render_results[0]*255).astype(np.uint8)
          
        return img, person1_contact_img, person2_contact_img, obj_contact_img


def get_human_color_table(N):
    # RGB
    color_table = [
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
    ]

    return color_table[:N]


def get_object_color_table(N):
    # RGB
    color_table = [
        [255, 255, 0],  # yellow
        [0, 255, 255],
        [255, 0, 255],
    ]

    return color_table[:N]


def add_human_skeleton_on_image(img, human_pixel, color):
    color = tuple(color)
    for p in human_pixel:
        # p = (p[0] - 20, p[1] + 20)  # !!!
        cv2.circle(img, p, 5, color, -1)


def add_object_pcd_on_image(img, object_pixel, color):
    color = tuple(color)
    for p in object_pixel:
        # p = (p[0] - 20, p[1] + 20)  # !!!
        cv2.circle(img, p, 3, color, 2)


def render_mocap_data_on_camera_space(joint_datas, objposes, objmodels, camera_pose, camera_intrinsic, rgb_data=None, start_mocap_idx=0, img_size=(1280, 720), model_sample_point_cnt=1000, save_path=None, fps=90):
    """
    可视化一段采集视频中的所有mocap数据

    joint_datas: 每个人的joint在世界系下的3D坐标
    objposes: 每个物体在世界系下的pose
    objmodels: 每个物体的mesh (open3d.geometry.TriangleMesh), 单位: m
    camera_pose: 相机的pose
    camera_intrinsic: 相机内参
    img_size: 相机RGB图片的分辨率
    model_sample_point_cnt: 从每个object mesh采一些vertex, 只可视化这些vertex而不去用renderer渲染mesh
    save_path: mp4保存的路径, 如果为None则不保存
    fps: 保存的mp4的帧率
    """

    if not joint_datas is None:
        N_person = len(joint_datas)  # 人数
    else:
        N_person = 0
    N_obj = len(objmodels)  # 物体数
    if not objposes is None:
        assert len(objposes) == N_obj
    N = camera_pose.shape[0]  # 帧数
    if not joint_datas is None:
        for jd in joint_datas:
            assert jd.shape[0] == N
    if not objposes is None:
        for op in objposes:
            assert op.shape[0] == N
    
    # 预处理每个object mesh的采样点云
    obj_pts = []
    for om in objmodels:
        pcd = om.sample_points_poisson_disk(model_sample_point_cnt)
        obj_pts.append(np.float32(pcd.points))

    # 染色方案
    human_color_table = get_human_color_table(N_person)
    obj_color_table = get_object_color_table(N_obj)

    print("[render_mocap_data_on_camera_space] converting points to the image space...")
    
    # 把所有点转到图像系下
    camera_extrinsic = np.linalg.inv(camera_pose)  # shape = (N, 4, 4)
    joint_pixels = []
    if not joint_datas is None:
        for jd in joint_datas:
            p = np.concatenate((jd, np.ones((N, jd.shape[1], 1))), axis=-1)  # (N, 74, 4), in world space
            p = p @ camera_extrinsic.transpose(0, 2, 1)
            p = p[:, :, :3]  # (N, 74, 3), in camera space
            uv = p @ camera_intrinsic.transpose(1, 0)
            uv = uv[:, :, :2] / uv[:, :, 2:]  # (N, 74, 2), in image space
            joint_pixels.append(uv.astype(np.int32))
    obj_pixels = []
    if not objposes is None:
        for (o_pose, o_pts) in zip(objposes, obj_pts):
            p = np.concatenate((o_pts[None, :].repeat(N, axis=0), np.ones((N, model_sample_point_cnt, 1))), axis=-1)  # (N, 100, 4), in object space
            p = p @ o_pose.transpose(0, 2, 1)  # (N, 100, 4), in world space
            p = p @ camera_extrinsic.transpose(0, 2, 1)
            p = p[:, :, :3]  # (N, 100, 3), in camera space
            uv = p @ camera_intrinsic.transpose(1, 0)
            uv = uv[:, :, :2] / uv[:, :, 2:]  # (N, 100, 2), in image space
            obj_pixels.append(uv.astype(np.int32))
    else:
        pass
        # pose = I
        # for o_pts in obj_pts:
        #     p = np.concatenate((o_pts[None, :].repeat(N, axis=0), np.ones((N, model_sample_point_cnt, 1))), axis=-1)  # (N, 100, 4), in object space
        #     pass  # object space = world space
        #     p = p @ camera_extrinsic.transpose(0, 2, 1)
        #     p = p[:, :, :3]  # (N, 100, 3), in camera space
        #     uv = p @ camera_intrinsic.transpose(1, 0)
        #     uv = uv[:, :, :2] / uv[:, :, 2:]  # (N, 100, 2), in image space
        #     obj_pixels.append(uv.astype(np.int32))

    if rgb_data is None:  # 无需和rgb数据对齐, 只需单独渲染mocap数据即可
        print("[render_mocap_data_on_camera_space] start saving...")
        # render
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(save_path, fourcc, fps, img_size)
        W, H = img_size[0], img_size[1]
        for idx in range(N):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            for (i, jp) in enumerate(joint_pixels):
                add_human_skeleton_on_image(img, jp[idx], human_color_table[i])
            for (i, op) in enumerate(obj_pixels):
                add_object_pcd_on_image(img, op[idx], obj_color_table[i])
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            videoWriter.write(img)
        videoWriter.release()
    
    # 需要把mocap数据和rgb数据对齐
    mocap_fps = 90
    start_tstamp = rgb_data[0][0]  # start_tstamp对应mocap数据的第start_mocap_idx帧
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (int(img_size[0] / 2), int(img_size[1] / 2)))
    cnt = -1
    for (tstamp, img) in rgb_data:
        idx = start_mocap_idx + int((tstamp - start_tstamp) * mocap_fps)
        if idx >= N:
            break
        cnt += 1
        img_cp = img.copy()
        for (i, jp) in enumerate(joint_pixels):
            add_human_skeleton_on_image(img, jp[idx], human_color_table[i])
        for (i, op) in enumerate(obj_pixels):
            add_object_pcd_on_image(img, op[idx], obj_color_table[i])
        
        diff = np.sum((img-img_cp)**2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (int(img_size[0] / 2), int(img_size[1] / 2)))
        videoWriter.write(img)
    videoWriter.release()


def read_a_image_from_video(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    suc = cap.isOpened()
    idx = -1
    img = None
    while True:
        idx += 1
        suc, img = cap.read()
        if not suc:
            break
        if idx == frame_idx:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            break
    cap.release()
    return img


def video2imgs(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    suc = cap.isOpened()
    imgs = []
    while True:
        suc, img = cap.read()
        if not suc:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    cap.release()
    return imgs


def render_SMPLX_sequence(data_dir, camera_name, camera_intrin_path, camera_pose_path, frame_range, M, save_path, cfg, object_data=None, paired_frames=None, render_contact=False, visualize_head_orientation=False, video_writer=None, device="cuda:0"):
    """
    frame_range: [L, R], close range
    """
    if frame_range[0] is None:
        frame_range[0] = 0
    if frame_range[1] is None:
        assert not paired_frames is None
        frame_range[1] = len(paired_frames) - 1
    
    if not (cfg["render_person1"] and cfg["render_person2"] and cfg["render_object"]):
        print("[error] cannot render contact, change render_contact to False !!!")
        render_contact = False
    
    rgb_video_path = join(data_dir, "_{}_camera{}_color_image_raw.mp4".format(camera_name.split("_")[0], camera_name.split("_")[1]))
    camera_intrin, _ = txt2intrinsic(camera_intrin_path)
    camera_pose = np.loadtxt(camera_pose_path)
    camera_extrin = np.linalg.inv(camera_pose)
    
    if paired_frames is None:  # fake version
        start_mocap_idx = 0
        rgb_imgs = video2imgs(rgb_video_path)
    else:  # real case
        start_mocap_idx = 0
        raw_rgb_imgs = video2imgs(rgb_video_path)
        camera_id_dict = {
            "d455_1": 0,
            "d455_2": 1,
            "d455_3": 2,
            "d455_4": 3,
        }
        selected_frames = [x[camera_id_dict[camera_name]] for x in paired_frames]
        rgb_imgs = [raw_rgb_imgs[x] for x in selected_frames]

    pyt3d_wrapper = Pyt3DWrapper(image_size=(1280, 720), use_fixed_cameras=True, intrin=camera_intrin, extrin=camera_extrin, device=device)
    pyt3d_wrapper2 = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[0.1, 3.0, 0.0]], device=device)  # eye 不能和 [0,1,0] 方向重合
    pyt3d_wrapper3 = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[0.0, 2.0, 2.0]], device=device)
    pyt3d_wrapper4 = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[0.0, 2.0, -2.0]], device=device)
    pyt3d_wrapper5 = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[2.0, 2.0, 0.0]], device=device)
    pyt3d_wrapper6 = Pyt3DWrapper(image_size=(640, 360), use_fixed_cameras=False, eyes=[[-2.0, 2.0, 0.0]], device=device)

    imgs = []
    for i in range(frame_range[0], frame_range[1]+1, M):
        top_i = min(i+M-1, frame_range[1])
        print("visualizing {} to {} ...".format(str(i), str(top_i)))
        if cfg["render_person1"]:
            person1_result_dir = join(data_dir, "SMPLX_fitting", "person_1", "{}to{}.npz".format(str(i), str(top_i)))
            person1_result = np.load(person1_result_dir, allow_pickle=True)["results"].item()
        if cfg["render_person2"]:
            person2_result_dir = join(data_dir, "SMPLX_fitting", "person_2", "{}to{}.npz".format(str(i), str(top_i)))
            person2_result = np.load(person2_result_dir, allow_pickle=True)["results"].item()
        
        num_pca_comps = person1_result["left_hand_pose"].shape[1]
        smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=num_pca_comps, flat_hand_mean=True)
        smplx_model.to("cuda:0")

        for j in range(top_i-i+1):
            paired_frame_idx = int((i + j) - start_mocap_idx / 3)
            img = rgb_imgs[paired_frame_idx]
            
            # # render person1
            # if cfg["render_person1"]:
            #     betas = person1_result["betas"][j:j+1].to(device)
            #     body_pose = person1_result["body_pose"][j:j+1].to(device)
            #     transl = person1_result["transl"][j:j+1].to(device)
            #     global_orient = person1_result["global_orient"][j:j+1].to(device)
            #     left_hand_pose = person1_result["left_hand_pose"][j:j+1].to(device)
            #     right_hand_pose = person1_result["right_hand_pose"][j:j+1].to(device)
            #     img = render_SMPLX(pyt3d_wrapper, smplx_model, betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose, rgb_img=img, frame_idx=None, suffix="", save=False)
            
            # # render person2
            # if cfg["render_person2"]:
            #     betas = person2_result["betas"][j:j+1].to(device)
            #     body_pose = person2_result["body_pose"][j:j+1].to(device)
            #     transl = person2_result["transl"][j:j+1].to(device)
            #     global_orient = person2_result["global_orient"][j:j+1].to(device)
            #     left_hand_pose = person2_result["left_hand_pose"][j:j+1].to(device)
            #     right_hand_pose = person2_result["right_hand_pose"][j:j+1].to(device)
            #     img = render_SMPLX(pyt3d_wrapper, smplx_model, betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose, rgb_img=img, frame_idx=None, suffix="", save=False)

            data = {
                "person1": None,
                "person2": None,
                "object": None,
            }
            if cfg["render_person1"]:
                data["person1"] = {
                    "betas": person1_result["betas"][j:j+1].to(device),
                    "body_pose": person1_result["body_pose"][j:j+1].to(device),
                    "transl": person1_result["transl"][j:j+1].to(device),
                    "global_orient": person1_result["global_orient"][j:j+1].to(device),
                    "left_hand_pose": person1_result["left_hand_pose"][j:j+1].to(device),
                    "right_hand_pose": person1_result["right_hand_pose"][j:j+1].to(device),
                }
            if cfg["render_person2"]:
                data["person2"] = {
                    "betas": person2_result["betas"][j:j+1].to(device),
                    "body_pose": person2_result["body_pose"][j:j+1].to(device),
                    "transl": person2_result["transl"][j:j+1].to(device),
                    "global_orient": person2_result["global_orient"][j:j+1].to(device),
                    "left_hand_pose": person2_result["left_hand_pose"][j:j+1].to(device),
                    "right_hand_pose": person2_result["right_hand_pose"][j:j+1].to(device),
                }
            if cfg["render_object"]:
                data["object"] = {
                    "mesh": object_data["mesh"],
                    "obj2world": object_data["obj2world"][paired_frame_idx]
                }
            if not render_contact:
                img = render_HHO(pyt3d_wrapper, smplx_model, data, rgb_img=img, frame_idx=None, suffix="", save=False, device=device, render_contact=False)
            else:
                img, contact_person1, contact_person2, contact_obj = render_HHO(pyt3d_wrapper, smplx_model, data, rgb_img=img, frame_idx=None, suffix="", save=False, device=device, render_contact=True)
                img2 = render_HHO(pyt3d_wrapper2, smplx_model, data, rgb_img=None, frame_idx=None, suffix="", save=False, device=device, render_contact=False)
                img3 = render_HHO(pyt3d_wrapper3, smplx_model, data, rgb_img=None, frame_idx=None, suffix="", save=False, device=device, render_contact=False)
                img4 = render_HHO(pyt3d_wrapper4, smplx_model, data, rgb_img=None, frame_idx=None, suffix="", save=False, device=device, render_contact=False)
                img5 = render_HHO(pyt3d_wrapper5, smplx_model, data, rgb_img=None, frame_idx=None, suffix="", save=False, device=device, render_contact=False)
                img6 = render_HHO(pyt3d_wrapper6, smplx_model, data, rgb_img=None, frame_idx=None, suffix="", save=False, device=device, render_contact=False)

            if visualize_head_orientation:
                expression = torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to(device)
                if cfg["render_person1"]:
                    betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose = data["person1"]["betas"], data["person1"]["body_pose"], data["person1"]["transl"], data["person1"]["global_orient"], data["person1"]["left_hand_pose"], data["person1"]["right_hand_pose"]
                    result_model = smplx_model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
                    result_joints = result_model.joints.detach().cpu().numpy()[0]
                    global_head_pos = result_joints[15]
                    pelvis_rot = batch_rodrigues(data["person1"]["global_orient"][0].view(1, 3))[0]
                    local_rot = batch_rodrigues(data["person1"]["body_pose"][0])
                    global_head_rot = (pelvis_rot @ local_rot[2] @ local_rot[5] @ local_rot[8] @ local_rot[11] @ local_rot[14]).detach().cpu().numpy()
                    img = draw_head_orientation(img, global_head_pos, global_head_rot, camera_intrin, camera_extrin)
                if cfg["render_person2"]:
                    betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose = data["person2"]["betas"], data["person2"]["body_pose"], data["person2"]["transl"], data["person2"]["global_orient"], data["person2"]["left_hand_pose"], data["person2"]["right_hand_pose"]
                    result_model = smplx_model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
                    result_joints = result_model.joints.detach().cpu().numpy()[0]
                    global_head_pos = result_joints[15]
                    pelvis_rot = batch_rodrigues(data["person2"]["global_orient"][0].view(1, 3))[0]
                    local_rot = batch_rodrigues(data["person2"]["body_pose"][0])
                    global_head_rot = (pelvis_rot @ local_rot[2] @ local_rot[5] @ local_rot[8] @ local_rot[11] @ local_rot[14]).detach().cpu().numpy()
                    img = draw_head_orientation(img, global_head_pos, global_head_rot, camera_intrin, camera_extrin)

            img = cv2.resize(img, (640, 360))
            if not render_contact:
                imgs.append(img)
            else:
                merged_img = np.zeros((360 * 3, 640 * 3, 3)).astype(np.uint8)
                merged_img[:360, :640] = img
                merged_img[:360, 640:1280] = img2
                merged_img[360:720, :640] = img3
                merged_img[360:720, 640:1280] = img4
                merged_img[720:, :640] = img5
                merged_img[720:, 640:1280] = img6
                merged_img[:360, 1280:] = contact_obj
                merged_img[360:720, 1280:] = contact_person1
                merged_img[720:, 1280:] = contact_person2
                cv2.putText(merged_img, "/".join(data_dir.split("/")[-2:]) + " " + str(i + j), (800, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                imgs.append(merged_img)
    
    if video_writer is None:
        imageio.mimsave(save_path, imgs)
    else:
        for img in imgs:
            video_writer.write(img[:, :, ::-1].astype(np.uint8))


def visualize_world_pcd(rgb_video_path, depth_imgs_dir, intrinsic_mat, cam_pose, frame_idx):
    cap = cv2.VideoCapture(rgb_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # get rgbd data
    rgb = None
    frame_cnt = -1
    while True:
        frame_cnt += 1
        suc, img = cap.read()
        if not suc:
            break
        if frame_cnt == int(frame_idx):
            img = img[:, :, ::-1].astype(np.uint8)
            rgb = o3d.geometry.Image(img)
            break
    assert not rgb is None
    depth = o3d.io.read_image(join(depth_imgs_dir, str(frame_idx) + ".png"))

    # get camera-space pcd
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False, depth_trunc=10.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # cam2world
    points = np.float32(pcd.points)
    points = points @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def visualize_mv_pcds(data_dir, camera_info_dir, cfg, paired_frame_idx=0):
    if not cfg["vis_camera1"]:
        raise NotImplementedError
    
    # # prepare depth imgs
    # avi2depth(join(data_dir, "_d455_camera1_aligned_depth_to_color_image_raw.avi"), join(data_dir, "decode_d455_camera1"))
    # if cfg["vis_camera2"]:
    #     avi2depth(join(data_dir, "_d455_camera2_aligned_depth_to_color_image_raw.avi"), join(data_dir, "decode_d455_camera2"))
    # if cfg["vis_camera3"]:
    #     avi2depth(join(data_dir, "_d455_camera3_aligned_depth_to_color_image_raw.avi"), join(data_dir, "decode_d455_camera3"))
    # if cfg["vis_camera4"]:
    #     avi2depth(join(data_dir, "_d455_camera4_aligned_depth_to_color_image_raw.avi"), join(data_dir, "decode_d455_camera4"))
    
    # read camera params
    cam1_intrinsic, _ = txt2intrinsic(join(camera_info_dir, "d455_1", "intrinsic.txt"))
    cam1_pose = np.loadtxt(join(camera_info_dir, "d455_1", "camera2world.txt"))
    cam2_intrinsic, _ = txt2intrinsic(join(camera_info_dir, "d455_2", "intrinsic.txt"))
    cam2_pose = np.loadtxt(join(camera_info_dir, "d455_2", "camera2world.txt"))
    cam3_intrinsic, _ = txt2intrinsic(join(camera_info_dir, "d455_3", "intrinsic.txt"))
    cam3_pose = np.loadtxt(join(camera_info_dir, "d455_3", "camera2world.txt"))
    cam4_intrinsic, _ = txt2intrinsic(join(camera_info_dir, "d455_4", "intrinsic.txt"))
    cam4_pose = np.loadtxt(join(camera_info_dir, "d455_4", "camera2world.txt"))

    paried_frames = txt_to_paried_frameids(join(data_dir, "aligned_frame_ids.txt"))
    ids = paried_frames[paired_frame_idx][:4]

    pcd = visualize_world_pcd(join(data_dir, "_d455_camera1_color_image_raw.mp4"), join(data_dir, "decode_d455_camera1"), cam1_intrinsic, cam1_pose, ids[0])
    if cfg["vis_camera2"]:
        pcd += visualize_world_pcd(join(data_dir, "_d455_camera2_color_image_raw.mp4"), join(data_dir, "decode_d455_camera2"), cam2_intrinsic, cam2_pose, ids[1])
    if cfg["vis_camera3"]:
        pcd += visualize_world_pcd(join(data_dir, "_d455_camera3_color_image_raw.mp4"), join(data_dir, "decode_d455_camera3"), cam3_intrinsic, cam3_pose, ids[2])
    if cfg["vis_camera4"]:
        pcd += visualize_world_pcd(join(data_dir, "_d455_camera4_color_image_raw.mp4"), join(data_dir, "decode_d455_camera4"), cam4_intrinsic, cam4_pose, ids[3])
    
    o3d.io.write_point_cloud(join(data_dir, "vis_mvpcd_{}.ply".format(str(paired_frame_idx))), pcd)
    

if __name__ == "__main__":
    #############################################
    # data_path = "/home/liuyun/HHO-dataset/data_processing/exp_data/20230428_data/SMPLX_fitting/1300to1319.npz"
    # video_path = "/home/liuyun/HHO-dataset/data_processing/exp_data/20230428_data/rgbd/_d435i_camera2_color_image_raw_compressed.mp4"
    # rgb_img = read_a_image_from_video(video_path, frame_idx=int((1319-80/90*30)/30*30))
    # camera_intrin, _ = txt2intrinsic("/home/liuyun/HHO-dataset/data_processing/camera_info/d435i_2/intrinsic.txt")
    # camera_pose = np.loadtxt("/home/liuyun/HHO-dataset/data_processing/camera_info/d435i_2/camera2world.txt")
    # camera_extrin = np.linalg.inv(camera_pose)

    # pyt3d_wrapper = Pyt3DWrapper(image_size=(1280, 720), use_fixed_cameras=True, intrin=camera_intrin, extrin=camera_extrin, device="cuda:0")

    # data = np.load(data_path, allow_pickle=True)["results"].item()
    # num_pca_comps = data["left_hand_pose"].shape[1]
    # smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=num_pca_comps, flat_hand_mean=True)
    # smplx_model.to("cuda:0")

    # print("finish loading data")

    # # relative frame 19
    # betas = data["betas"][-1:]
    # body_pose = data["body_pose"][-1:]
    # transl = data["transl"][-1:]
    # global_orient = data["global_orient"][-1:]
    # left_hand_pose = data["left_hand_pose"][-1:]
    # right_hand_pose = data["right_hand_pose"][-1:]

    # render_SMPLX(pyt3d_wrapper, smplx_model, betas, body_pose, transl, global_orient, left_hand_pose, right_hand_pose, rgb_img=rgb_img, camera_intrin=camera_intrin, camera_extrin=camera_extrin, frame_idx=None, suffix="")
    #############################################

    seq_dir = "/share/datasets/HHO_dataset/20230724/test1"
    cfg = {
        "camera1": True,
        "camera2": True,
        "camera3": True,
        "camera4": True,
        "person1": False,
        "person2": False,
        "object": False,
    }
    time_align(seq_dir, cfg, threshould=40000000)
    cfg = {
        "vis_camera1": True,
        "vis_camera2": False,
        "vis_camera3": False,
        "vis_camera4": False,
    }
    visualize_mv_pcds(seq_dir, "/home/liuyun/HHO-dataset/data_processing/camera_info", cfg, paired_frame_idx=0)
    exit(0)

    device = "cuda:0"
    cfg = {
        "render_person1": True,
        "render_person2": True,
    }

    data_dir = "/share/datasets/HHO_dataset/20230701/012"
    camera_name = "d455_1"
    camera_intrin_path = "/home/liuyun/HHO-dataset/data_processing/camera_info/{}/intrinsic.txt".format(camera_name)
    camera_pose_path = "/home/liuyun/HHO-dataset/data_processing/camera_info/{}/camera2world.txt".format(camera_name)
    frame_range = [0, 300]
    M = 50
    save_path = "./20230701_012.gif"
    render_SMPLX_sequence(data_dir, camera_name, camera_intrin_path, camera_pose_path, frame_range, M, save_path, device, cfg)
