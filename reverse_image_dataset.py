
from genericpath import exists
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from skimage.transform import resize
import face_alignment

import pickle
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import imageio
from PIL import Image
import cv2
from model import BiSeNet
import os
import sys
sys.path.append('./models/')
from FLAME import FLAME, FLAMETex
from Flamerenderer import FlameRenderer as Renderer
import util
torch.backends.cudnn.benchmark = True

import tensor_util
import argparse
from tqdm import tqdm
import datetime
import pathlib
import pdb
import json
#----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k",
                     type=int,
                     default=0)
    parser.add_argument("--imgsize",
                     type=int,
                     default=512)
    return parser.parse_args()

parse = parse_args()

    
def vis_tensor(image_tensor = None, image_path = None, land_tensor = None, cam = None,  visind =0, device = torch.device("cuda")):
    if land_tensor is not None:
        lmark = util.batch_orth_proj(land_tensor.to(device), cam.to(device))
        lmark[..., 1:] = - lmark[..., 1:]
        lmark = util.tensor_vis_landmarks(image_tensor.to(device)[visind].unsqueeze(0),lmark[visind].unsqueeze(0))
        output = lmark.squeeze(0)
    else:
        output = image_tensor.data[visind].detach().cpu() #  * self.stdtex + self.meantex 
    output = tensor_util.tensor2im(output  , normalize = False)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = util.writeText(output, image_path)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = np.clip(output, 0, 255)

    return output

def normalize_to_cube(v: torch.Tensor) -> torch.Tensor:
    """
    Normalizes vertices (of a mesh) to the unit cube [-1;1]^3.
    Args:
        v: [B, N, 3] - vertices.
        Handles [N, 3] vertices as well.
    Returns:
        (modified) v: [B, N, 3].
    """
    if v.ndim == 2:
        v = v[None]
    v = v - v.min(1, True)[0]
    v = v - 0.5 * v.max(1, True)[0]
    return v / v.max(-1, True)[0].max(-2, True)[0]




def to_camera(points_world, extrinsics):
    """map points in world space to camera space
    Args:
        points_world (B, 3, H, W): points in world space.
        extrinsics (B, 3, 4): [R, t] of camera.
    Returns:
        points_camera (B, 3, H, W): points in camera space.
    """
    B, p_dim, H, W = points_world.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)
    
    # map to camera:
    # R'^T * (p - t') where t' of (B, 3, 1), R' of (B, 3, 3) and p of (B, 3, H*W)
    R_tgt = extrinsics[..., :p_dim]
    t_tgt = extrinsics[..., -1:]
    points_cam = torch.bmm(R_tgt.transpose(1, 2), points_world.view(B, p_dim, -1) - t_tgt)

    return points_cam.view(B, p_dim, H, W)


def point_to_image(point_world, intrinsics, extrinsics, resolution):
    """map point in world space to image pixel
    Args:
        point_world (1, 3): point in world space.
        intrinsics (3, 3): camera intrinsics of target camera
        extrinsics (3, 4): camera extrinsics of target camera
        resolution (1): scalar, image resolution.
    Returns:
        pixel_image2 (2): pixel coordinates on target image plane.
    """
    K = intrinsics
    RT = extrinsics
    points_cam2 = to_camera(point_world.unsqueeze(-1).unsqueeze(-1), RT.unsqueeze(0))[:, :, 0, 0]
    points_image2 = ((points_cam2 / points_cam2[:, 2]) * K[0, 0] + K[0, 2]) * resolution
    pixel_image2 = points_image2[0, :2]
    return pixel_image2


class PhotometricFitting(object):
    def __init__(self, config, device='cuda'):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.config = config
        self.device = device
        #
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)
        
        self._setup_landmark_detector()
        self._setup_face_parser()
        self._setup_renderer()

    def _setup_landmark_detector(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    
    def _setup_face_parser(self):
        self.parse_net = BiSeNet(n_classes=19)
        self.parse_net.cuda()
        self.parse_net.load_state_dict(torch.load("/home/us000218/lelechen/basic/flame_data/data/79999_iter.pth"))
        self.parse_net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.frontal_regions = [1, 2, 3, 4, 5, 10, 12, 13]

    def _setup_renderer(self):
        mesh_file = '/home/us000218/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

    def get_face_landmarks(self, img):
        preds = self.fa.get_landmarks(img)
        if len(preds) == 0:
            print("ERROR: no face detected!")
            exit(1)
        return preds[0] # (68,2)
    
    def get_front_face_mask(self, img):
        with torch.no_grad():
            img = Image.fromarray(img)
            w, h = img.size
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.parse_net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)        
        msk = np.zeros_like(parsing)
        for i in self.frontal_regions:
            msk[np.where(parsing==i)] = 255        
        msk = msk.astype(np.uint8)
        msk = resize(msk, (h,w))
        return msk # (h,w), 0~1

    def optimize(self, images, landmarks, image_masks, savefolder=None, show = False):
        itt = 4000
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        # pdb.set_trace()
        cam = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        e_opt = torch.optim.Adam(
            [shape, exp, pose, cam, tex, lights],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt_rigid = torch.optim.Adam(
            [pose, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        gt_landmark = landmarks
        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(int(itt * 0.3)):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if show:
                if k % 10 == 0:
                    print(loss_info)

                if k % 499 == 0:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))

                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                    # grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
         # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for k in range(int(itt * 0.3), itt):
            losses = {}
            vertices, landmarks2d, landmarks3d_save = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * self.config.w_pose_reg

            ## render
            albedos = self.flametex(tex, self.image_size) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * self.config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if show:
                if k % 10 == 0:
                    print(loss_info)            
                if k % 499 == 0:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                    grids['albedoimage'] = torchvision.utils.make_grid(
                        (ops['albedo_images'])[visind].detach().cpu())
                    grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                    shape_images = self.render.render_shape(vertices, trans_vertices, images)
                    grids['shape'] = torchvision.utils.make_grid(shape_images[visind]).detach().float().cpu()
                    grids['tex'] = torchvision.utils.make_grid(albedos[visind]).detach().cpu()
                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))

        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'trans_verts': trans_vertices.detach().cpu().numpy(),
            'verts': vertices.detach().cpu().numpy(),
            'landmark3d': landmarks3d_save.detach().cpu().numpy(),
            'albedos':albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy(),
            'image_masks': image_masks.detach().cpu().numpy()
        }

        pdb.set_trace()
        verts = single_params['verts'][0]
        faces = torch.load('../Consistent_Facial_Landmarks/model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        output_dir = os.path.join('./gg_512', 'verts.obj')
        with open(output_dir, 'w') as f:
            for vertex in verts:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))
        pdb.set_trace()
        verts = single_params['trans_verts'][0]
        faces = torch.load('../Consistent_Facial_Landmarks/model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        output_dir = os.path.join('./gg_512', 'trans_verts.obj')
        with open(output_dir, 'w') as f:
            for vertex in verts:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))
        pdb.set_trace()

        return single_params

    
    def eg3d_cam_to_flame_pose(self, cam_ext):
        from scipy.spatial.transform import Rotation as R
        r_ext = R.from_matrix(cam_ext.cpu().numpy()[:3, :3])
        r_ext_euler = r_ext.as_euler('zyx', degrees=True)
        # r_ext_rectified = R.from_euler('zyx', r_ext_euler - np.array([0, 0, 180]), degrees=True)
        r_ext_rectified = R.from_euler('zyx', r_ext_euler - np.array([0, 0, 0]), degrees=True)
        r_ext_rectified_rotmat = r_ext_rectified.as_matrix()
        r_ext_rectified_rotvec = r_ext_rectified.as_rotvec()
        ext_rotvec = torch.from_numpy(r_ext_rectified_rotvec).unsqueeze(0).float().to(self.device)
        ext_rotmat = torch.from_numpy(r_ext_rectified_rotmat).unsqueeze(0).float().to(self.device)
        return ext_rotvec, ext_rotmat
    
    
    def optimize_ortho_mouth(self, images, landmarks, image_masks, cam_npy=None, savefolder=None, show = False):
        itt = 4000 # 4000
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        # pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        pose_mouth = nn.Parameter(torch.zeros(bz, 3).float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        cam = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))
        
        cam_ext = torch.from_numpy(cam_npy['extrinsics']).float().to(self.device)
        ext_angles = self.eg3d_cam_to_flame_pose(cam_ext)
        pose = torch.cat((ext_angles, pose_mouth), dim=1)
        # r = R.from_matrix([[0.8897, 0.0185, -0.4561], [0.0111, -0.9998, -0.0187], [-0.4563, 0.0116, -0.8897]])
        # r.as_rotvec()
        # array([ 3.03861397,  0.02379947, -0.73360272])

        # pdb.set_trace()
        
        e_opt = torch.optim.Adam(
            [shape, exp, pose_mouth, tex, lights, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt_rigid = torch.optim.Adam(
            [pose_mouth, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        gt_landmark = landmarks
        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(int(itt * 0.3)):
            losses = {}
            pose = torch.cat((ext_angles, pose_mouth), dim=1)
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if show:
                if k % 10 == 0:
                    print(loss_info)

                if k % 499 == 0:
                    # pdb.set_trace()
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))

                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                    # grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
            
        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for k in range(int(itt * 0.3), itt):
            losses = {}
            pose = torch.cat((ext_angles, pose_mouth), dim=1)
            vertices, landmarks2d, landmarks3d_save = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * self.config.w_pose_reg

            ## render
            albedos = self.flametex(tex, self.image_size) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * self.config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if show:
                if k % 10 == 0:
                    print(loss_info)            
                if k % 499 == 0:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                    grids['albedoimage'] = torchvision.utils.make_grid(
                        (ops['albedo_images'])[visind].detach().cpu())
                    grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                    shape_images = self.render.render_shape(vertices, trans_vertices, images)
                    grids['shape'] = torchvision.utils.make_grid(shape_images[visind]).detach().float().cpu()
                    grids['tex'] = torchvision.utils.make_grid(albedos[visind]).detach().cpu()
                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        
        pose = torch.cat((ext_angles, pose_mouth), dim=1)
        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': vertices.detach().cpu().numpy(),
            'trans_verts': trans_vertices.detach().cpu().numpy(),
            'landmark3d': landmarks3d_save.detach().cpu().numpy(),
            'albedos':albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy(),
            'image_masks': image_masks.detach().cpu().numpy()
        }

        # pdb.set_trace()
        verts = single_params['verts'][0]
        faces = torch.load('../Consistent_Facial_Landmarks/model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        output_dir = os.path.join('./gg_512', 'verts.obj')
        with open(output_dir, 'w') as f:
            for vertex in verts:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))
        # pdb.set_trace()
        verts = single_params['trans_verts'][0]
        faces = torch.load('../Consistent_Facial_Landmarks/model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        output_dir = os.path.join('./gg_512', 'trans_verts.obj')
        with open(output_dir, 'w') as f:
            for vertex in verts:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))
        # pdb.set_trace()


        return single_params


    def optimize_ortho_mouth_cam(self, images, landmarks, image_masks, cam_npy=None, savefolder=None, show = False):
        itt = 4000 # 4000
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        # pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        pose_mouth = nn.Parameter(torch.zeros(bz, 3).float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        cam = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))
        
        cam_ext = torch.from_numpy(cam_npy['extrinsics']).float().to(self.device)
        
        ROT_COLMAP_TO_NORMAL = np.diag([1, 1, 1])
        extr = cam_npy['extrinsics']
        # pdb.set_trace()
        R, t = extr[:3, :3], extr[:3, -1:]
        Rc = ROT_COLMAP_TO_NORMAL.dot(R.T).dot(ROT_COLMAP_TO_NORMAL.T)
        tc = -Rc.dot(ROT_COLMAP_TO_NORMAL.T).dot(t)
        extrinsics = np.concatenate([Rc, tc], axis=1)
        extr_2 = torch.from_numpy(np.concatenate([extrinsics, np.array([[0, 0, 0, 1]])], axis=0)).float().to(self.device)

        ext_angles, ext_mat = self.eg3d_cam_to_flame_pose(extr_2)
        # ext_angles, ext_mat = self.eg3d_cam_to_flame_pose(cam_ext)
        pose = torch.cat((ext_angles, pose_mouth), dim=1)
        # r = R.from_matrix([[0.8897, 0.0185, -0.4561], [0.0111, -0.9998, -0.0187], [-0.4563, 0.0116, -0.8897]])
        # r.as_rotvec()
        # array([ 3.03861397,  0.02379947, -0.73360272])
        
        e_opt = torch.optim.Adam(
            [shape, exp, pose_mouth, tex, lights, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt_rigid = torch.optim.Adam(
            [pose_mouth, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        gt_landmark = landmarks
        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        
        for k in range(int(itt * 0.3)):
            losses = {}
            pose = torch.cat((ext_angles, pose_mouth), dim=1)
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            # trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            # landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            # landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if show:
                if k % 10 == 0:
                    print(loss_info)

                if k % 499 == 0:
                    # pdb.set_trace()
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))

                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                    # grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
                    
                    # pdb.set_trace()
            
        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for k in range(int(itt * 0.3), itt):
            losses = {}
            pose = torch.cat((ext_angles, pose_mouth), dim=1)
            vertices, landmarks2d, landmarks3d_save = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            # trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            # landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)
            # landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * self.config.w_pose_reg

            ## render
            albedos = self.flametex(tex, self.image_size) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * self.config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if show:
                if k % 10 == 0:
                    print(loss_info)            
                if k % 499 == 0:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                    grids['albedoimage'] = torchvision.utils.make_grid(
                        (ops['albedo_images'])[visind].detach().cpu())
                    grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                    shape_images = self.render.render_shape(vertices, trans_vertices, images)
                    grids['shape'] = torchvision.utils.make_grid(shape_images[visind]).detach().float().cpu()
                    grids['tex'] = torchvision.utils.make_grid(albedos[visind]).detach().cpu()
                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
                    # pdb.set_trace()
        
        pose = torch.cat((ext_angles, pose_mouth), dim=1)
        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': vertices.detach().cpu().numpy(),
            'trans_verts': trans_vertices.detach().cpu().numpy(),
            'landmark3d': landmarks3d_save.detach().cpu().numpy(),
            'albedos':albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy(),
            'image_masks': image_masks.detach().cpu().numpy()
        }

        verts = single_params['verts'][0]
        faces = torch.load('../Consistent_Facial_Landmarks/model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        output_dir = os.path.join('./gg_512', 'verts.obj')
        with open(output_dir, 'w') as f:
            for vertex in verts:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))
        verts = single_params['trans_verts'][0]
        faces = torch.load('../Consistent_Facial_Landmarks/model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
        output_dir = os.path.join('./gg_512', 'trans_verts.obj')
        with open(output_dir, 'w') as f:
            for vertex in verts:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))

        return single_params


    def run(self, img, vis_folder, imgmask_path = None, config = None, imglmark = None, imgcam = None ):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []

        # camera
        # pdb.set_trace()
        cam_ = np.load(imgcam)
        cam_ext = cam_[:16].reshape(4, 4)
        cam_int_ = cam_[16:].reshape(3, 3)
        cam_int = np.concatenate((cam_int_, np.zeros_like(cam_int_[:, :1])), axis=1)
        # cam = torch.from_numpy(np.matmul(cam_int, cam_ext)).to(self.device)
        cam = {'extrinsics':cam_ext, 'intrinsics':cam_int}

        # image = cv2.imread(imagepath).astype(np.float32) / 255.
        # image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        image = img.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))

        if imgmask_path is None:
            image_mask = self.get_front_face_mask(img)
            image_mask = image_mask[..., None].astype('float32')
            print (image_mask.shape)
            image_mask = np.expand_dims(cv2.resize(image_mask, (config.image_size,self.config.image_size), interpolation = cv2.INTER_AREA), axis = 0)

            # image_mask = image_mask.transpose(2, 0, 1)
        # np.save(imgmask_path, image_mask)
        else:
            image_mask = np.load(imgmask_path)
            image_mask = np.expand_dims(cv2.resize(np.load(imgmask_path).transpose(1,2,0), (config.image_size,self.config.image_size), interpolation = cv2.INTER_AREA), axis = 0)

        image_masks.append(torch.from_numpy(image_mask[None, :, :, :]).to(self.device))
        if imglmark is None:
            landmark = self.get_face_landmarks(img).astype(np.float32)
        else:
            # with open(imglmark, 'rb') as f:
            #     pdb.set_trace()
            #     landmark = np.load(f, allow_pickle=True).tolist()['point']
            landmark = np.load(imglmark)
            landmark = landmark.astype(np.float32)
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
        landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))

        images = torch.cat(images, dim=0)
        image_masks = torch.cat(image_masks, dim=0)
        landmarks = torch.cat(landmarks, dim=0)
        
        # optimize
        # single_params = self.optimize(images, landmarks, image_masks, savefolder= vis_folder, show = True)
        single_params = self.optimize_ortho_mouth_cam(images, landmarks, image_masks, cam_npy=cam, savefolder= vis_folder, show = True)
        # single_params = self.optimize2_persp_cont0(images, landmarks, image_masks, cam_npy=cam, savefolder= vis_folder, show = True)
        # single_params = self.optimize5(images, landmarks, image_masks, cam=cam, savefolder= vis_folder, show = True)
        # single_params = self.optimize4(images, landmarks, image_masks, cam_npy=cam, savefolder= vis_folder, show = True)
        # self.render.save_obj(filename=savefile[:-4]+'.obj',
        #                      vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
        #                      textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
        #                      )
        with open(f"{vis_folder}/flame_p.pickle", 'wb') as handle:
            pickle.dump(single_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # np.save(f"{vis_folder}/flame_p.npy", single_params)
        return single_params
    

    def render_views(self, base_params_path, view_img, view_cam_ext, view_dir):
        file = open(base_params_path, 'rb')
        params = pickle.load(file)
        
        ROT_COLMAP_TO_NORMAL = np.diag([1, 1, 1])
        view_ext = view_cam_ext.cpu().numpy()
        R, t = view_ext[:3, :3], view_ext[:3, -1:]
        Rc = ROT_COLMAP_TO_NORMAL.dot(R.T).dot(ROT_COLMAP_TO_NORMAL.T)
        tc = -Rc.dot(ROT_COLMAP_TO_NORMAL.T).dot(t)
        extrinsics = np.concatenate([Rc, tc], axis=1)
        extr_2 = torch.from_numpy(np.concatenate([extrinsics, np.array([[0, 0, 0, 1]])], axis=0)).float().to(self.device)

        view_ext_rotvec, _ = self.eg3d_cam_to_flame_pose(extr_2.to(self.device))
        # view_ext_rotvec, _ = self.eg3d_cam_to_flame_pose(view_cam_ext.to(self.device))
        view_intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]).float().to(self.device)

        images = []
        view_name = view_dir.split('.jpg')[0].split('/')[-1]
        scene_path = view_dir.split('jpg/')[0]
        scene_render_path = os.path.join(scene_path, 'render')
        os.makedirs(scene_render_path, exist_ok=True)
        anchor_point_path = os.path.join(scene_path, 'anchor_point.npy')
        anchor_point = torch.from_numpy(np.load(anchor_point_path)).float().to(self.device)
        anchor_point_proj = point_to_image(anchor_point, view_intrinsics, view_cam_ext[:3, :], view_img.shape[0])
                
        image = view_img.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))
        images = torch.cat(images, dim=0)
        
        pose = torch.cat((view_ext_rotvec, torch.from_numpy(params['pose'][:, 3:]).to(self.device)), dim=1)
        # pose = torch.from_numpy(params['pose']).to(self.device)
        shape = torch.from_numpy(params['shape']).to(self.device)
        exp = torch.from_numpy(params['exp']).to(self.device)
        cam = torch.from_numpy(params['cam']).to(self.device)
        tex = torch.from_numpy(params['tex']).to(self.device)
        albedos = torch.from_numpy(params['albedos']).to(self.device)
        lights = torch.from_numpy(params['lit']).to(self.device)

        vertices, landmarks2d, landmarks3d_save = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
        # pdb.set_trace()
        trans_vertices = util.batch_orth_proj(vertices, cam)
        # trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        # landmarks2d[..., 1:] = - landmarks2d[..., 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)
        # landmarks3d[..., 1:] = - landmarks3d[..., 1:]
        
        ops = self.render(vertices, trans_vertices, albedos, lights)
        predicted_images = ops['images']

        grids = {}
        bz = 1
        visind = range(bz)
        grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
        grids['landmarks_gt'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
        grids['landmarks2d'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
        vis_landmarks2d, predicted_landmark2d = util.tensor_vis_landmarks2_anchor_point(images[visind], landmarks2d[visind], anchor_point_proj)
        grids['landmarks2d_anchor'] = torchvision.utils.make_grid(vis_landmarks2d)
        grids['landmarks3d'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
        vis_landmarks3d, predicted_landmark3d = util.tensor_vis_landmarks2_anchor_point(images[visind], landmarks3d[visind], anchor_point_proj)
        grids['landmarks3d_anchor'] = torchvision.utils.make_grid(vis_landmarks3d)
        grids['albedoimage'] = torchvision.utils.make_grid(
            (ops['albedo_images'])[visind].detach().cpu())
        grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
        shape_images = self.render.render_shape(vertices, trans_vertices, images)
        grids['shape'] = torchvision.utils.make_grid(shape_images[visind]).detach().float().cpu()
        grids['tex'] = torchvision.utils.make_grid(albedos[visind]).detach().cpu()
        grid = torch.cat(list(grids.values()), 1)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

        cv2.imwrite('{}/{}.jpg'.format(scene_render_path, str(view_name).zfill(3)), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        np.save('{}/{}_lm2d_68.npy'.format(scene_render_path, str(view_name).zfill(3)), predicted_landmark3d[:, :2])
        
config = {
        # FLAME
        "savefolder" : '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf/flame2/',
        'flame_model_path': '/home/us000218/lelechen/basic/flame_data/data/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': '/home/us000218/lelechen/basic/flame_data/data/landmark_embedding.npy',
        'tex_space_path': '/home/us000218/lelechen/basic/flame_data/data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'batch_size': 1,
        'image_size': parse.imgsize,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }

config = util.dict2obj(config)


def demo(config):
    images_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/cropped_img'
    cams_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/aligned_camera'
    lmarks_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/lmark'
    config.savefolder=  './gg_512' 

    os.makedirs(config.savefolder, exist_ok= True)
    k =  parse_args().k
    gpuid = k % 7
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%gpuid)

    # aligned_camera_path = os.path.join(cams_path, 'cropping_params.json')
    # with open(aligned_camera_path) as f:
    #     cropping_params = json.load(f)
    sample_cnt = 0
    # for im_path, cropping_dict in tqdm(cropping_params.items()):
    # for im_path in ['4bbfe013-c963-4a7c-94fc-65818ee73603.png', '4bbfe534-3108-4d1c-a0a1-e34b267f25ae.png', '4bbff300-7133-4c81-aa9d-33ab819e0718.png', \
    #     '4bc04d8e-423b-4f8f-b6ce-8b7d2f739bef.png', '4bc08e3b-91cc-4d0a-89de-b34fbda6acb3.png', '4bc4dbbd-fbeb-438b-b586-2d092819fd5d.png', \
    #     '4bc129df-5c58-4f2f-9ec4-3107a3fb8e57.png', '4bdec9d6-c8d3-4372-9563-d7877619dce5.png', '4bc96479-9556-4762-afe1-1e786bae77fc.png', \
    #     '4bdb0b46-8c20-401d-b772-219724bef37e.png', '4bdd51ba-0f11-4aed-8829-960da5e2bb8f.png', '4bdd826f-5244-4047-b887-22fe750d80b5.png']:
    for im_path in ['4bdb0b46-8c20-401d-b772-219724bef37e', '4bc96479-9556-4762-afe1-1e786bae77fc', '4bbfe534-3108-4d1c-a0a1-e34b267f25ae', '4bbfe013-c963-4a7c-94fc-65818ee73603']:
        if sample_cnt >= 10:
            break
        sample_cnt += 1

        img_name = im_path
        cam_name = img_name.replace('.png', '.npy')
        lmark_name = img_name.replace('.png', '__cropped.npy')
        img_path = os.path.join(images_path, img_name)
        cam_path = os.path.join(cams_path.replace('aligned_camera', 'extracted_camera'), cam_name)
        lmark_path = os.path.join(lmarks_path, lmark_name)
        output_folder = os.path.join(config.savefolder, img_name.split('.')[0])
        os.makedirs(output_folder, exist_ok= True)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        params = fitting.run(img, vis_folder = output_folder, config=config, imglmark=lmark_path, imgcam=cam_path )
        
        # pdb.set_trace()

    pdb.set_trace()
              

def demo_views_lm2d(config):
    cams_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/aligned_camera'
    views_path = '/nfs/home/us000218/tmpdata/outputs/gg/out'
    view_cams_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/cam2world_pose.pt'
    config.savefolder=  './outputs_views' 

    os.makedirs(config.savefolder, exist_ok= True)
    k =  parse_args().k
    gpuid = k % 7
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%gpuid)

    aligned_camera_path = os.path.join(cams_path, 'cropping_params.json')
    with open(aligned_camera_path) as f:
        cropping_params = json.load(f)
    cam2world_poses = torch.load(view_cams_path)

    sample_cnt = 0
    for im_path, cropping_dict in tqdm(cropping_params.items()):
        if sample_cnt >= 10:
            break
        sample_cnt += 1
        scene_name = im_path.split('.')[0]
        params_path = os.path.join('./gg_512', scene_name, 'flame_p.pickle')
        scene_views_path = os.path.join(views_path, scene_name+'.npy')
        scene_output_path = os.path.join(config.savefolder, scene_name)
        os.makedirs(scene_output_path, exist_ok= True)
        scene_output_jpg_path = os.path.join(scene_output_path, 'jpg')
        os.makedirs(scene_output_jpg_path, exist_ok= True)
        scene_output_lm2d_path = os.path.join(scene_output_path, 'lm2d_68')
        os.makedirs(scene_output_lm2d_path, exist_ok= True)
        scene_views = np.load(scene_views_path)
        # for i in range(scene_views.shape[0]):
        for i in [0, 125, 250, 375, 390, 448]:
            pdb.set_trace()
            view_img = scene_views[i].transpose(1, 2, 0) # 512x512x3
            img = cv2.cvtColor((view_img * 127.5 + 128).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            view_dir = os.path.join(scene_output_jpg_path, 'frame_{}.jpg'.format(str(i).zfill(3)))
            # cv2.imwrite(view_dir, img)
            cam_ext = cam2world_poses[i][0]
            
            fitting.render_views(params_path, img, cam_ext, view_dir)
    # fitting.render_views3(params_path, image_path, cam_path, views_path)

    pdb.set_trace()


def demo_views(config):
    cams_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/aligned_camera'
    views_path = '/nfs/home/uss00054/projects/flame_fitting/outputs_views_ortho_mouth_cam'
    view_cams_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/cam2world_pose.pt'
    config.savefolder=  './outputs_views_ortho_mouth_cam' 

    os.makedirs(config.savefolder, exist_ok= True)
    k =  parse_args().k
    gpuid = k % 7
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%gpuid)

    # aligned_camera_path = os.path.join(cams_path, 'cropping_params.json')
    # with open(aligned_camera_path) as f:
    #     cropping_params = json.load(f)
    cam2world_poses = torch.load(view_cams_path)

    sample_cnt = 0
    # for im_path, cropping_dict in tqdm(cropping_params.items()):
    # for im_path in ['4bbfe013-c963-4a7c-94fc-65818ee73603.png', '4bbfe534-3108-4d1c-a0a1-e34b267f25ae.png', '4bbff300-7133-4c81-aa9d-33ab819e0718.png', \
    #     '4bc04d8e-423b-4f8f-b6ce-8b7d2f739bef.png', '4bc08e3b-91cc-4d0a-89de-b34fbda6acb3.png', '4bc4dbbd-fbeb-438b-b586-2d092819fd5d.png', \
    #     '4bc129df-5c58-4f2f-9ec4-3107a3fb8e57.png', '4bdec9d6-c8d3-4372-9563-d7877619dce5.png', '4bc96479-9556-4762-afe1-1e786bae77fc.png', \
    #     '4bdb0b46-8c20-401d-b772-219724bef37e.png', '4bdd51ba-0f11-4aed-8829-960da5e2bb8f.png', '4bdd826f-5244-4047-b887-22fe750d80b5.png']:
    # for im_path in ['4bdb0b46-8c20-401d-b772-219724bef37e', '4bc96479-9556-4762-afe1-1e786bae77fc', '4bbfe534-3108-4d1c-a0a1-e34b267f25ae', '4bbfe013-c963-4a7c-94fc-65818ee73603']:
    # for im_path in ['4bc96479-9556-4762-afe1-1e786bae77fc', '4bbfe534-3108-4d1c-a0a1-e34b267f25ae', '4bbfe013-c963-4a7c-94fc-65818ee73603']:
    for im_path in ['4bdb0b46-8c20-401d-b772-219724bef37e']:
        if sample_cnt >= 10:
            break
        sample_cnt += 1
        scene_name = im_path.split('.')[0]
        params_path = os.path.join('./gg_512', scene_name, 'flame_p.pickle')
        scene_views_path = os.path.join(views_path, scene_name)
        scene_output_path = os.path.join(config.savefolder, scene_name)
        os.makedirs(scene_output_path, exist_ok= True)
        scene_output_jpg_path = os.path.join(scene_output_path, 'jpg')
        os.makedirs(scene_output_jpg_path, exist_ok= True)
        # for i in range(scene_views.shape[0]):
        # for i in [0, 125, 250, 375, 390, 448]:
        for i in range(512):
            view_dir = os.path.join(scene_output_jpg_path, 'frame_{}.jpg'.format(str(i).zfill(3)))
            img = cv2.imread(view_dir)
            cam_ext = cam2world_poses[i][0]
            
            fitting.render_views(params_path, img, cam_ext, view_dir)
    # fitting.render_views3(params_path, image_path, cam_path, views_path)

   
def main_ffhq_cips3d(config,start_idx =1):
    # image_path = "./test_images/69956.png"
    # img = imageio.imread(image_path)

    config.savefolder = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_cips3d/flame2/'
    
    k =  parse_args().k
    # gpuid = k % 7
    # gpuid = 6
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%0)

    root = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_cips3d'
    
    for idx in tqdm(range(max(10000 * k,1 ),(k + 1) * 10000 )):
        try:
            img_p = os.path.join( root, 'images', '%d.png'%idx)
            if not os.path.exists( config.savefolder + '/%d/flame_p.pickle'%idx):
                os.makedirs(config.savefolder + '/%d'%idx, exist_ok = True)
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                params = fitting.run(img, vis_folder = config.savefolder + '%d'%idx)
            else:
                print (img_p,'======')
        except:
            print (img_p, '==++++++')
            continue 


def main_ffhq(config):
    config.savefolder = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_cips3d/flame/'
    
    k =  parse_args().k
    gpuid = k % 7
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%gpuid)
    
    root = '/nfs/STG/CodecAvatar/lelechen/FFHQ/ffhq-dataset'
    image_list_file = '/nfs/STG/CodecAvatar/lelechen/FFHQ/ffhq-dataset/downsample_ffhq_256x256.zip'

    num_files, input_iter = open_image_zip(image_list_file,max_images = None)
    
    
    pbar = tqdm(enumerate(input_iter), total=num_files)
    for idx, image in pbar:
        if idx > k * 70000 and idx < (k+1)*70000:
            try:
                if not os.path.exists( config.savefolder + image['label'][:-4] + '/flame_p.pickle'):
                    util.check_mkdir(config.savefolder + image['label'][:-4])
                    params = fitting.run(image['img'], vis_folder = config.savefolder + image['label'][:-4])
                else:
                    print (idx, image['label'],'======')
            except:
                print (idx, image['label'])
                continue 


def main_ffhq_stylenerf(config = config, parse = parse):

    k =  parse.k
    config.batch_size = 1
    config.image_size = parse.imgsize
    fitting = PhotometricFitting(config, device="cuda:%d"%0)

    root = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf'
    
    for idx in tqdm(range(max(10000 * k,1 ),(k + 1) * 10000 )):
        # if  idx > 166000 :
        #     continue
        try:
                img_p = os.path.join( root, 'images', '%06d.png'%idx)

                # # check the file modity time.
                # f_name = pathlib.Path(config.savefolder + '/%06d/flame_p.pickle'%idx)
                # # get modification time
                # m_timestamp = f_name.stat().st_mtime
                # m_time = datetime.datetime.fromtimestamp(m_timestamp)
                # day = int(str(m_time).split('-')[2][:2])    
                # if day > 24:
                #     continue
               
                if not os.path.exists( config.savefolder + '/%06d/flame_p.pickle'%idx):
                    os.makedirs(config.savefolder + '/%06d'%idx, exist_ok = True)
                    img = cv2.imread(img_p)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (config.image_size,config.image_size), interpolation = cv2.INTER_AREA)

                    imgmask_path = os.path.join( root, 'imagemasks', '%06d.npy'%idx)
                    params = fitting.run(img, vis_folder = config.savefolder + '%06d'%idx, imgmask_path=imgmask_path,config =config)
                # else:
                #     print (img_p,'======')
        except:
            print (img_p, '==++++++')
            continue 


def varify(config = config, parse = parse):
    device ='cuda'
    k =  parse.k
    config.batch_size = 1
    config.image_size = parse.imgsize
    flame = FLAME(config).to(device)
    flametex = FLAMETex(config).to(device)
    mesh_file = '/home/us000218/lelechen/basic/flame_data/data/head_template_mesh.obj'
    render = Renderer(config.image_size, obj_filename=mesh_file).to(device)

    root = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf'
    for idx in tqdm(range(max(100 * k,1 ),(k + 1) * 100 )):
        img_p = os.path.join( root, 'images', '%06d.png'%idx)
        flame_path = config.savefolder + '/%06d/flame_p.pickle'%idx
        with open(flame_path, 'rb') as f:
            flame_p = pickle.load(f, encoding='latin1')
        shape = flame_p['shape']#[1,100]
        exp = flame_p['exp'] #[1,50]
        pose = flame_p['pose'] #[1,6]
        cam = flame_p['cam'] #[1,3]
        tex = flame_p['tex'] #[1,50]
        lit = flame_p['lit'] #[1,9,3]
        
        shape = torch.FloatTensor(shape).to(device)
        exp = torch.FloatTensor(exp).to(device)
        pose = torch.FloatTensor(pose).to(device)
        cam = torch.FloatTensor(cam).to(device)
        tex = torch.FloatTensor(tex).to(device)
        lights = torch.FloatTensor(lit).to(device)

        vertices, landmarks2d, landmarks3d_save = flame(shape_params=shape, expression_params=exp, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        landmarks2d[..., 1:] = - landmarks2d[..., 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)
        landmarks3d[..., 1:] = - landmarks3d[..., 1:]
        ## render
        albedos = flametex(tex, config.image_size ) / 255.
        ops = render(vertices, trans_vertices, albedos, lights)
        predicted_images = ops['images']

        genimage = vis_tensor(image_tensor= predicted_images, 
                image_path = img_p,
                device = device
                )
        genimage = cv2.cvtColor(genimage, cv2.COLOR_RGB2BGR)
        gtimage = cv2.imread(img_p)
        gtimage = cv2.resize(gtimage, (config.image_size,config.image_size), interpolation = cv2.INTER_AREA)

        img = cv2.hconcat([genimage, gtimage])
        cv2.imwrite(root + '/tmp/%d.png'%idx, img)

# varify()
# main_ffhq_stylenerf()

# demo(config)
demo_views(config)
