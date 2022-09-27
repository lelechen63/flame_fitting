
from genericpath import exists
import torch
import torchvision
import torchvision.transforms as transforms
import PIL 

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
import zipfile
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
def is_image_ext(fname):
    ext = str(fname).split('.')[-1].lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore


def open_image_folder(source, max_images = None):
    tmp = os.listdir(source)
    input_images = []
    for i in tmp:
        if i[-3:] == 'png':
            input_images.append(i)
    if max_images is not None:
        max_idx = min(len(input_images), max_images)
    else:
        max_idx = len(input_images)
    return max_idx, input_images


def optimize_ortho_mouth_cam(self, images, landmarks, image_masks, cam_npy=None, savefolder=None, show = False):
    itt = 4000 # 4000
    bz = images.shape[0]
    shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
    tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
    exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
    pose_mouth = nn.Parameter(torch.zeros(bz, 3).float().to(self.device))
    lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
    cam = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
    cam = nn.Parameter(cam.float().to(self.device))
    cam_ext = torch.from_numpy(cam_npy['extrinsics']).float().to(self.device)
    ROT_COLMAP_TO_NORMAL = np.diag([1, 1, 1])
    extr = cam_npy['extrinsics']
    R, t = extr[:3, :3], extr[:3, -1:]
    Rc = ROT_COLMAP_TO_NORMAL.dot(R.T).dot(ROT_COLMAP_TO_NORMAL.T)
    tc = -Rc.dot(ROT_COLMAP_TO_NORMAL.T).dot(t)
    extrinsics = np.concatenate([Rc, tc], axis=1)
    extr_2 = torch.from_numpy(np.concatenate([extrinsics, np.array([[0, 0, 0, 1]])], axis=0)).float().to(self.device)
    ext_angles, ext_mat = self.eg3d_cam_to_flame_pose(extr_2)
    pose = torch.cat((ext_angles, pose_mouth), dim=1)

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
                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
                        
    # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
    for k in range(int(itt * 0.3), itt):
        losses = {}
        pose = torch.cat((ext_angles, pose_mouth), dim=1)
        vertices, landmarks2d, landmarks3d_save = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)

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

    return single_params



class PhotometricFitting(object):
    def __init__(self, config, device='cuda', ):
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
                    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)
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

                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            'landmark3d': landmarks3d_save.detach().cpu().numpy(),
            'albedos':albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy(),
            'image_masks': image_masks.detach().cpu().numpy()
        }
        return single_params

    def run(self, img, vis_folder, imgmask_path = None, config = None, imglmark = None ):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []

        image = img.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))

        if imgmask_path is None:
            image_mask = self.get_front_face_mask(img)
            image_mask = image_mask[..., None].astype('float32')
            image_mask = np.expand_dims(cv2.resize(image_mask, (config.image_size,self.config.image_size), interpolation = cv2.INTER_AREA), axis = 0)
        else:
            image_mask = np.load(imgmask_path)
            image_mask = np.expand_dims(cv2.resize(np.load(imgmask_path).transpose(1,2,0), (config.image_size,self.config.image_size), interpolation = cv2.INTER_AREA), axis = 0)

        image_masks.append(torch.from_numpy(image_mask[None, :, :, :]).to(self.device))
        if imglmark is None:
            landmark = self.get_face_landmarks(img).astype(np.float32)
        else:
            with open(imglmark, 'rb') as f:
                landmark = np.load(f, allow_pickle=True).tolist()['point']
            landmark = landmark.astype(np.float32)
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
        landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))

        images = torch.cat(images, dim=0)
        image_masks = torch.cat(image_masks, dim=0)
        landmarks = torch.cat(landmarks, dim=0)
        # optimize
        single_params = self.optimize(images, landmarks , image_masks, savefolder= vis_folder, show = True)
        # self.render.save_obj(filename=savefile[:-4]+'.obj',
        #                      vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
        #                      textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
        #                      )
        with open(f"{vis_folder}/flame_p.pickle", 'wb') as handle:
            pickle.dump(single_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # np.save(f"{vis_folder}/flame_p.npy", single_params)
        return single_params


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
    image_path = '/nfs/STG/CodecAvatar/lelechen/libingzeng/EG3D_Inversion/dataset_preprocessing/ffhq/DAD-3DHeads-Datasets/val_lm2d/preprocess_images_68/realign1500/cropped_images/00e84619-0bdf-4181-af4a-cf5a27950a4d_align1500.png'
    lmark_path = '/nfs/STG/CodecAvatar/lelechen/libingzeng/EG3D_Inversion/dataset_preprocessing/ffhq/DAD-3DHeads-Datasets/val_lm2d/preprocess_images_68/realign1500_lm/00e84619-0bdf-4181-af4a-cf5a27950a4d_align1500_transformed_1500_lm2d_points_cropped.npy'
    # image_path = '/nfs/STG/CodecAvatar/lelechen/libingzeng/Consistent_Facial_Landmarks/temp/2.png'
    # image_path = "/nfs/STG/CodecAvatar/lelechen/libingzeng/EG3D_Inversion/dataset_preprocessing/ffhq/preprocess_image_local2/realign1500_local/2.png_align1500.png"
    config.savefolder=  './gg_512' 

    img = cv2.imread(image_path)
    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        
    os.makedirs(config.savefolder, exist_ok= True)
    k =  parse_args().k
    gpuid = k % 7
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%gpuid)
    params = fitting.run(img, vis_folder = config.savefolder, config=config, imglmark=lmark_path )
              
def main_ffhq(config):
    config.savefolder = '/nfs/STG/CodecAvatar/lelechen/FFHQ/ffhq-dataset/flame/'
    config.imgfolder = '/nfs/STG/CodecAvatar/lelechen/FFHQ/ffhq-dataset/images1024x1024_cropped/'
    
    k =  parse_args().k
    gpuid = k % 7
    config.batch_size = 1
    fitting = PhotometricFitting(config, device="cuda:%d"%gpuid)

    num_files, img_names = open_image_folder(config.imgfolder,max_images = None)
    
    pbar = tqdm(enumerate(img_names), total=num_files)
    for idx, image in pbar:
        print (image)
        t_f = int(int(image[:-4]) / 1000)

        print (config.savefolder +'%02d000/'%(t_f) + image[:-4] + '/flame_p.pickle')
        if os.path.exists(config.savefolder +'%02d000/'%(t_f) + image[:-4] + '/flame_p.pickle' ):            
            pass
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
main_ffhq(config)
# demo(config)