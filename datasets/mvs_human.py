from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import torch
import scipy.io as io
import copy, json 

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, 
        ndepths=192, interval_scale=1.06, 
        img_mean=None, img_std=None, out_scale=1.0, self_norm=False, color_mode="RGB", 
        is_stage=False, stage_info=None, random_view=False, img_interp="linear", 
        random_crop=False, crop_h=512, crop_w=256, depth_num=4, **kwargs):
        super(MVSDataset, self).__init__()
        if mode=="train":
            self.peoplenames=["wangjingjing","wangyijie","zhaoyunfei","zhengjunshen","zhonglin","zyf","wenjiayi"]#os.listdir(datapath)
        elif mode=="val" or mode=="test":
            self.peoplenames=["zrr"]#["wangjingjing","wangyijie","wenjiayi","yefan","zhaoyunfei","zhengjunshen","zhonglin"]#os.listdir(datapath)
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.img_mean = img_mean
        self.img_std = img_std
        self.out_scale = out_scale
        self.self_norm = self_norm
        self.color_mode = color_mode
        self.is_stage = is_stage
        self.stage_info = stage_info
        self.random_view = random_view
        if img_interp == "linear":
            self.img_interp = cv2.INTER_LINEAR
        elif img_interp == "nearest":
            self.img_interp = cv2.INTER_NEAREST
        else:
            self.img_interp = cv2.INTER_LINEAR
        self.random_crop = random_crop
        self.depth_num = depth_num
        self.camparam=self.readcampara()

        assert self.mode in ["train", "val","test"]
        self.metas = self.build_list()
    def readcampara(self):
        cam_param={}
        for peoplename in self.peoplenames:
            optimizedcameraKM=json.load(open(self.datapath+peoplename+"/Cameraparamters.json","r"))
            depth_optimized=io.loadmat(self.datapath+peoplename+"/depth_optimized.mat")
            depth_optimized=depth_optimized['depth_optimized']
            errors=depth_optimized[0,0][1]
            optimizedcameraKM={
                "camparam":optimizedcameraKM,
                "systermerrors":errors
            }
            cam_param.update({peoplename:optimizedcameraKM})
        return cam_param
    def readview(self,path):
        view = []
        lines = open(path).readlines()
        for line in lines:
            a = line.split('\n')[0].split(',')
            view.append(a)
        return view    

    def build_list(self):
        metas = []
        for peoplename in self.peoplenames:
            with open(self.datapath+peoplename+"/"+self.listfile) as f:
                imgs = f.readlines()
                imgs = [line.split('\n')[0] for line in imgs]
            views = self.readview(self.datapath+peoplename+"/mvs_views.txt")#'/data3/MVFR_DATASET/Res256/meta/6_4/three_views.txt'
            # lines
            for imgn in imgs :
                for view in views:
                    ref_view = view[0]
                    src_views = view[1:self.nviews]
                    metas.append((peoplename,imgn, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, peoplename,imgn,vid):
        extrinsics = copy.deepcopy(self.camparam[peoplename]["camparam"][imgn][vid]["Rt"])
        extrinsics=np.array(extrinsics).reshape([4,4])
        intrinsics= copy.deepcopy(self.camparam[peoplename]["camparam"][imgn][vid]["K"])
        intrinsics=np.array(intrinsics).reshape([3,3])
        systermerror = float(copy.deepcopy(self.camparam[peoplename]["systermerrors"][int(vid)]))
        return intrinsics, extrinsics,systermerror

    def read_img(self, filename, color_mode=None):
        if color_mode == "BGR":
            img = cv2.imread(filename)
        elif color_mode == "RGB" or color_mode is None:
            img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        return np_img

    def norm_img(self, np_img, self_norm=False, img_mean=None, img_std=None):
        if self_norm:
            var = np.var(np_img, axis=(0, 1), keepdims=True)
            mean = np.mean(np_img, axis=(0, 1), keepdims=True)
            np_img = (np_img - mean) / (np.sqrt(var) + 1e-7)
            return np_img
        else:
            # scale 0~255 to 0~1
            np_img = np_img / 255.
            if (img_mean is not None) and (img_std is not None):
                # scale with given mean and std
                img_mean = np.array(img_mean, dtype=np.float32)
                img_std = np.array(img_std, dtype=np.float32)
                np_img = (np_img - img_mean) / img_std
        return np_img

    def read_depth(self, filename,syserror):
        img = cv2.imread(filename,-1)
        depth = np.array(img, dtype=np.float32)-syserror
        return depth

    def scale_img(self, img, max_h=None, max_w=None, scale=None, interpolation=cv2.INTER_LINEAR): 
        h, w = img.shape[:2]
        if scale:
            new_w, new_h = int(scale * w), int(scale * h)
            img = cv2.resize(img, [new_w, new_h], interpolation=interpolation)
        elif h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = int(scale * w), int(scale * h)
            img = cv2.resize(img, [new_w, new_h], interpolation=interpolation)
        return img
    
    def crop_img(self, img, new_h=None, new_w=None, base=8):
        h, w = img.shape[:2]

        if new_h is None or new_w is None:
            new_h = h // base * base
            new_w = w // base * base

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            img = img[start_h:finish_h, start_w:finish_w]
        return img

    def crop_img_any(self, img, start_h, start_w, new_h, new_w):
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        img = img[start_h:finish_h, start_w:finish_w]
        return img

    def crop_cam_any(self, intrinsics, start_h, start_w):
        new_intrinsics = np.copy(intrinsics)
        # principle point:
        new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
        new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
        return new_intrinsics
    
    def scale_cam(self, intrinsics, h=None, w=None, max_h=None, max_w=None, scale=None):
        if scale:
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0, :] *= scale
            new_intrinsics[1, :] *= scale
        elif h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0, :] *= scale
            new_intrinsics[1, :] *= scale
        return new_intrinsics
    
    def crop_cam(self, intrinsics, h, w, new_h=None, new_w=None, base=8):
        if new_h is None or new_w is None:
            new_h = h // base * base
            new_w = w // base * base

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
            new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
            return new_intrinsics
        else:
            return intrinsics
    
    def __getitem__(self, idx):
        meta = self.metas[idx]
        peoplename,imgn, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        stage_num = len(self.stage_info["scale"])
        proj_matrices = {str(i):[] for i in range(stage_num)}
        cams = {str(i):[] for i in range(stage_num)}
        ref_imgs = {str(i):None for i in range(stage_num)}
        ref_cams = {str(i):None for i in range(stage_num)}
        depths = {str(i):None for i in range(stage_num)}
        masks = {str(i):None for i in range(stage_num)}

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,peoplename+'/'+vid+"/image/"+imgn+'.jpg')
            mask_filename = os.path.join(self.datapath,peoplename+'/'+vid+"/dis_mask/"+imgn+'_mask.png')
            depth_filename = os.path.join(self.datapath,peoplename+'/'+vid+"/depth/"+imgn+'.png')
            smpldepth_filename = os.path.join(self.datapath,peoplename+'/'+vid+"/smpl_depth/"+imgn+'.png')
            img = self.read_img(img_filename, color_mode=self.color_mode)             
            intrinsics, extrinsics ,syserror= self.read_cam_file(peoplename,imgn,vid)

            
            # begin stages
            for stage_id in range(stage_num):
                stage_scale = self.stage_info["scale"][str(stage_id)]
                stage_intrinsics = self.scale_cam(intrinsics=intrinsics, scale=stage_scale)
                stage_proj_mat = extrinsics.copy()
                stage_proj_mat[:3, :4] = np.matmul(stage_intrinsics, stage_proj_mat[:3, :4])
                proj_matrices[str(stage_id)].append(stage_proj_mat)

                stage_cam = np.zeros([2, 4, 4], dtype=np.float32)
                stage_cam[0, :4, :4] = extrinsics
                stage_cam[1, :3, :3] = stage_intrinsics
                cams[str(stage_id)].append(stage_cam)
            
                if i == 0:  # reference view
                    stage_ref_img = img.copy()
                    stage_ref_img = self.scale_img(stage_ref_img, scale=stage_scale, interpolation=self.img_interp)
                    stage_ref_img = np.array(stage_ref_img, dtype=np.uint8)
                    ref_imgs[str(stage_id)] = stage_ref_img
                    ref_cams[str(stage_id)] = stage_cam
            
            img = self.norm_img(img, self_norm=self.self_norm, img_mean=self.img_mean, img_std=self.img_std)
            imgs.append(img)

            if i == 0:  # reference view
                mask = self.read_img(mask_filename)
                mask = self.norm_img(mask)
                mask[mask > 0.0] = 1.0
                depth = self.read_depth(depth_filename,syserror)
                depthsmpl = self.read_depth(smpldepth_filename,0)
                depthsmpl = depthsmpl[depthsmpl!=0]
                try:
                    depth_min_max = np.array([min(depthsmpl)-200,max(depthsmpl)+200], dtype=np.float32)
                except ValueError:
                    print(smpldepth_filename)
                    depth_min_max = np.array([1200,2500], dtype=np.float32)
                depth_range = depth_min_max[1] - depth_min_max[0]

                # begin stages
                for stage_id in range(stage_num):
                    stage_scale = self.stage_info["scale"][str(stage_id)]
                    stage_mask = self.scale_img(img=mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                    stage_depth = self.scale_img(img=depth, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                    masks[str(stage_id)] = stage_mask
                    depths[str(stage_id)] = stage_depth

        binary_tree =  np.zeros([2, depths[str(0)].shape[0], depths[str(0)].shape[1]], dtype=np.int64)
        binary_tree[0, :, :] = binary_tree[0, :, :] + 1

            # binary_tree[0] is level, binary_tree[1] is key
        depth_min = depth_min_max[0]
        sample_interval = depth_range / self.depth_num
        sample_depth = []
        for i in range(self.depth_num):
            sample_depth.append(np.ones_like(depths[str(0)]) * (sample_interval * (i + i + 1) / 2.0 + depth_min))

        sample_depth = np.stack(sample_depth, axis=0)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = {str(j):np.stack(proj_matrices[str(j)], axis=0) for j in range(stage_num)}
        cams = {str(j):np.stack(cams[str(j)], axis=0) for j in range(stage_num)}

        return {"scan_name": peoplename+"_"+imgn,
                "img_id": int(ref_view),
                "ref_imgs": ref_imgs,
                "ref_cams": ref_cams,
                "imgs": imgs,
                "proj_matrices": proj_matrices,
                "cams": cams,
                "depths": depths,
                #"depth_values": depth_values,
                "depth_min_max": depth_min_max,
                "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                "masks": masks}

 
