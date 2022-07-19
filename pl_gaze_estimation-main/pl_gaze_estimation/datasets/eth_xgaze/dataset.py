import json
import pathlib
from typing import Callable, Optional, Tuple
import os
import glob
from PIL import Image
import cv2

import h5py
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig

from ...pl_utils.dataset import Dataset as PlDataset
from ...utils import str2path
from .transforms import create_transform

def get_saliency_from_gaze(e, g, R, K_inv, depth_rect, k):
    # Get 3D points from depth map
    #X = d(x)K_inv @ x
    X,Y = np.mgrid[0:942, 0:489]
    xy = np.vstack((X.flatten(order='C'), Y.flatten(order='C'))).T
    z = np.reshape(np.ones(489*942), ((489*942), 1))
    xyz = np.hstack((xy, z))


    xyz_3D_flat = np.dot(K_inv,xyz.T).T

    xyz_3D = np.reshape(xyz_3D_flat, (942,489,3), order='C')
    xyz_3D = np.transpose(xyz_3D,(1,0,2))

    depth_rect_mult = np.reshape(depth_rect, (489,942,1))
    xyz_3D = np.transpose(xyz_3D, (2, 0 , 1))
    depth_rect_mult = np.transpose(depth_rect_mult, (2, 0 , 1))


    xyz_3D = np.multiply(depth_rect_mult, xyz_3D)
    xyz_3D = np.transpose(xyz_3D, (1, 2, 0))
    X = xyz_3D



    X = np.reshape(X, ((489*942), 3))
    X = X - e
    X_norm = np.linalg.norm(X, axis=1)

    X = np.divide(X.T, np.reshape(X_norm,(1,(489*942))))
    X = X.T
    X = (R @ X.T).T

    s_X = X

    # Get saliency value from 3D s vectors and gaze direction g
    s_g = np.exp(k * np.power((np.transpose(g) @ np.transpose(s_X)), 4))
    sum = np.sum(s_g)
    s_g = s_g / sum
    s_g = s_g.reshape((489,942))

    #REMOVE
    #s_g = s_g[5:485,151:791]
    s_g = s_g[5:485,71:-71]
    sum = np.sum(s_g)
    s_g = s_g / sum

    return s_g

def get_saliency_from_gaze_torch(e, g, R, K_inv, depth_rect, k):
    #convert pitchyaw to 3d vector
    g = g.unsqueeze(0)
    n = g.size()[0]

    sin = torch.sin(g).cuda()
    cos = torch.cos(g).cuda()
    out = torch.empty((n, 3)).cuda()
    out[:, 0] = torch.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.multiply(cos[:, 0], cos[:, 1])

    out = -1 * out[0]
    g_torch = out


    #torch
    x = torch.arange(942).cuda()
    y = torch.arange(489).cuda()
    X_torch, Y_torch = torch.meshgrid(x, y)


    #torch
    xy_torch = torch.vstack((X_torch.flatten(), Y_torch.flatten())).T
    z_torch = torch.reshape(torch.ones(489*942), ((489*942), 1)).cuda()
    xyz_torch = torch.hstack((xy_torch, z_torch))


    #torch
    xyz_3D_flat_torch = torch.mm(K_inv,xyz_torch.double().T).T
    xyz_3D_torch = torch.reshape(xyz_3D_flat_torch, (942,489,3))
    xyz_3D_torch = torch.transpose(xyz_3D_torch,0,1)


    #torch
    depth_rect_mult_torch = torch.reshape(depth_rect, (489,942,1))
    xyz_3D_torch = torch.transpose(xyz_3D_torch, 0, 2)
    xyz_3D_torch = torch.transpose(xyz_3D_torch, 1, 2)
    depth_rect_mult_torch = torch.transpose(depth_rect_mult_torch, 0, 2)
    depth_rect_mult_torch = torch.transpose(depth_rect_mult_torch, 1, 2)


    #torch
    xyz_3D_torch = torch.multiply(depth_rect_mult_torch, xyz_3D_torch)
    xyz_3D_torch = torch.transpose(xyz_3D_torch, 0, 2)
    xyz_3D_torch = torch.transpose(xyz_3D_torch, 0, 1)
    X = xyz_3D_torch

    X = torch.reshape(X, ((489*942), 3))
    X = X - e
    X_norm = torch.linalg.norm(X, dim=1)

    X = torch.divide(X.T, torch.reshape(X_norm,(1,(489*942))))
    X = X.T
    X = (torch.mm(R, X.T)).T

    s_X = X

    # Get saliency value from 3D s vectors and gaze direction g

    s_g = torch.exp(k * torch.pow(torch.mm(torch.unsqueeze(torch.t(g_torch), 0).float(), torch.t(s_X).float()), 4))

    s_g = s_g.double()
    s_g = s_g[0]


    sum = torch.sum(s_g)

    s_g = s_g / sum
    s_g = s_g.reshape((489,942))

    s_g = s_g[5:485,71:-71]
    sum = torch.sum(s_g)
    s_g = s_g / sum
    return s_g

class OnePersonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: pathlib.Path, transform: Callable):
        self.dataset_path = dataset_path
        self.transform = transform
        self.random_horizontal_flip = False
        self._length = self._get_length()
        self.id_list = []
        file_list = glob.glob(str(dataset_path) + "*")
        for num_path in file_list:
            user_name = num_path.split("/")[-1]
            for num_path in glob.glob(num_path + "/face_ims/*.png"):
                self.id_list.append([num_path[-17:-9], num_path[:-26]])

    def _get_length(self) -> int:
        path, dirs, files = next(os.walk(str(self.dataset_path) + '/face_ims/'))
        length = len(files)
        return length

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        id_num, path_name = self.id_list[index]

        H1 = np.load('calib/H1.npy')
        H2 = np.load('calib/H2.npy')
        k1 = np.load('calib/k.npy')
        k2 = np.load('calib/k2.npy')
        r_real = np.load('calib/r.npy')
        t_real = np.load('calib/t.npy')
        b = np.load('calib/b.npy')
        transform_in = np.load("calib/transform_in.npy")
        transform_out = np.load("calib/transform_out.npy")
        K = np.load("calib/k.npy")

        label_path = path_name + '/gaze_info/' + id_num + '_gaze.txt'


        with open(label_path, 'r') as infile:
            label_gaze = [[x.replace("[", "").replace("]", "").replace(":", "").replace(",", "") for x in line.split()] for line in infile]
            label_gaze = [list(filter(None, label_gaze_x)) for label_gaze_x in label_gaze]
            if len(label_gaze) == 5:
                label = np.asarray(label_gaze[2][1:], dtype=np.float64, order='C')
                eye_loc_2d = np.asarray(label_gaze[3][1:], dtype=np.float64, order='C')
                eye_loc_3d = np.asarray(label_gaze[4][1:], dtype=np.float64, order='C')
            else:
                #label_1 = np.array(map(float, label_gaze[2][1:]))
                label_1 = np.asarray(label_gaze[2][1:], dtype=np.float64, order='C')
                label_2 = np.asarray(label_gaze[5][1:], dtype=np.float64, order='C')
                label = (label_1 + label_2) / 2

                eye_loc_1_2d = np.asarray(label_gaze[3][1:], dtype=np.float64, order='C')
                eye_loc_2_2d = np.asarray(label_gaze[6][1:], dtype=np.float64, order='C')
                eye_loc_2d = (eye_loc_1_2d + eye_loc_2_2d) / 2

                eye_loc_1_3d = np.asarray(label_gaze[4][1:], dtype=np.float64, order='C')
                eye_loc_2_3d = np.asarray(label_gaze[7][1:], dtype=np.float64, order='C')
                eye_loc_3d = (eye_loc_1_3d + eye_loc_2_3d) / 2

        gaze_for_sal = label
        #gaze = np.asarray([np.arctan(label[0]/(-1 * label[1])) , np.arctan(np.sqrt(np.square(label[0]) + np.square(label[1]))/label[2])])
        label = label * -1
        vectors = np.asarray([label])
        n = vectors.shape[0]
        out = np.empty((n, 2))
        vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
        out[:, 0] = np.arcsin(vectors[:, 1])  # theta
        out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
        gaze = out[0]
        # print("label_" + id_num, label)
        # print("out_" + id_num, out)


        #pose = np.zeros(2)
        pose = np.random.rand(2)


        face_path = path_name + '/face_ims/' + id_num + '_face.png'

        sal_mode = True
        gaze_mode = True

        if gaze_mode:
            image = np.array(Image.open(face_path))
            crop_x = [int(eye_loc_2d[1])-100, int(eye_loc_2d[1])+144]
            crop_y = [int(eye_loc_2d[0])-122, int(eye_loc_2d[0])+122]
            if int(eye_loc_2d[1])-100 < 0:
                crop_x[0] = 0
                crop_x[1] = 244
            if int(eye_loc_2d[0])-122 < 0:
                crop_y[0] = 0
                crop_y[1] = 244
            if int(eye_loc_2d[1])+144 >= 800:
                crop_x[0] = 556
                crop_x[1] = 800
            if int(eye_loc_2d[0])+122 >= 800:
                crop_y[0] = 556
                crop_y[1] = 800

            image = image[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]
        else:
            image = [[0]]
            crop_x = [0]
            crop_y = [0]


        if sal_mode:
            #Saliency from gaze
            depth_rect_path = path_name + '/scene_depth/' + id_num + '_depth.npy'
            depth_rect = np.load(depth_rect_path)
            e = eye_loc_3d
            g = gaze_for_sal
            i_T_o = np.linalg.inv(transform_in) @ transform_out
            R = i_T_o[:3,:3]
            R = R @ r_real
            K_inv = np.linalg.inv(K)

            k = 30

            s_g = get_saliency_from_gaze(e, g, R, K_inv, depth_rect, k)

            scene_path = path_name + '/scene_ims/' + id_num + '_scene.png'
            scene_image = np.array(Image.open(scene_path))

            scene_image = scene_image[5:485,71:-71]

        else:
            s_g = [[0]]
            scene_image = [[0]]
            depth_rect = [[0]]

        if self.random_horizontal_flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
            scene_image = scene_image[:, ::-1]
            s_g = s_g[:, ::-1] - np.zeros_like(s_g)
            pose *= np.array([1, -1])
            gaze *= np.array([1, -1])

        if gaze_mode:
            image = self.transform(image)
        if sal_mode:
            scene_image = self.transform(scene_image)

        pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        path = [face_path, crop_x, crop_y]


        return image, path, gaze, s_g, scene_image, depth_rect, eye_loc_3d

    def __len__(self) -> int:
        return self._length


class Dataset(PlDataset):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_root_dir = str2path(self.config.DATASET.DATASET_ROOT_DIR)
        assert dataset_root_dir.exists()

        # if self.config.DATASET.SUPERVISED:
        #     split_file = dataset_root_dir / 'train_test_split_supervised.json'
        # else:
        #     split_file = dataset_root_dir / 'train_test_split.json'
        # if self.config.DATASET.TEST:
        #     split_file = dataset_root_dir / 'train_test_split_test_data.json'
        # else:
        #     split_file = dataset_root_dir / 'train_test_split.json'
        # if self.config.DATASET.TEST == 'supervised':
        #     split_file = dataset_root_dir / 'train_test_split_supervised.json'
        # elif self.config.DATASET.TEST == 'test1':
        #     split_file = dataset_root_dir / 'train_test_split_test1.json'
        # elif self.config.DATASET.TEST == 'test2':
        #     split_file = dataset_root_dir / 'train_test_split_test2.json'
        # else:
        #     split_file = dataset_root_dir / 'train_test_split.json'

        # path_split = 'train_test_split.json'
        # #change back for self -> #
        path_split = 'train_test_split_' + self.config.DATASET.TEST + '.json'

        split_file = dataset_root_dir / path_split

        #split_file = dataset_root_dir / 'train_test_split.json'


        with open(split_file) as f:
            split = json.load(f)
        train_paths = [
            dataset_root_dir / 'train' / name for name in split['train']
        ]
        for path in train_paths:
            assert path.exists()

        if stage is None or stage == 'fit':
            train_transform = create_transform(self.config, 'train')
            if (self.config.VAL.VAL_RATIO > 0
                    and self.config.VAL.VAL_INDICES is not None):
                raise ValueError
            elif self.config.VAL.VAL_RATIO > 0:
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, train_transform)
                    for path in train_paths
                ])
                val_ratio = self.config.VAL.VAL_RATIO
                assert val_ratio < 1
                val_num = int(len(train_dataset) * val_ratio)
                train_num = len(train_dataset) - val_num
                lengths = [train_num, val_num]
                (self.train_dataset,
                 self.val_dataset) = torch.utils.data.dataset.random_split(
                     train_dataset, lengths)
                val_transform = create_transform(self.config, 'val')
                self.val_dataset.transform = val_transform
            elif self.config.VAL.VAL_INDICES is not None:
                val_indices = set(self.config.VAL.VAL_INDICES)
                assert 0 < len(val_indices) < 80
                for index in val_indices:
                    assert 0 <= index < 80

                self.train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, train_transform)
                    for i, path in enumerate(train_paths)
                    if i not in val_indices
                ])
                print(val_indices)
                print(train_paths)
                val_transform = create_transform(self.config, 'val')
                self.val_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, val_transform)
                    for i, path in enumerate(train_paths) if i in val_indices
                ])
            else:
                self.train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, train_transform)
                    for path in train_paths
                ])
            if self.config.DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP:
                for dataset in self.train_dataset.datasets:
                    dataset.random_horizontal_flip = True
