#!/usr/bin/env python

import logging

import coloredlogs
import torch
import torch.nn.functional as F
import functools
from pytorch_lightning.utilities.seed import seed_everything

from pl_gaze_estimation.models.unisal.model_unisal import UNISAL
from pl_gaze_estimation.models.unisal.model import Model
from pl_gaze_estimation.datasets.eth_xgaze.dataset import get_saliency_from_gaze_torch

from pl_gaze_estimation.config import parse_args
from pl_gaze_estimation.datasets import create_dataset
from pl_gaze_estimation.models import create_model
from pl_gaze_estimation.pl_utils import get_trainer

from pl_gaze_estimation.models.unisal import utils
from pl_gaze_estimation.models.utils.gaze import compute_angle_error
from pl_gaze_estimation.models.eth_xgaze.model import draw_gaze

import importlib
import shutil

import torch
from torch import optim
import numpy as np
import os
import cv2
from omegaconf import DictConfig
import copy
import omegaconf

coloredlogs.install(level='DEBUG',
                    logger=logging.getLogger('pl_gaze_estimation'))

def get_model(config) -> torch.nn.Module:
    module = importlib.import_module(
        f'pl_gaze_estimation.models.unisal.model')
    model = getattr(module, 'Model')(config)
    return model



def main():
    #mode = 'gaze'
    #mode = 'sal'
    mode = 'self'

    config = parse_args()


    if mode == 'self':
        num_epochs = 10
        batch_size = config.TRAIN.BATCH_SIZE  # To actually change batch size do in eth yaml
        load_pretrained = True

        #Creat self supervised dataloader
        seed_everything(seed=config.EXPERIMENT.SEED, workers=True)
        dataset = create_dataset(config)
        dataset.setup("fit")
        train_dataloader = dataset.train_dataloader()
        val_dataloader = dataset.val_dataloader()




        #create supervised dataloader
        config_sup = copy.deepcopy(config)
        with omegaconf.read_write(config_sup):
            config_sup.DATASET.SUPERVISED = True
        dataset_sup = create_dataset(config_sup)
        dataset_sup.setup("fit")

        train_dataloader_sup = dataset_sup.train_dataloader()
        data_iter_sup = iter(train_dataloader_sup)

        #print(next(iter(train_dataloader_sup)))


        model_gaze = create_model(config)
        model_sal = get_model(config)

        #Load pretrained models
        if load_pretrained:
            # chkpnt = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/unisal-master/training_runs/2022-02-15_02:21:15_unisal/chkpnt_epoch0003.pth")
            # model.load_state_dict(chkpnt['model_state_dict'])
            # model = model_gaze.load_from_checkpoint(ckpt_file_path)
            # pass
            # model.load_state_dict(torch.load(PATH))
            # model.load_state_dict(torch.load(PATH))

            # # ------------------ 20/60/20
            # #model_sal = model_sal.load_from_checkpoint("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            # model_sal.load_state_dict(checkpoint["state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0009/epoch=0012.ckpt")
            # model_gaze.load_state_dict(checkpoint["state_dict"])

            # # ------------------ 60/20/20
            # #model_sal = model_sal.load_from_checkpoint("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0285/epoch=0002.ckpt")
            # model_sal.load_state_dict(checkpoint["state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0289/epoch=0012.ckpt")
            # model_gaze.load_state_dict(checkpoint["state_dict"])

            # # ------------------ 5/75/20
            # #model_sal = model_sal.load_from_checkpoint("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0302/epoch=0004.ckpt")
            model_sal.load_state_dict(checkpoint["state_dict"])
            checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0301/epoch=0004.ckpt")
            model_gaze.load_state_dict(checkpoint["state_dict"])

            #new split test
            # #model_sal = model_sal.load_from_checkpoint("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0503/epoch=0007.ckpt")
            # model_sal.load_state_dict(checkpoint["state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0502/epoch=0014.ckpt")
            # model_gaze.load_state_dict(checkpoint["state_dict"])

        model_gaze = model_gaze.cuda()
        model_sal = model_sal.cuda()



        #define optimizer
        learning_rate = 0.000000005
        #learning_rate = 0.0000005
        optimizer = optim.Adam(list(model_gaze.parameters()) + list(model_sal.parameters()), lr = learning_rate)

        #load calibration info
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

        i_T_o = np.linalg.inv(transform_in) @ transform_out
        R = i_T_o[:3,:3]
        R = R @ r_real
        K_inv = np.linalg.inv(K)

        K_inv = torch.tensor(K_inv).cuda()
        R = torch.tensor(R).cuda()


        #get length of datasets
        total_train = len(train_dataloader.dataset)
        batch_total = total_train / batch_size

        total_val = len(val_dataloader.dataset)
        batch_total_val = total_val / batch_size

        #start training
        for epoch in range(num_epochs):
            print("EPOCH: " + str(epoch))
            batch_num = 0
            running_loss = 0
            batch_loss_50 = 0
            angle_50 = 0
            angle_500 = 0
            running_angle_error = 0
            MAE_500 = 0
            MAE_50 = 0

            #change models to train mode
            model_sal.train()
            model_gaze.train()

            #iterate through dataloader
            for (batch_idx, batch) in enumerate(train_dataloader):
                #separate and cuda
                image = batch[0]
                paths = batch[1]
                gaze = batch[2]
                sal_from_gt_gazes = batch[3]
                scene_image = batch[4]
                depth = batch[5]
                eye_loc_3d = batch[6]
                scene_image = torch.unsqueeze(scene_image, 1).cuda()
                image = image.cuda()
                gaze = gaze.cuda()
                sal_from_gt_gazes = sal_from_gt_gazes.cuda()
                depth = depth.cuda()
                eye_loc_3d = eye_loc_3d.cuda()


                # ================= FORWARD=================
                out_gaze = model_gaze(image)
                out_sal = model_sal(scene_image)

                out_gaze = out_gaze.float()
                out_sal = out_sal.float()


                #Compute angle error
                #(Note: this error uses the GT gaze only to validate training is working and does not use for loss)
                angle_error = compute_angle_error(out_gaze, gaze)
                angle_error = torch.mean(angle_error)

                #compute loss
                #compute saliency from predicted gaze s_g
                k = torch.tensor(30.0).cuda()
                s_g = torch.empty(0).cuda()
                for i in range(0,out_gaze.size()[0]):
                    s_g = torch.cat((s_g ,torch.unsqueeze(get_saliency_from_gaze_torch(eye_loc_3d[i], out_gaze[i], R, K_inv, depth[i], k).float(), 0)))

                s_g = torch.unsqueeze(s_g, 1)
                s_g = torch.unsqueeze(s_g, 1)

                #MAE metric
                sal_from_gt_gazes_s = torch.unsqueeze(sal_from_gt_gazes, 1)
                sal_from_gt_gazes_s = torch.unsqueeze(sal_from_gt_gazes_s, 1)
                out_sal_norm = out_sal.exp()
                diff_sal = out_sal_norm - sal_from_gt_gazes_s

                diff_sal = torch.squeeze(diff_sal, 1)
                diff_sal = torch.squeeze(diff_sal, 1)

                batch_avg_sal_norm = torch.mean(torch.norm(torch.norm(diff_sal, dim=1), dim=1))

                MAE_sal = torch.sum(torch.sum(torch.abs(diff_sal), dim=1), dim=1)
                #MAE_sal_all = torch.cat((MAE_sal_all, MAE_sal), dim=0)

                #compute loss
                losses = []

                loss_kld = utils.kld_loss(out_sal, s_g)
                loss_kld = loss_kld.float()
                losses.append(loss_kld)

                #cc loss
                loss_cc = utils.corr_coeff(out_sal.exp(), s_g)
                loss_cc = loss_cc.float()
                losses.append(loss_cc)

                #compute total loss with weights
                #loss_summands = self.loss_sequences(outputs, sals, fix, metrics=('kld','cc'))
                loss_weights = (1, -0.1)
                loss_summands = [l.mean(1).mean(0) for l in losses]
                loss = sum(weight * l for weight, l in
                                  zip(loss_weights, loss_summands))

                # ================= Supervised/Constrained Loss =================
                # loop through supervised dataloader and create again if at the end
                try:
                    image_sup, paths_sup, gaze_sup, sal_from_gt_gazes_sup, scene_image_sup, depth_sup, eye_loc_3d_sup = next(data_iter_sup)
                except StopIteration:
                    data_iter_sup = iter(train_dataloader_sup)
                    image_sup, paths_sup, gaze_sup, sal_from_gt_gazes_sup, scene_image_sup, depth_sup, eye_loc_3d_sup = next(data_iter_sup)
                scene_image_sup = torch.unsqueeze(scene_image_sup, 1).cuda()
                image_sup = image_sup.cuda()
                gaze_sup = gaze_sup.cuda()
                sal_from_gt_gazes_sup = sal_from_gt_gazes_sup.cuda().float()
                sal_from_gt_gazes_sup = torch.unsqueeze(sal_from_gt_gazes_sup, 1)
                sal_from_gt_gazes_sup = torch.unsqueeze(sal_from_gt_gazes_sup, 1)


                out_gaze_sup = model_gaze(image_sup)
                out_sal_sup = model_sal(scene_image_sup)

                out_gaze_sup = out_gaze_sup.float()
                out_sal_sup = out_sal_sup.float()
                gaze_sup = gaze_sup.float()


                # compute gaze loss
                gaze_loss_fn = functools.partial(F.l1_loss, reduction='mean')
                gaze_loss = gaze_loss_fn(out_gaze, gaze_sup)

                # compute sal loss
                #compute loss
                losses_sup = []

                loss_kld_sup = utils.kld_loss(out_sal_sup, sal_from_gt_gazes_sup)
                loss_kld_sup = loss_kld_sup.float()
                losses_sup.append(loss_kld_sup)

                #cc loss
                loss_cc_sup = utils.corr_coeff(out_sal_sup.exp(), sal_from_gt_gazes_sup)
                loss_cc_sup = loss_cc_sup.float()
                losses_sup.append(loss_cc_sup)

                #compute total loss with weights
                #loss_summands = self.loss_sequences(outputs, sals, fix, metrics=('kld','cc'))
                loss_weights_sup = (1, -0.1)
                loss_summands_sup = [l.mean(1).mean(0) for l in losses_sup]
                loss_sup = sum(weight * l for weight, l in
                                  zip(loss_weights_sup, loss_summands_sup))


                # Sum Losses
                loss = loss + (2.0 * gaze_loss) + (2.0 * loss_sup)

                # ================= Track Losses =================


                running_loss += loss.item()
                batch_loss_50 += loss.item()
                running_angle_error += angle_error.item()
                angle_50 += angle_error.item()
                MAE_50 += torch.mean(MAE_sal).item()
                if batch_num % 50 == 0:
                    print("batch num: " + str(batch_num) + "/" + str(int(batch_total)) + " loss: " + str(loss.cpu().detach().numpy()) + " angle_error: " + str(angle_error.cpu().detach().numpy()) + " Loss_50: " + str(batch_loss_50 / 50) +  " Angle_50: " + str(angle_50 / 50) +  " MAE_50: " + str(MAE_50 / 50))
                    batch_loss_50 = 0
                    angle_50 = 0
                    MAE_50 = 0
                # ================= BACKWARD =================

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # ================= Visualize Train =================
                if batch_num % 100 == 0:
                    outputs = out_sal.detach().cpu().numpy()
                    sals = s_g.detach().cpu().numpy()
                    sal_from_gt_gazes = sal_from_gt_gazes.detach().cpu().numpy()
                    path = os.getcwd()
                    dirpath = os.path.join(path , 'visualizations_self_train/epoch_' + str(epoch) + '_batch_' + str(batch_num))#str(self.current_epoch))
                    if not os.path.exists(dirpath):
                        os.mkdir(os.path.join(path , 'visualizations_self_train/epoch_' + str(epoch) + '_batch_' + str(batch_num)))#str(self.current_epoch)))
                    for i in range(0,len(paths[0])):
                        name = str(paths[0][i])
                        index = -27
                        name = name[0 : index : ] + name[index + 1 : :]

                        name = name[:-26] + 'scene_ims/' + name[-17:-9] + '_scene.png'
                        im = cv2.imread(name)
                        im = im[5:485,71:-71]

                        pred = np.exp(outputs[i])
                        pred = (pred / np.amax(pred) * 255).astype(np.uint8)
                        pred = pred[0][0]

                        sal = sals[i][0][0]
                        sal = (sal / np.max(sal)) * 255

                        sal_from_gt_gaze = sal_from_gt_gazes[i]
                        sal_from_gt_gaze = (sal_from_gt_gaze / np.max(sal_from_gt_gaze)) * 255

                        sal = cv2.cvtColor(sal.astype('float32'),cv2.COLOR_GRAY2RGB)
                        pred = cv2.cvtColor(pred.astype('float32'),cv2.COLOR_GRAY2RGB)
                        sal_from_gt_gaze = cv2.cvtColor(sal_from_gt_gaze.astype('float32'),cv2.COLOR_GRAY2RGB)

                        im = cv2.putText(im, 'Outward Scene', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        sal_from_gt_gaze = cv2.putText(sal_from_gt_gaze, 'Sal From GT', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        pred = cv2.putText(pred, 'Sal From Sal_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        sal = cv2.putText(sal, 'Sal From Gaze_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        im_out_top = np.concatenate((im, sal_from_gt_gaze), axis=1)
                        im_out_bottom = np.concatenate((pred, sal), axis=1)
                        im_out = np.concatenate((im_out_top, im_out_bottom), axis=0)

                        path = os.getcwd()
                        cv2.imwrite(os.path.join(path , 'visualizations_self_train/epoch_' + str(epoch) + '_batch_' + str(batch_num) + '/' + name[-17:-9] + '.png'), im_out)


                    outputs = out_gaze.detach().cpu().numpy()
                    gazes = gaze.detach().cpu().numpy()
                    angle_error_vals = compute_angle_error(out_gaze, gaze).detach().cpu().numpy()
                    crop_x = paths[1]
                    crop_y = paths[2]
                    crop_x_0 = crop_x[0].detach().cpu().numpy()
                    crop_x_1 = crop_x[1].detach().cpu().numpy()
                    crop_y_0 = crop_y[0].detach().cpu().numpy()
                    crop_y_1 = crop_y[1].detach().cpu().numpy()
                    path = os.getcwd()
                    dirpath = os.path.join(path , 'visualizations_self_train/epoch_' + str(epoch) + '_batch_' + str(batch_num))
                    if not os.path.exists(dirpath):
                        os.mkdir(os.path.join(path , 'visualizations_self_train/epoch_' + str(epoch) + '_batch_' + str(batch_num)))
                    for i in range(0,len(paths[0])):
                        name = str(paths[0][i])
                        index = -27
                        name = name[0 : index : ] + name[index + 1 : :]
                        im = cv2.imread(name)

                        im = im[crop_x_0[i]:crop_x_1[i], crop_y_0[i]:crop_y_1[i]]

                        pitchyaw = outputs[i]
                        im_out = draw_gaze(im, pitchyaw, thickness=2, color=(0, 0, 255))
                        im_out = cv2.putText(im_out, 'Predicted', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        pitchyaw = gazes[i]
                        im_out = draw_gaze(im_out, pitchyaw, thickness=2, color=(0, 255, 0))



                        vectors = np.asarray([[1, 0, 0]])
                        n = vectors.shape[0]
                        out = np.empty((n, 2))
                        vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
                        out[:, 0] = np.arcsin(vectors[:, 1])  # theta
                        out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
                        pitch_yaw = out[0]
                        im_out = draw_gaze(im_out, pitch_yaw, thickness=1, color=(255, 0, 0))

                        vectors = np.asarray([[0, 1, 0]])
                        n = vectors.shape[0]
                        out = np.empty((n, 2))
                        vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
                        out[:, 0] = np.arcsin(vectors[:, 1])  # theta
                        out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
                        pitch_yaw = out[0]
                        im_out = draw_gaze(im_out, pitch_yaw, thickness=1, color=(255, 0, 0))



                        im_out = cv2.putText(im_out, 'GT', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        im_out = cv2.putText(im_out, 'Err: ' + str(np.around(angle_error_vals[i], 2)), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        path = os.getcwd()
                        cv2.imwrite(os.path.join(path , 'visualizations_self_train/epoch_' + str(epoch) + '_batch_' + str(batch_num) + '/' + name[-17:-9] + '_gaze.png'), im_out)

                angle_500 += angle_error.item()
                MAE_500 += torch.mean(MAE_sal).item()
                if batch_num % 500 == 0 and batch_num != 0:
                    print("Checkpoint angle: ", str(angle_500 / 500))
                    print("Checkpoint MAE sal: ", str(MAE_500 / 500))
                    angle_500 = 0
                    MAE_500 = 0
                    print("Saving Checkpoint")
                    PATH = os.path.join(path , 'saved_checkpoints/gaze_epoch_' + str(epoch) + '_batch_' + str(batch_num) + '_lr_' + str(learning_rate) + '.pt')
                    LOSS = running_loss/batch_total
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_gaze.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSS,
                    }, PATH)

                    PATH = os.path.join(path , 'saved_checkpoints/sal_epoch_' + str(epoch)  + '_batch_' + str(batch_num) + '_lr_' + str(learning_rate) + '.pt')
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_sal.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSS,
                    }, PATH)

                batch_num += 1

            # ================= Validation Train / Visualization =================
            model_sal.eval()
            model_gaze.eval()
            print("VALIDATION")
            with torch.no_grad():
                batch_num_val = 0
                running_loss_val = 0
                running_angle_error_val = 0
                for (batch_idx, batch) in enumerate(val_dataloader):
                    #preprocess
                    image = batch[0]
                    paths = batch[1]
                    gaze = batch[2]
                    sal_from_gt_gazes = batch[3]
                    scene_image = batch[4]
                    depth = batch[5]
                    eye_loc_3d = batch[6]
                    scene_image = torch.unsqueeze(scene_image, 1).cuda()
                    image = image.cuda()
                    gaze = gaze.cuda()
                    sal_from_gt_gazes = sal_from_gt_gazes.cuda()
                    depth = depth.cuda()
                    eye_loc_3d = eye_loc_3d.cuda()


                    # ================= FORWARD=================
                    out_gaze = model_gaze(image)
                    out_sal = model_sal(scene_image)

                    out_gaze = out_gaze.float()
                    out_sal = out_sal.float()

                    angle_error_val = compute_angle_error(out_gaze, gaze)
                    angle_error_val = torch.mean(angle_error_val)

                    #compute loss
                    #compute saliency from gaze
                    #get e, g, R, K_inv, depth_rect, k
                    k = torch.tensor(30.0).cuda()
                    s_g = torch.empty(0).cuda()
                    for i in range(0,out_gaze.size()[0]):
                        s_g = torch.cat((s_g ,torch.unsqueeze(get_saliency_from_gaze_torch(eye_loc_3d[i], out_gaze[i], R, K_inv, depth[i], k).float(), 0)))

                    s_g = torch.unsqueeze(s_g, 1)
                    s_g = torch.unsqueeze(s_g, 1)

                    #compute loss
                    losses = []

                    loss_kld = utils.kld_loss(out_sal, s_g)
                    loss_kld = loss_kld.float()
                    losses.append(loss_kld)

                    #cc loss
                    loss_cc = utils.corr_coeff(out_sal.exp(), s_g)
                    loss_cc = loss_cc.float()
                    losses.append(loss_cc)

                    #compute total loss with weights
                    #loss_summands = self.loss_sequences(outputs, sals, fix, metrics=('kld','cc'))
                    loss_weights = (1, -0.1)
                    loss_summands = [l.mean(1).mean(0) for l in losses]
                    loss = sum(weight * l for weight, l in
                                      zip(loss_weights, loss_summands))

                    running_loss_val += loss.item()
                    running_angle_error_val += angle_error_val.item()
                    if batch_num_val % 50 == 0:
                        print("batch num val: " + str(batch_num_val) + "/" + str(int(batch_total_val)) + " loss: " + str(loss.cpu().detach().numpy()) + " angle_error: " + str(angle_error_val.cpu().detach().numpy()))
                    # ================= BACKWARD =================

                    # ================= Visualize Val =================
                    if batch_num_val % 100 == 0:
                        outputs = out_sal.detach().cpu().numpy()
                        sals = s_g.detach().cpu().numpy()
                        sal_from_gt_gazes = sal_from_gt_gazes.detach().cpu().numpy()
                        path = os.getcwd()
                        dirpath = os.path.join(path , 'visualizations_self_val/epoch_' + str(epoch) + '_batch_' + str(batch_num_val))#str(self.current_epoch))
                        if not os.path.exists(dirpath):
                            os.mkdir(os.path.join(path , 'visualizations_self_val/epoch_' + str(epoch) + '_batch_' + str(batch_num_val)))#str(self.current_epoch)))
                        for i in range(0,len(paths[0])):
                            name = str(paths[0][i])
                            index = -27
                            name = name[0 : index : ] + name[index + 1 : :]

                            name = name[:-26] + 'scene_ims/' + name[-17:-9] + '_scene.png'
                            im = cv2.imread(name)
                            im = im[5:485,71:-71]

                            pred = np.exp(outputs[i])
                            pred = (pred / np.amax(pred) * 255).astype(np.uint8)
                            pred = pred[0][0]

                            sal = sals[i][0][0]
                            sal = (sal / np.max(sal)) * 255


                            sal_from_gt_gaze = sal_from_gt_gazes[i]
                            sal_from_gt_gaze = (sal_from_gt_gaze / np.max(sal_from_gt_gaze)) * 255

                            sal = cv2.cvtColor(sal.astype('float32'),cv2.COLOR_GRAY2RGB)
                            pred = cv2.cvtColor(pred.astype('float32'),cv2.COLOR_GRAY2RGB)
                            sal_from_gt_gaze = cv2.cvtColor(sal_from_gt_gaze.astype('float32'),cv2.COLOR_GRAY2RGB)

                            im = cv2.putText(im, 'Outward Scene', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            sal_from_gt_gaze = cv2.putText(sal_from_gt_gaze, 'Sal From GT', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            pred = cv2.putText(pred, 'Sal From Sal_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            sal = cv2.putText(sal, 'Sal From Gaze_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                            im_out_top = np.concatenate((im, sal_from_gt_gaze), axis=1)
                            im_out_bottom = np.concatenate((pred, sal), axis=1)
                            im_out = np.concatenate((im_out_top, im_out_bottom), axis=0)

                            path = os.getcwd()
                            cv2.imwrite(os.path.join(path , 'visualizations_self_val/epoch_' + str(epoch) + '_batch_' + str(batch_num_val) + '/' + name[-17:-9] + '.png'), im_out)

                        outputs = out_gaze.detach().cpu().numpy()
                        gazes = gaze.detach().cpu().numpy()
                        crop_x = paths[1]
                        crop_y = paths[2]
                        crop_x_0 = crop_x[0].detach().cpu().numpy()
                        crop_x_1 = crop_x[1].detach().cpu().numpy()
                        crop_y_0 = crop_y[0].detach().cpu().numpy()
                        crop_y_1 = crop_y[1].detach().cpu().numpy()
                        path = os.getcwd()
                        dirpath = os.path.join(path , 'visualizations_self_val/epoch_' + str(epoch) + '_batch_' + str(batch_num))
                        if not os.path.exists(dirpath):
                            os.mkdir(os.path.join(path , 'visualizations_self_val/epoch_' + str(epoch) + '_batch_' + str(batch_num)))
                        for i in range(0,len(paths[0])):
                            name = str(paths[0][i])
                            index = -27
                            name = name[0 : index : ] + name[index + 1 : :]
                            im = cv2.imread(name)

                            im = im[crop_x_0[i]:crop_x_1[i], crop_y_0[i]:crop_y_1[i]]

                            pitchyaw = outputs[i]
                            im_out = draw_gaze(im, pitchyaw, thickness=2, color=(0, 0, 255))
                            im_out = cv2.putText(im_out, 'Predicted', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            pitchyaw = gazes[i]
                            im_out = draw_gaze(im_out, pitchyaw, thickness=2, color=(0, 255, 0))
                            im_out = cv2.putText(im_out, 'GT', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            path = os.getcwd()
                            cv2.imwrite(os.path.join(path , 'visualizations_self_val/epoch_' + str(epoch) + '_batch_' + str(batch_num) + '/' + name[-17:-9] + '_gaze.png'), im_out)


                    batch_num_val += 1

            print("Train Loss:", running_loss/batch_total)
            print("Validation Loss:", running_loss_val/batch_total_val)
            print("Train Angle Error:", running_angle_error/batch_total)
            print("Validation Angle Error:", running_angle_error_val/batch_total_val)


            # ================= Save Checkpoint =================
            PATH = os.path.join(path , 'saved_checkpoints/gaze_epoch_' + str(epoch) + '.pt')
            LOSS = running_loss/batch_total
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_gaze.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

            PATH = os.path.join(path , 'saved_checkpoints/sal_epoch_' + str(epoch) + '.pt')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_sal.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)


    else:
        seed_everything(seed=config.EXPERIMENT.SEED, workers=True)
        dataset = create_dataset(config)

        if mode == 'gaze':
            model = create_model(config)

        elif mode == 'sal':
            model = get_model(config)

        print(type(model))



        trainer = get_trainer(config)
        trainer.fit(model, dataset)



    # if config.TEST.RUN_TEST:
    #     trainer.test(ckpt_path=None, verbose=False)
    #     trainer.test(model, dataloader_test)


if __name__ == '__main__':
    main()
