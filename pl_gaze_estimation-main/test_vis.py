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
import scipy.io

coloredlogs.install(level='DEBUG',
                    logger=logging.getLogger('pl_gaze_estimation'))

def get_model(config) -> torch.nn.Module:
    module = importlib.import_module(
        f'pl_gaze_estimation.models.unisal.model')
    model = getattr(module, 'Model')(config)
    return model

# class Optimizer():
#     def __init__(self, params, lr=0.00001): self.params,self.lr=list(params),lr
#
#     def step(self):
#         with torch.no_grad():
#             for p in self.params: p -= p.grad * lr
#
#     def zero_grad(self):
#         for p in self.params: p.grad.data.zero_()


def main():
    #mode = 'gaze'
    #mode = 'sal'
    mode = 'self'

    config = parse_args()

    #Can't optimize both models in pytorch lightning so using pytorch method
    if mode == 'self':
        batch_size = config.TRAIN.BATCH_SIZE  # To actually change batch size do in eth yaml
        load_pretrained = False
        load_trained = True
        load_baseline = False

        #Creat self supervised dataloader
        # seed_everything(seed=config.EXPERIMENT.SEED, workers=True)
        # dataset = create_dataset(config)
        # dataset.setup("fit")
        # test_dataloader = dataset.train_dataloader()
        #val_dataloader = dataset.val_dataloader()
        #test_dataloader = dataset.test_dataloader()
        config_test = copy.deepcopy(config)
        with omegaconf.read_write(config_test):
            config_test.DATASET.TEST = 'valid'
        dataset_test = create_dataset(config_test)
        dataset_test.setup("fit")
        test_dataloader = dataset_test.train_dataloader()




        #create supervised dataloader
        # config_test = copy.deepcopy(config)
        # with omegaconf.read_write(config_test):
        #     config_test.DATASET.TEST = True
        # dataset_test = create_dataset(config_test)
        # dataset_test.setup("fit")
        #
        # train_dataloader_sup = dataset_sup.train_dataloader()
        # data_iter_sup = iter(train_dataloader_sup)
        # test_dataloader = dataset_test.train_dataloader()

        #UPDATE TO USE OUR TRAINED MODELS #TODO
        model_gaze = create_model(config)
        model_sal = get_model(config)

        model_gaze_sup = create_model(config)
        model_sal_sup = get_model(config)

        # checkpoint = torch.load("/home/isaac/.ptgaze/models/eth-xgaze_resnet18.pth")
        # loaded_dict = checkpoint['model']
        # prefix = 'model.'
        # n_clip = len(prefix)
        # adapted_dict = {prefix + k: v for k, v in loaded_dict.items()}
        # model_gaze.load_state_dict(adapted_dict)

        if load_baseline:
            checkpoint = torch.load("/home/isaac/.ptgaze/models/eth-xgaze_resnet18.pth")
            loaded_dict = checkpoint['model']
            prefix = 'model.'
            n_clip = len(prefix)
            adapted_dict = {prefix + k: v for k, v in loaded_dict.items()}
            model_gaze_sup.load_state_dict(adapted_dict)

            # checkpoint = torch.load("/home/isaac/.ptgaze/models/eth-xgaze_resnet18.pth")
            # loaded_dict = checkpoint['model']
            # prefix = 'model.'
            # n_clip = len(prefix)
            # adapted_dict = {prefix + k: v for k, v in loaded_dict.items()}
            # model_gaze.load_state_dict(adapted_dict)

        if load_pretrained:
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            # model_sal_sup.load_state_dict(checkpoint["state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0009/epoch=0012.ckpt")
            # model_gaze_sup.load_state_dict(checkpoint["state_dict"])
            # ------------------ 20/60/20
            #model_sal = model_sal.load_from_checkpoint("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/sal_epoch=0006.ckpt")
            # model_sal_sup.load_state_dict(checkpoint["state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch=0012.ckpt")
            # model_gaze_sup.load_state_dict(checkpoint["state_dict"])
            #
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/epoch=0003.ckpt")
            # model_sal_sup.load_state_dict(checkpoint["state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/epoch=0010.ckpt")
            # model_gaze_sup.load_state_dict(checkpoint["state_dict"])

            checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/sal_epoch=0004.ckpt")
            model_sal_sup.load_state_dict(checkpoint["state_dict"])
            checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/gaze_epoch=0004.ckpt")
            model_gaze_sup.load_state_dict(checkpoint["state_dict"])


        if load_trained:
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/sal_epoch_0_batch_8500_lr_5e-09.pt")
            # #checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/sal_epoch_0.pt")
            # #gaze_epoch_0_batch_8500_lr_5e-09.pt
            # model_sal.load_state_dict(checkpoint["model_state_dict"])
            # #checkpoint = torch.load("/home/isaac/Documents/best_run/gaze_epoch_0.pt")
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch_0_batch_8500_lr_5e-09.pt")
            # #checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/gaze_epoch_0.pt")
            # #sal_epoch_0_batch_8500_lr_5e-09.pt
            # model_gaze.load_state_dict(checkpoint["model_state_dict"])
            #
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/60_20_20/sal_epoch_1.pt")
            # model_sal.load_state_dict(checkpoint["model_state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/60_20_20/gaze_epoch_1.pt")
            # model_gaze.load_state_dict(checkpoint["model_state_dict"])
            #
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/sal_epoch_0.pt")
            # model_sal.load_state_dict(checkpoint["model_state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/gaze_epoch_0.pt")
            # model_gaze.load_state_dict(checkpoint["model_state_dict"])

            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/sal_epoch_1_batch_1000_lr_5e-09.pt")
            # model_sal.load_state_dict(checkpoint["model_state_dict"])
            # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/gaze_epoch_1_batch_1000_lr_5e-09.pt")
            # model_gaze.load_state_dict(checkpoint["model_state_dict"])

            checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/sal_epoch_0_batch_8500_lr_5e-09.pt")
            model_sal.load_state_dict(checkpoint["model_state_dict"])
            checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch_1_batch_3500_lr_5e-09.pt")
            model_gaze.load_state_dict(checkpoint["model_state_dict"])



        model_gaze = model_gaze.cuda()
        model_sal = model_sal.cuda()

        model_gaze_sup = model_gaze_sup.cuda()
        model_sal_sup = model_sal_sup.cuda()



        #define optimizer
        #optimizer = optim.Adam(list(net.parameters()) + list(clas.parameters()), lr=lr)
        #optimizer = optim.Adam(list(model_gaze.parameters()) + list(model_sal.parameters()), lr = 0.00000001)

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

        total_test = len(test_dataloader.dataset)
        batch_total = total_test / batch_size

        # total_val = len(val_dataloader.dataset)
        # batch_total_val = total_val / batch_size

        #image, path, gaze, s_g, scene_image, depth, eye_loc_3d = dataiter.next()

        batch_num = 0
        running_loss = 0
        batch_loss_50 = 0
        angle_50 = 0
        running_angle_error = 0
        model_sal.eval()
        model_gaze.eval()
        model_sal_sup.eval()
        model_gaze_sup.eval()

        #for image, paths, gaze, s_g, scene_image, depth, eye_loc_3d in train_dataloader:
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_dataloader):
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

                out_gaze_sup = model_gaze_sup(image)
                out_sal_sup = model_sal_sup(scene_image)

                out_gaze_sup  = out_gaze_sup.float()
                out_sal_sup  = out_sal_sup.float()


                #Compute angle error
                angle_error = compute_angle_error(out_gaze, gaze)
                angle_error = torch.mean(angle_error)

                #compute loss
                #compute saliency from gaze
                #get e, g, R, K_inv, depth_rect, k
                k = torch.tensor(30.0).cuda()
                s_g_sup = torch.empty(0).cuda()
                for i in range(0,out_gaze_sup.size()[0]):
                    s_g_sup = torch.cat((s_g_sup ,torch.unsqueeze(get_saliency_from_gaze_torch(eye_loc_3d[i], out_gaze_sup[i], R, K_inv, depth[i], k).float(), 0)))

                s_g_sup = torch.unsqueeze(s_g_sup, 1)
                s_g_sup = torch.unsqueeze(s_g_sup, 1)


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

                # ================= Supervised/Constrained Loss =================
                # try:
                #     image_sup, paths_sup, gaze_sup, sal_from_gt_gazes_sup, scene_image_sup, depth_sup, eye_loc_3d_sup = next(data_iter_sup)
                # except StopIteration:
                #     data_iter_sup = iter(train_dataloader_sup)
                #     image_sup, paths_sup, gaze_sup, sal_from_gt_gazes_sup, scene_image_sup, depth_sup, eye_loc_3d_sup = next(data_iter_sup)
                # scene_image_sup = torch.unsqueeze(scene_image_sup, 1).cuda()
                # image_sup = image_sup.cuda()
                # gaze_sup = gaze_sup.cuda()
                # sal_from_gt_gazes_sup = sal_from_gt_gazes_sup.cuda().float()
                # sal_from_gt_gazes_sup = torch.unsqueeze(sal_from_gt_gazes_sup, 1)
                # sal_from_gt_gazes_sup = torch.unsqueeze(sal_from_gt_gazes_sup, 1)


                # out_gaze_sup = model_gaze(image_sup)
                # out_sal_sup = model_sal(scene_image_sup)
                #
                # out_gaze_sup = out_gaze_sup.float()
                # out_sal_sup = out_sal_sup.float()
                # gaze_sup = gaze_sup.float()


                # # compute gaze loss
                # gaze_loss_fn = functools.partial(F.l1_loss, reduction='mean')
                # gaze_loss = gaze_loss_fn(out_gaze, gaze_sup)

                # compute sal loss
                #compute loss
                # losses_sup = []
                #
                # loss_kld_sup = utils.kld_loss(out_sal_sup, sal_from_gt_gazes_sup)
                # loss_kld_sup = loss_kld_sup.float()
                # losses_sup.append(loss_kld_sup)
                #
                # #cc loss
                # loss_cc_sup = utils.corr_coeff(out_sal_sup.exp(), sal_from_gt_gazes_sup)
                # loss_cc_sup = loss_cc_sup.float()
                # losses_sup.append(loss_cc_sup)
                #
                # #compute total loss with weights
                # #loss_summands = self.loss_sequences(outputs, sals, fix, metrics=('kld','cc'))
                # loss_weights_sup = (1, -0.1)
                # loss_summands_sup = [l.mean(1).mean(0) for l in losses_sup]
                # loss_sup = sum(weight * l for weight, l in
                #                   zip(loss_weights_sup, loss_summands_sup))


                # Sum Losses
                #loss = loss + (0.8 * gaze_loss) + (0.8 * loss_sup)

                # ================= Track Losses =================


                running_loss += loss.item()
                batch_loss_50 += loss.item()
                running_angle_error += angle_error.item()
                angle_50 += angle_error.item()
                if batch_num % 50 == 0:
                    print("batch num: " + str(batch_num) + "/" + str(int(batch_total)) + " loss: " + str(loss.cpu().detach().numpy()) + " angle_error: " + str(angle_error.cpu().detach().numpy()) + " Loss_50: " + str(batch_loss_50 / 50) +  " Angle_50: " + str(angle_50 / 50))
                    batch_loss_50 = 0
                    angle_50 = 0
                # ================= BACKWARD =================

                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()

                # ================= Visualize Train =================
                if batch_num % 1 == 0:
                    outputs = out_sal.detach().cpu().numpy()
                    sals = s_g.detach().cpu().numpy()

                    outputs_sup = out_sal_sup.detach().cpu().numpy()
                    sals_sup = s_g_sup.detach().cpu().numpy()

                    depth_vis = depth.detach().cpu().numpy()


                    sal_from_gt_gazes = sal_from_gt_gazes.detach().cpu().numpy()

                    path = os.getcwd()
                    dirpath = os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num))#str(self.current_epoch))
                    #if not os.path.exists(dirpath):
                        #os.mkdir(os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num)))#str(self.current_epoch)))
                    for i in range(0,len(paths[0])):

                        name = str(paths[0][i])
                        index = -27
                        name = name[0 : index : ] + name[index + 1 : :]

                        name = name[:-26] + 'scene_ims/' + name[-17:-9] + '_scene.png'
                        im = cv2.imread(name)

                        name = str(paths[0][i])
                        index = -27
                        name = name[0 : index : ] + name[index + 1 : :]

                        im = im[5:485,71:-71]

                        pred = np.exp(outputs[i])
                        pred = (pred / np.amax(pred) * 255).astype(np.uint8)
                        pred = pred[0][0]

                        sal = sals[i][0][0]
                        sal = (sal / np.max(sal)) * 255

                        sal_from_gt_gaze = sal_from_gt_gazes[i]
                        sal_from_gt_gaze = (sal_from_gt_gaze / np.max(sal_from_gt_gaze)) * 255
                        #0003652_
                        # if name[-17:-9] == '0002577_':
                        #     print(np.shape(depth_vis[i]))
                        #     scipy.io.savemat(name[-17:-9] + '/self_sup/Out_Scene.mat', {'Out_Scene': im})
                        #     scipy.io.savemat(name[-17:-9] + '/self_sup/Sal_From_GT.mat', {'Sal_From_GT': sal_from_gt_gaze})
                        #     scipy.io.savemat(name[-17:-9] + '/self_sup/Sal_From_Sal_Net.mat', {'Out_Scene': pred})
                        #     scipy.io.savemat(name[-17:-9] + '/self_sup/Sal_From_Gaze_Net.mat', {'Sal_From_Gaze_Net': sal})
                        #     scipy.io.savemat(name[-17:-9] + '/self_sup/Depth_Vis.mat', {'depth_vis': depth_vis[i][5:485,71:-71]})
                        #
                        #     cv2.imwrite(name[-17:-9] + '/self_sup/Out_Scene.png', im)
                        #     cv2.imwrite(name[-17:-9] + '/self_sup/Sal_From_GT.png', sal_from_gt_gaze)
                        #     cv2.imwrite(name[-17:-9] + '/self_sup/Sal_From_Sal_Net.png', pred)
                        #     cv2.imwrite(name[-17:-9] + '/self_sup/Sal_From_Gaze_Net.png', sal)

                        #path = os.getcwd()
                        #cv2.imwrite(os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num) + '/' + name[-17:-9] + '.png'), im)

                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/out_scene_mat/' + name[-17:-9] + '.mat'), {'im': im})
                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/out_scene/' + name[-17:-9] + '.png'), im)

                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/sals_self_mat/' + name[-17:-9] + '.mat'), {'im': pred})
                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/sals_from_gaze_self_mat/' + name[-17:-9] + '.mat'), {'im': sal})

                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals_self/' + name[-17:-9] + '.png'), pred)
                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals_from_gaze_self/' + name[-17:-9] + '.png'), sal)



                        sal = cv2.cvtColor(sal.astype('float32'),cv2.COLOR_GRAY2RGB)
                        pred = cv2.cvtColor(pred.astype('float32'),cv2.COLOR_GRAY2RGB)
                        sal_from_gt_gaze = cv2.cvtColor(sal_from_gt_gaze.astype('float32'),cv2.COLOR_GRAY2RGB)

                        im = cv2.putText(im, 'Outward Scene', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        sal_from_gt_gaze = cv2.putText(sal_from_gt_gaze, 'Sal From GT', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        pred = cv2.putText(pred, 'Sal From Sal_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        sal = cv2.putText(sal, 'Sal From Gaze_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)




                        pred_sup = np.exp(outputs_sup[i])
                        pred_sup = (pred_sup / np.amax(pred_sup) * 255).astype(np.uint8)
                        pred_sup = pred_sup[0][0]

                        sal_sup = sals_sup[i][0][0]
                        sal_sup = (sal_sup / np.max(sal_sup)) * 255

                        sal_from_gt_gaze = sal_from_gt_gazes[i]
                        sal_from_gt_gaze = (sal_from_gt_gaze / np.max(sal_from_gt_gaze)) * 255


                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/sals_sup_mat/' + name[-17:-9] + '.mat'), {'im': pred_sup})
                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/sals_from_gaze_sup_mat/' + name[-17:-9] + '.mat'), {'im': sal_sup})

                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals_sup/' + name[-17:-9] + '.png'), pred_sup)
                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals_from_gaze_sup/' + name[-17:-9] + '.png'), sal_sup)


                        sal_sup = cv2.cvtColor(sal_sup.astype('float32'),cv2.COLOR_GRAY2RGB)
                        pred_sup = cv2.cvtColor(pred_sup.astype('float32'),cv2.COLOR_GRAY2RGB)
                        sal_from_gt_gaze = cv2.cvtColor(sal_from_gt_gaze.astype('float32'),cv2.COLOR_GRAY2RGB)

                        #im = cv2.putText(im, 'Outward Scene', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        sal_from_gt_gaze = cv2.putText(sal_from_gt_gaze, 'Sal From GT', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        pred_sup = cv2.putText(pred_sup, 'Sal From Sal_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        sal_sup = cv2.putText(sal_sup, 'Sal From Gaze_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



                        im_out_top = np.concatenate((im, sal_from_gt_gaze), axis=1)
                        im_out_bottom = np.concatenate((pred, sal), axis=1)
                        sal_im_out = np.concatenate((im_out_top, im_out_bottom), axis=0)

                        im_out_top_sup = np.concatenate((im, sal_from_gt_gaze), axis=1)
                        im_out_bottom_sup = np.concatenate((pred_sup, sal_sup), axis=1)
                        im_out_sup = np.concatenate((im_out_top_sup, im_out_bottom_sup), axis=0)

                        sal_im = np.concatenate((im_out_sup, sal_im_out), axis=1)

                        #sal_im = copy.deepcopy(im_out)

                        path = os.getcwd()
                        #cv2.imwrite(os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num) + '/' + name[-17:-9] + '.png'), im_out)
                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals/' + name[-17:-9] + '.png'), sal_im)


                    outputs = out_gaze.detach().cpu().numpy()
                    gazes = gaze.detach().cpu().numpy()

                    outputs_sup = out_gaze_sup.detach().cpu().numpy()

                    angle_error_vals = compute_angle_error(out_gaze, gaze).detach().cpu().numpy()
                    angle_error_vals_sup = compute_angle_error(out_gaze_sup, gaze).detach().cpu().numpy()


                    crop_x = paths[1]
                    crop_y = paths[2]
                    crop_x_0 = crop_x[0].detach().cpu().numpy()
                    crop_x_1 = crop_x[1].detach().cpu().numpy()
                    crop_y_0 = crop_y[0].detach().cpu().numpy()
                    crop_y_1 = crop_y[1].detach().cpu().numpy()
                    path = os.getcwd()
                    dirpath = os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num))
                    #if not os.path.exists(dirpath):
                    #    os.mkdir(os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num)))
                    for i in range(0,len(paths[0])):
                        name = str(paths[0][i])
                        index = -27
                        name = name[0 : index : ] + name[index + 1 : :]

                        im = cv2.imread(name)


                        im = im[crop_x_0[i]:crop_x_1[i], crop_y_0[i]:crop_y_1[i]]
                        im_sup = copy.deepcopy(im)

                        pitchyaw = outputs[i]
                        im_out = draw_gaze(im, pitchyaw, thickness=2, color=(0, 0, 255))
                        #im_out = cv2.putText(im_out, 'Predicted', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        pitchyaw = gazes[i]
                        im_out = draw_gaze(im_out, pitchyaw, thickness=2, color=(0, 255, 0))
                        #im_out = cv2.putText(im_out, 'GT', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        im_out = cv2.putText(im_out, 'Err: ' + str(np.around(angle_error_vals[i], 2)), (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        # if name[-17:-9] == '00002577':
                        #     scipy.io.savemat('0002577_' + '/self_sup/Face_Self.mat', {'Face_Self': im_out})
                        #     cv2.imwrite('0002577_' + '/self_sup/Face_Self.png', im_out)

                        pitchyaw = outputs_sup[i]
                        im_out_sup = im_sup
                        im_out_sup = draw_gaze(im_sup, pitchyaw, thickness=2, color=(0, 0, 255))
                        #im_out_sup = cv2.putText(im_out_sup, 'Predicted', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        pitchyaw = gazes[i]
                        im_out_sup = draw_gaze(im_out_sup, pitchyaw, thickness=2, color=(0, 255, 0))
                        #im_out_sup = cv2.putText(im_out_sup, 'GT', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        im_out_sup = cv2.putText(im_out_sup, 'Err: ' + str(np.around(angle_error_vals_sup[i], 2)), (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        # if name[-17:-9] == '00002577':
                        #     scipy.io.savemat('0002577_' + '/supervised/Face_Sup.mat', {'Face_Sup': im_out_sup})
                        #     cv2.imwrite('0002577_' + '/supervised/Face_Sup.png', im_out_sup)

                        #im_out = np.concatenate((im_out_sup, im_out), axis=1)

                        path = os.getcwd()
                        # cv2.imwrite(os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num) + '/' + name[-17:-9] + '_gaze.png'), im_out)

                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/gazes_self/' + name[-17:-9] + '.png'), im_out)
                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/gazes_self_mat/' + name[-17:-9] + '.mat'), {'im': im_out})
                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/gazes_super/' + name[-17:-9] + '.png'), im_out_sup)
                        scipy.io.savemat(os.path.join(path , 'visualizations_self_test/gazes_super_mat/' + name[-17:-9] + '.mat'), {'im': im_out_sup})


                        # x_offset=1000
                        # y_offset=50
                        # sal_im[y_offset:y_offset+im_out_sup.shape[0], x_offset:x_offset+im_out_sup.shape[1]] = im_out_sup
                        # x_offset=2650
                        # y_offset=50
                        # sal_im[y_offset:y_offset+im_out.shape[0], x_offset:x_offset+im_out.shape[1]] = im_out
                        # cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals/' + name[-17:-9] + '.png'), sal_im)

                        cv2.imwrite(os.path.join(path , 'visualizations_self_test/sals/' + name[-17:-9] + '_face.png'), im_out)


                batch_num += 1

        # # ================= Validation Train / Visualization =================
        # model_sal.eval()
        # model_gaze.eval()
        # print("VALIDATION")
        # with torch.no_grad():
        #     batch_num_val = 0
        #     running_loss_val = 0
        #     running_angle_error_val = 0
        #     for (batch_idx, batch) in enumerate(val_dataloader):
        #         #preprocess
        #         image = batch[0]
        #         paths = batch[1]
        #         gaze = batch[2]
        #         sal_from_gt_gazes = batch[3]
        #         scene_image = batch[4]
        #         depth = batch[5]
        #         eye_loc_3d = batch[6]
        #         scene_image = torch.unsqueeze(scene_image, 1).cuda()
        #         image = image.cuda()
        #         gaze = gaze.cuda()
        #         sal_from_gt_gazes = sal_from_gt_gazes.cuda()
        #         depth = depth.cuda()
        #         eye_loc_3d = eye_loc_3d.cuda()
        #
        #
        #         # ================= FORWARD=================
        #         out_gaze = model_gaze(image)
        #         out_sal = model_sal(scene_image)
        #
        #         out_gaze = out_gaze.float()
        #         out_sal = out_sal.float()
        #
        #         angle_error_val = compute_angle_error(out_gaze, gaze)
        #         angle_error_val = torch.mean(angle_error_val)
        #
        #         #compute loss
        #         #compute saliency from gaze
        #         #get e, g, R, K_inv, depth_rect, k
        #         k = torch.tensor(30.0).cuda()
        #         s_g = torch.empty(0).cuda()
        #         for i in range(0,out_gaze.size()[0]):
        #             s_g = torch.cat((s_g ,torch.unsqueeze(get_saliency_from_gaze_torch(eye_loc_3d[i], out_gaze[i], R, K_inv, depth[i], k).float(), 0)))
        #
        #         s_g = torch.unsqueeze(s_g, 1)
        #         s_g = torch.unsqueeze(s_g, 1)
        #
        #         #compute loss
        #         losses = []
        #
        #         loss_kld = utils.kld_loss(out_sal, s_g)
        #         loss_kld = loss_kld.float()
        #         losses.append(loss_kld)
        #
        #         #cc loss
        #         loss_cc = utils.corr_coeff(out_sal.exp(), s_g)
        #         loss_cc = loss_cc.float()
        #         losses.append(loss_cc)
        #
        #         #compute total loss with weights
        #         #loss_summands = self.loss_sequences(outputs, sals, fix, metrics=('kld','cc'))
        #         loss_weights = (1, -0.1)
        #         loss_summands = [l.mean(1).mean(0) for l in losses]
        #         loss = sum(weight * l for weight, l in
        #                           zip(loss_weights, loss_summands))
        #
        #         running_loss_val += loss.item()
        #         running_angle_error_val += angle_error_val.item()
        #         if batch_num_val % 50 == 0:
        #             print("batch num val: " + str(batch_num_val) + "/" + str(int(batch_total_val)) + " loss: " + str(loss.cpu().detach().numpy()) + " angle_error: " + str(angle_error_val.cpu().detach().numpy()))
        #         # ================= BACKWARD =================
        #
        #         # ================= Visualize Val =================
        #         if batch_num_val % 100 == 0:
        #             outputs = out_sal.detach().cpu().numpy()
        #             sals = s_g.detach().cpu().numpy()
        #             sal_from_gt_gazes = sal_from_gt_gazes.detach().cpu().numpy()
        #             path = os.getcwd()
        #             dirpath = os.path.join(path , 'visualizations_self_val/epoch_' + str(batch_num_val))#str(self.current_epoch))
        #             if not os.path.exists(dirpath):
        #                 os.mkdir(os.path.join(path , 'visualizations_self_val/epoch_' + str(batch_num_val)))#str(self.current_epoch)))
        #             for i in range(0,len(paths[0])):
        #                 name = str(paths[0][i])
        #                 index = -27
        #                 name = name[0 : index : ] + name[index + 1 : :]
        #
        #                 name = name[:-26] + 'scene_ims/' + name[-17:-9] + '_scene.png'
        #                 im = cv2.imread(name)
        #                 im = im[5:485,71:-71]
        #
        #                 pred = np.exp(outputs[i])
        #                 pred = (pred / np.amax(pred) * 255).astype(np.uint8)
        #                 pred = pred[0][0]
        #
        #                 sal = sals[i][0][0]
        #                 sal = (sal / np.max(sal)) * 255
        #
        #
        #                 sal_from_gt_gaze = sal_from_gt_gazes[i]
        #                 sal_from_gt_gaze = (sal_from_gt_gaze / np.max(sal_from_gt_gaze)) * 255
        #
        #                 sal = cv2.cvtColor(sal.astype('float32'),cv2.COLOR_GRAY2RGB)
        #                 pred = cv2.cvtColor(pred.astype('float32'),cv2.COLOR_GRAY2RGB)
        #                 sal_from_gt_gaze = cv2.cvtColor(sal_from_gt_gaze.astype('float32'),cv2.COLOR_GRAY2RGB)
        #
        #                 im = cv2.putText(im, 'Outward Scene', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #                 sal_from_gt_gaze = cv2.putText(sal_from_gt_gaze, 'Sal From GT', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #                 pred = cv2.putText(pred, 'Sal From Sal_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #                 sal = cv2.putText(sal, 'Sal From Gaze_Net', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #
        #                 im_out_top = np.concatenate((im, sal_from_gt_gaze), axis=1)
        #                 im_out_bottom = np.concatenate((pred, sal), axis=1)
        #                 im_out = np.concatenate((im_out_top, im_out_bottom), axis=0)
        #
        #                 path = os.getcwd()
        #                 cv2.imwrite(os.path.join(path , 'visualizations_self_val/epoch_' + str(batch_num_val) + '/' + name[-17:-9] + '.png'), im_out)
        #
        #             outputs = out_gaze.detach().cpu().numpy()
        #             gazes = gaze.detach().cpu().numpy()
        #             crop_x = paths[1]
        #             crop_y = paths[2]
        #             crop_x_0 = crop_x[0].detach().cpu().numpy()
        #             crop_x_1 = crop_x[1].detach().cpu().numpy()
        #             crop_y_0 = crop_y[0].detach().cpu().numpy()
        #             crop_y_1 = crop_y[1].detach().cpu().numpy()
        #             path = os.getcwd()
        #             dirpath = os.path.join(path , 'visualizations_self_val/epoch_' + str(batch_num))
        #             if not os.path.exists(dirpath):
        #                 os.mkdir(os.path.join(path , 'visualizations_self_val/epoch_' + str(batch_num)))
        #             for i in range(0,len(paths[0])):
        #                 name = str(paths[0][i])
        #                 index = -27
        #                 name = name[0 : index : ] + name[index + 1 : :]
        #                 im = cv2.imread(name)
        #
        #                 im = im[crop_x_0[i]:crop_x_1[i], crop_y_0[i]:crop_y_1[i]]
        #
        #                 pitchyaw = outputs[i]
        #                 im_out = draw_gaze(im, pitchyaw, thickness=2, color=(0, 0, 255))
        #                 im_out = cv2.putText(im_out, 'Predicted', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #                 pitchyaw = gazes[i]
        #                 im_out = draw_gaze(im_out, pitchyaw, thickness=2, color=(0, 255, 0))
        #                 im_out = cv2.putText(im_out, 'GT', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #                 path = os.getcwd()
        #                 cv2.imwrite(os.path.join(path , 'visualizations_self_val/epoch_' + str(batch_num) + '/' + name[-17:-9] + '_gaze.png'), im_out)
        #
        #
        #         batch_num_val += 1
        #
        print("Test Loss:", running_loss/batch_total)
        #print("Validation Loss:", running_loss_val/batch_total_val)
        print("Test Angle Error:", running_angle_error/batch_total)
        #print("Validation Angle Error:", running_angle_error_val/batch_total_val)
        #
        #
        # # ================= Save Checkpoint =================
        # PATH = os.path.join(path , 'saved_checkpoints/gaze_epoch_' + str(epoch) + '.pt')
        # LOSS = running_loss/batch_total
        # torch.save({
        # 'epoch': epoch,
        # 'model_state_dict': model_gaze.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': LOSS,
        # }, PATH)
        #
        # PATH = os.path.join(path , 'saved_checkpoints/sal_epoch_' + str(epoch) + '.pt')
        # torch.save({
        # 'epoch': epoch,
        # 'model_state_dict': model_sal.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': LOSS,
        # }, PATH)


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
