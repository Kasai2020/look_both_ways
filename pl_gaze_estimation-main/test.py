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
from pl_gaze_estimation.models.utils.gaze import compute_angle_error, compute_angle_error_rads
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
        #load_pretrained = True
        #load_trained_self = Talse

        data_splits = ['test1', 'test2']
        model_types = ['pretrained', 'self']
        for data_split in data_splits:
            for model_type in model_types:
                print("Data split: " + data_split + ' Model_Type: ' + model_type)
                #if data_split == 'test1' and model_type == 'pretrained':
                #    continue
                # if data_split == 'supervised':
                #     continue
                # if data_split == 'test1' and model_type == 'baseline':
                #     continue
                # if data_split == 'test1' and model_type == 'pretrained':
                #     continue
                #Creat self supervised dataloader
                # seed_everything(seed=config.EXPERIMENT.SEED, workers=True)
                # dataset = create_dataset(config)
                # dataset.setup("fit")
                # test_dataloader = dataset.train_dataloader()
                # #val_dataloader = dataset.val_dataloader()
                # #test_dataloader = dataset.test_dataloader()


                #create supervised dataloader
                if data_split == 'supervised':
                    config_test = copy.deepcopy(config)
                    with omegaconf.read_write(config_test):
                        config_test.DATASET.TEST = 'supervised'
                    dataset_test = create_dataset(config_test)
                    dataset_test.setup("fit")
                    test_dataloader = dataset_test.train_dataloader()
                elif data_split == 'test1':
                    config_test = copy.deepcopy(config)
                    with omegaconf.read_write(config_test):
                        config_test.DATASET.TEST = 'test1'
                    dataset_test = create_dataset(config_test)
                    dataset_test.setup("fit")
                    test_dataloader = dataset_test.train_dataloader()
                elif data_split == 'test2':
                    config_test = copy.deepcopy(config)
                    with omegaconf.read_write(config_test):
                        config_test.DATASET.TEST = 'test2'
                    dataset_test = create_dataset(config_test)
                    dataset_test.setup("fit")
                    test_dataloader = dataset_test.train_dataloader()
                else:
                    print("ERROR")


                #UPDATE TO USE OUR TRAINED MODELS #TODO
                model_gaze = create_model(config)
                model_sal = get_model(config)

                # model_gaze_sup = create_model(config)
                # model_sal_sup = get_model(config)

                #model = timm.create_model(config.model.name, num_classes=2)
                #model = create_model(self._config)
                #evaluate xgaze
                if model_type == 'baseline':
                    checkpoint = torch.load("/home/isaac/.ptgaze/models/eth-xgaze_resnet18.pth")
                    loaded_dict = checkpoint['model']
                    prefix = 'model.'
                    n_clip = len(prefix)
                    adapted_dict = {prefix + k: v for k, v in loaded_dict.items()}
                    model_gaze.load_state_dict(adapted_dict)


                elif model_type =='pretrained':
                    pass
                    # ------------------ 20/60/20
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/sal_epoch=0006.ckpt")
                    # model_sal.load_state_dict(checkpoint["state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch=0012.ckpt")
                    # model_gaze.load_state_dict(checkpoint["state_dict"])

                    # ------------------ 60/20/20
                    #model_sal = model_sal.load_from_checkpoint("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0261/epoch=0006.ckpt")
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0285/epoch=0002.ckpt")
                    # model_sal.load_state_dict(checkpoint["state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0289/epoch=0012.ckpt")
                    # model_gaze.load_state_dict(checkpoint["state_dict"])

                    # ------------------ 40/40/20
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/epoch=0003.ckpt")
                    # model_sal.load_state_dict(checkpoint["state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/epoch=0010.ckpt")
                    # model_gaze.load_state_dict(checkpoint["state_dict"])

                    # ------------------ 5/75/20
                    checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/sal_epoch=0004.ckpt")
                    model_sal.load_state_dict(checkpoint["state_dict"])
                    checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/gaze_epoch=0004.ckpt")
                    model_gaze.load_state_dict(checkpoint["state_dict"])

                    # ------------------ Rebuttal -----------------
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0503/epoch=0007.ckpt")
                    # model_sal.load_state_dict(checkpoint["state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/experiments/eth-xgaze/exp0502/epoch=0014.ckpt")
                    # model_gaze.load_state_dict(checkpoint["state_dict"])

                elif model_type =='self':

                    # ------------------ 20/60/20
                    # #checkpoint = torch.load("/home/isaac/Documents/best_run/sal_epoch_0.pt")
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/sal_epoch_0_batch_8500_lr_5e-09.pt")
                    # #checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/sal_epoch_0.pt")
                    # #gaze_epoch_0_batch_8500_lr_5e-09.pt
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # #checkpoint = torch.load("/home/isaac/Documents/best_run/gaze_epoch_0.pt")
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch_0_batch_8500_lr_5e-09.pt")
                    # #checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/gaze_epoch_0.pt")
                    # #sal_epoch_0_batch_8500_lr_5e-09.pt
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])
                    # # model_sal = torch.load("/home/isaac/Documents/best_run/sal_epoch_0.pt").model_state_dict
                    # # model_gaze = torch.load("/home/isaac/Documents/best_run/gaze_epoch_0.pt").model_state_dict

                    # ------------------ 20/60/20
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/sal_epoch_1.pt")
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/gaze_epoch_1.pt")
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/sal_epoch_0_batch_8500_lr_5e-09.pt")
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch_0_batch_8500_lr_5e-09.pt")
                    # # model_gaze.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/20_60_20/gaze_epoch_1_batch_3500_lr_5e-09.pt")
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])


                    # ------------------ 40/40/20
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/sal_epoch_0.pt")
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/40_40_20/gaze_epoch_0.pt")
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])

                    #------------------ 40/40/20 different learning rate -----------------
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/sal_epoch_0_batch_13500_lr_5e-07.pt")
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/gaze_epoch_0_batch_13500_lr_5e-07.pt")
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])

                    # ------------------ 60/20/20 -----------------
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/60_20_20/sal_epoch_2.pt")
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/60_20_20/gaze_epoch_1.pt")
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])

                    # ------------------ 5/75/20
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/sal_epoch_1_batch_1000_lr_5e-09.pt")
                    # model_sal.load_state_dict(checkpoint["model_state_dict"])
                    # checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/models/5_75_20/gaze_epoch_1_batch_1000_lr_5e-09.pt")
                    # model_gaze.load_state_dict(checkpoint["model_state_dict"])

                    # ------------------ Rebuttal -----------------
                    checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/sal_epoch_3.pt")
                    model_sal.load_state_dict(checkpoint["model_state_dict"])
                    checkpoint = torch.load("/home/isaac/Documents/Self-Supervised-Gaze/pl_gaze_estimation-main/saved_checkpoints/gaze_epoch_3.pt")
                    model_gaze.load_state_dict(checkpoint["model_state_dict"])

                model_gaze = model_gaze.cuda()
                model_sal = model_sal.cuda()

                # model_gaze_sup = model_gaze_sup.cuda()
                # model_sal_sup = model_sal_sup.cuda()



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
                print("Data split: " + data_split + ' Model_Type: ' + model_type)
                # os.mkdir('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type)
                # with open('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type + '/gt_gazes.txt', 'w') as f:
                #     f.write('')
                # with open('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type + '/pred_gazes.txt', 'w') as f:
                #     f.write('')


                batch_num = 0
                running_loss = 0
                batch_loss_50 = 0
                angle_50 = 0
                running_angle_error = 0
                sal_error_50 = 0
                running_sal_error = 0
                MAE_sal_all = torch.empty(0).cuda()
                MAE_angle_error_all = torch.empty(0).cuda()
                Degree_angle_error_all = torch.empty(0).cuda()
                loss_kld_all = torch.empty(0).cuda()
                model_sal.eval()
                model_gaze.eval()
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
                        #
                        # with open('/media/isaac/easystore/Data/5_75_20/' + data_split + '_' + model_type + '/gt_gazes.txt', 'a') as f:
                        #     np.savetxt(f, gaze.detach().cpu().numpy())
                        #
                        # with open('/media/isaac/easystore/Data/5_75_20/' + data_split + '_' + model_type + '/pred_gazes.txt', 'a') as f:
                        #     np.savetxt(f, out_gaze.detach().cpu().numpy())



                        #Compute angle error
                        angle_error = compute_angle_error(out_gaze, gaze)
                        Degree_angle_error_all = torch.cat((Degree_angle_error_all, angle_error), dim=0)
                        angle_error = torch.mean(angle_error)

                        MAE_angle_error = compute_angle_error_rads(out_gaze, gaze)
                        MAE_angle_error_all = torch.cat((MAE_angle_error_all, MAE_angle_error), dim=0)


                        #compute loss
                        #compute saliency from gaze
                        #get e, g, R, K_inv, depth_rect, k
                        # k = torch.tensor(30.0).cuda()
                        # s_g = torch.empty(0).cuda()
                        # for i in range(0,out_gaze.size()[0]):
                        #     s_g = torch.cat((s_g ,torch.unsqueeze(get_saliency_from_gaze_torch(eye_loc_3d[i], out_gaze[i], R, K_inv, depth[i], k).float(), 0)))
                        #
                        # s_g = torch.unsqueeze(s_g, 1)
                        # s_g = torch.unsqueeze(s_g, 1)

                        sal_from_gt_gazes = torch.unsqueeze(sal_from_gt_gazes, 1)
                        sal_from_gt_gazes = torch.unsqueeze(sal_from_gt_gazes, 1)



                        #compute sal error metric
                        out_sal_norm = out_sal.exp()
                        #np.save('/media/isaac/easystore/Data/20_60_20/' + data_split + '_' + model_type + '/gt_sal_batch_' + str(batch_num) + '.npy', sal_from_gt_gazes.detach().cpu().numpy())
                        #np.save('/media/isaac/easystore/Data/20_60_20/' + data_split + '_' + model_type + '/pred_sal_batch_' + str(batch_num) + '.npy', out_sal_norm.detach().cpu().numpy())


                        diff_sal = out_sal_norm - sal_from_gt_gazes

                        diff_sal = torch.squeeze(diff_sal, 1)
                        diff_sal = torch.squeeze(diff_sal, 1)

                        batch_avg_sal_norm = torch.mean(torch.norm(torch.norm(diff_sal, dim=1), dim=1))

                        MAE_sal = torch.sum(torch.sum(torch.abs(diff_sal), dim=1), dim=1)
                        MAE_sal_all = torch.cat((MAE_sal_all, MAE_sal), dim=0)



                        # print(torch.sum(diff_sal))
                        # out_sal_norm = out_sal_norm.detach().cpu().numpy()
                        # sal_from_gt_gazes_detach = sal_from_gt_gazes.detach().cpu().numpy()
                        # for i in range(0,len(out_sal_norm)):
                        #     print(torch.norm(diff_sal[i]))
                        #     out_sal_norm_i = out_sal_norm[i][0][0]
                        #     sal_from_gt_gazes_detach_i = sal_from_gt_gazes_detach[i][0][0]
                        #     print(np.shape(out_sal_norm_i))
                        #     print(np.shape(sal_from_gt_gazes_detach_i))
                        #     out_sal_norm_i = (out_sal_norm_i / np.max(out_sal_norm_i)) * 255
                        #     sal_from_gt_gazes_detach_i = (sal_from_gt_gazes_detach_i / np.max(sal_from_gt_gazes_detach_i)) * 255
                        #     im_out = np.concatenate((out_sal_norm_i, sal_from_gt_gazes_detach_i), axis=1)
                        #     path = os.getcwd()
                        #     cv2.imwrite(os.path.join(path , 'visualizations_self_test/batch_' + str(batch_num) + '/' + str(i) + '.png'), im_out)
                        # ok



                        #compute loss
                        losses = []
                        loss_kld = utils.kld_loss(out_sal, sal_from_gt_gazes)
                        loss_kld = loss_kld.float()


                        loss_kld_all = torch.cat((loss_kld_all, loss_kld), dim=0)


                        losses.append(loss_kld)

                        #cc loss
                        loss_cc = utils.corr_coeff(out_sal.exp(), sal_from_gt_gazes)
                        loss_cc = loss_cc.float()
                        losses.append(loss_cc)

                        #compute total loss with weights
                        #loss_summands = self.loss_sequences(outputs, sals, fix, metrics=('kld','cc'))
                        loss_weights = (1, -0.1)
                        loss_summands = [l.mean(1).mean(0) for l in losses]
                        loss = sum(weight * l for weight, l in
                                          zip(loss_weights, loss_summands))



                        running_loss += loss.item()
                        batch_loss_50 += loss.item()
                        running_angle_error += angle_error.item()
                        angle_50 += angle_error.item()
                        running_sal_error += batch_avg_sal_norm.item()
                        sal_error_50 += batch_avg_sal_norm.item()
                        if batch_num % 50 == 0:
                            if batch_num % 100 == 0:
                                print("batch num: " + str(batch_num) + "/" + str(int(batch_total)) + " loss: " + str(loss.cpu().detach().numpy()) + " Angle_MAE: " + str(torch.mean(MAE_angle_error_all).cpu().detach().numpy()) + " Sal_MAE: " + str(torch.mean(MAE_sal_all).cpu().detach().numpy()) +  " Angle_Err: " + str(torch.mean(Degree_angle_error_all).cpu().detach().numpy()))
                            batch_loss_50 = 0
                            angle_50 = 0
                            sal_error_50 = 0


                        batch_num += 1


                print("Data split: " + data_split + ' Model_Type: ' + model_type)
                print("Test Loss:", running_loss/batch_total)
                #print("Validation Loss:", running_loss_val/batch_total_val)
                print("Test Angle Error:", torch.mean(Degree_angle_error_all).cpu().detach().numpy())
                print("Test Angle Std:", torch.std(Degree_angle_error_all).cpu().detach().numpy())

                #print("Validation Angle Error:", running_angle_error_val/batch_total_val)
                print("Sal Error:", torch.mean(MAE_sal_all).cpu().detach().numpy())
                print("Sal Std:", torch.std(MAE_sal_all).cpu().detach().numpy())

                print("MAE Error:", torch.mean(MAE_angle_error_all).cpu().detach().numpy())
                print("MAE Std:", torch.std(MAE_angle_error_all).cpu().detach().numpy())

                print("loss_kld_all Error:", torch.mean(loss_kld_all).cpu().detach().numpy())
                print("loss_kld_all Std:", torch.std(loss_kld_all).cpu().detach().numpy())

                # np.save('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type + '/all_MAE_deg_gaze_error.npy', Degree_angle_error_all.detach().cpu().numpy())
                # np.save('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type + '/all_MAE_gaze_error.npy', MAE_angle_error_all.detach().cpu().numpy())
                # np.save('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type + '/all_MAE_sal_error.npy', MAE_sal_all.detach().cpu().numpy())
                # np.save('/media/isaac/easystore/Data/60_20_20_test/' + data_split + '_' + model_type + '/loss_kld_all_error.npy', loss_kld_all.detach().cpu().numpy())


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
