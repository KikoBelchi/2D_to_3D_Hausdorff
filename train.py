
# coding: utf-8

# CAVEAT: 
# train.py is the continuation of transfer_learning_TowelWall.py, so look at its git history for early changes.
# transfer_learning_TowelWall.py, in turn, is the continuation of transfer_learning_TowelWall_ipynb_old.ipynb, so look at its git history for early changes.

# 
# Transfer Learning to learn 3D from 2D
# ===============================
# **Author**: `Francisco Belchí <frbegu@gmail.com>, <https://github.com/KikoBelchi/2d_to_3d>`_
# 





###
### Procedure for applying the normalization 0<=x,y,z<=1 to the whole dataset using the metadata from the training data:
### (the normalization 0<=Depth<=1 is completely analogous)
###
# Instanciate un-normalized dataset.
# Training/validation split using random seeds we will reuse later on for reproducing the same split.
# Find min and max of x, y, z coordinates of all vertices within the training set.
# Instanciate normalized dataset.
# Reproduce the same training/validation split with the random seed from above.





###
### Imports
###
from __future__ import print_function, division

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from skimage import io, transform # package 'scikit-image'
import pandas as pd
from PIL import Image
from tensorboardX import SummaryWriter
import argparse
import random

import data_loading
import diff_Hausdorff_loss
import functions_data_processing
import functions_train
from loss_functions import MSE_loss
import resnet_including_dropout

if __name__=='__main__':
    # Parser for entering training setting arguments and setting default ones
    args = functions_data_processing.parser()

    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)

    # Set directory in which to save the tensorboard visualizations and the trained model
    saving_directory=functions_train.trained_model_directory(args)
    # Create directory if it does not exist
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
        print('\nCreating directory:\n' + saving_directory + "\n")

    if args.all_normalizing_provided==0:
        # Instanciate dataset with un-normalized xyz and D
        if args.find_xyz_norm_given_normalized_uvD==1:
            kwargs_normalize['normalize_D_min'] = args.normalize_datasetClass_D_min
            kwargs_normalize['normalize_D_max'] = args.normalize_datasetClass_D_max
            transformed_dataset = data_loading.instanciate_dataset(args, **kwargs_normalize)
        else:
            transformed_dataset = data_loading.instanciate_dataset(args)


        # Training/validation/(test) split using random seeds (kept in args) for reproducing it below
        transformed_dataset_parts, dataset_sizes, dataloaders = functions_train.create_dataloaders(transformed_dataset, args)
        functions_train.print_split_info(transformed_dataset, transformed_dataset_parts, dataset_sizes, dataloaders, args.verbose)

        # Find min and max of variables for normalization
        if args.normalization == 3:
            # Find min and max of x, y, z coordinates of all vertices within the training set.
            print("x_min, y_min, z_min, x_max, y_max, z_max from training set (disregarding outliers):")
            x_min, y_min, z_min, x_max, y_max, z_max = functions_data_processing.find_min_max_xyz_training_wo_outliers(
                dataloaders, boxplot_on=args.boxplots_on, sequence_name=args.sequence_name, f_u=args.f_u, f_v=args.f_v, args=args)
        #     x_min, y_min, z_min, x_max, y_max, z_max = functions_data_processing.find_min_max_xyz_training(dataloaders)
            print('{0:.16f}'.format(x_min), '{0:.16f}'.format(y_min), '{0:.16f}'.format(z_min), '{0:.16f}'.format(x_max), '{0:.16f}'.format(y_max), '{0:.16f}'.format(z_max))
            normalize_xyz_min, normalize_xyz_max = [x_min, y_min, z_min], [x_max, y_max, z_max]   
            file_name = os.path.join(saving_directory, 'normalization_params_from_training.txt')
            with open(file_name, 'w') as x_file:
                text_to_save = "x_min, y_min, z_min, x_max, y_max, z_max from training set"
                print("Saving " + text_to_save + " to " + file_name + "...")
                text_to_save += ":\n" + str(x_min) + ' ' + str(y_min) + ' ' + str(z_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + str(z_max)
                x_file.write(text_to_save)
        else: 
            normalize_xyz_min, normalize_xyz_max = [0,0,0], [1,1,1]

        if args.D_normalization == 3:
            # Find min and max of depth of all vertices within the training set.
            print("D_min, D_max from training set (disregarding outliers):")
            D_min, D_max = functions_data_processing.find_min_max_D_training_wo_outliers(dataloaders, boxplot_on=args.boxplots_on)
        #     D_min, D_max = functions_data_processing.find_min_max_D_training(dataloaders)
            print('{0:.16f}'.format(D_min), '{0:.16f}'.format(D_max))
            normalize_D_min, normalize_D_max = D_min, D_max
            file_name = os.path.join(saving_directory, 'depth_params_from_training.txt')
            with open(file_name, 'w') as x_file:
                text_to_save = "D_min, D_max from training set"
                print("Saving " + text_to_save + " to " + file_name + "...")
                text_to_save += ":\n" + str(D_min) + ' ' + str(D_max)
                x_file.write(text_to_save)
        else: 
            normalize_D_min, normalize_D_max = 0, 1 # normalizing with these values does nothing
    else:
        normalize_xyz_min, normalize_xyz_max, normalize_D_min, normalize_D_max = None, None, None, None

    kwargs_normalize, kwargs_normalize_labels_pred = functions_data_processing.create_kwargs_normalize(normalize_xyz_min, normalize_xyz_max, normalize_D_min, normalize_D_max, args)

    # Create dataset again, this time, normalized as demanded by args
    if args.normalization == 3 or args.D_normalization == 3:
        # Instanciate normalized dataset
        print('Creating normalized dataset...')
        transformed_dataset = data_loading.vertices_Dataset(
            args, **kwargs_normalize)

        # Reproduce the same training/validation/(test) split with the same random seed as above.
        transformed_dataset_parts, dataset_sizes, dataloaders = functions_train.create_dataloaders(transformed_dataset, args)
        functions_train.print_split_info(transformed_dataset, transformed_dataset_parts, dataset_sizes, dataloaders, args.verbose)



    # Get template
    # CAVEAT: to make the first tests, I will take template = transformed_dataset[0],
    # which is Group 1, frame 1,
    # but this observation may not be in the training set, 
    # so if the unsupervised approach works, I will change this to 
    # make the template be an observation independent from the rest of the dataset.
    args.template = transformed_dataset[0]

    if args.train_w_annot_only==1:
        # If there is a split of the training set into annotated and non-annotated data 
        # but we only want to train with the annotated data of that split.
        args.lengths_proportion_train_nonAnnotated = None

###
### Training the model
###
def save_models(saving_directory, model, n_round = 1):
#     model_name = 'model.pt' if n_round==1 else 'model_rd' + str(n_round) + '.pt'
    model_name = 'model.pt' 
    model_filename = os.path.join(saving_directory, model_name)
    torch.save(model, model_filename)
    
def save_Nan(saving_directory):
    text_to_save = 'Nan'
    file_name = os.path.join(saving_directory, 'Nan.txt')
    with open(file_name, 'w') as x_file:
        x_file.write(text_to_save)
    
# import math
# def isnan(tensor):
#     # Gross: https://github.com/pytorch/pytorch/issues/4767
#     return (tensor != tensor) or (math.isnan(tensor)) or np.isnan(tensor)

def isnan(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return (tensor != tensor)

def hasnan(tensor):
    return isnan(tensor).any()

def remove_losses_we_dont_need(losses, args):
    """ Remove from the loss dictionary the losses with 0 weight which we don't need.
    Note that we will always keep uv_loss and xyz_loss even if they have 0 weight."""
    if args.n_outputs==1:
        if args.loss_w_chamfer==0: del losses['loss_chamfer']
        if args.loss_w_chamfer_contour==0: del losses['loss_chamfer_contour']
    else: 
        if args.loss_w_chamfer_GT_pred==0: del losses['loss_chamfer_GT_pred']
        if args.loss_w_chamfer_pred_GT==0: del losses['loss_chamfer_pred_GT']
        if args.loss_w_chamfer_GTcontour_pred==0: del losses['loss_chamfer_GTcontour_pred']
        if args.loss_w_chamfer_pred_GTcontour==0: del losses['loss_chamfer_pred_GTcontour']
    if args.loss_w_geo==0: del losses['loss_geo']
    if args.loss_w_horizConsecEdges==0: del losses['loss_horizConsecEdges']
    if args.loss_w_verConsecEdges==0: del losses['loss_verConsecEdges']
    return losses

def losses_forward(args, outputs, outputs_uv_coord, outputs_xyz_coord, 
                   labels, labels_uv_coord, labels_xyz_coord, criterion, sample_batched, phase, outputs_D=0, labels_D=0):
    # torch.nn MSELoss
    losses=OrderedDict()
    if args.loss_weights==0:
        losses['loss'] = criterion(outputs, labels) # old way of applying the weights, to the tensors from which to compute a single loss, rather than to the computed losses
#                         print("loss.data[0]:", loss.data[0])
#                         print("outputs:", outputs)
#                         print(labels)
    else:
        # MSE loss of predicted coordinates only, 
        # with no extra structure added such as distances between neighbouring vertices, normal vectors...
        if args.loss_w_uv!=0:
            losses['loss_uv_coord_only'] = criterion(outputs_uv_coord, labels_uv_coord) 
        else:
            losses['loss_uv_coord_only'] = torch.zeros([], dtype=outputs_xyz_coord.dtype, requires_grad=False, device=args.device) 
        if args.loss_w_xyz!=0:
            losses['loss_xyz_coord_only'] = criterion(outputs_xyz_coord, labels_xyz_coord) 
        else:
            losses['loss_xyz_coord_only'] = torch.zeros([], dtype=outputs_xyz_coord.dtype, requires_grad=False, device=args.device) 
        if args.loss_w_D!=0:
            losses['loss_D_coord_only'] = criterion(outputs_D, labels_D) 
        else:
            losses['loss_D_coord_only'] = torch.zeros([], dtype=outputs_D.dtype, requires_grad=False, device=args.device) 
        
        if args.n_outputs==1:
            # At the moment, when using the Hausdorff loss, 
            # I sum both Hausdorff directional distances in that one loss,
            # so I will sum the weights of the 2 directions. See functions_data_processing.py
            if args.loss_w_chamfer!=0:
                args.contour=0
                losses['loss_chamfer'] = diff_Hausdorff_loss.loss_Hausdorff_4_batches(
                    args.criterion_Hauss, outputs_uv_coord, sample_batched, args)
            else: 
                losses['loss_chamfer'] = torch.zeros([], dtype=outputs_xyz_coord.dtype, requires_grad=False, device=args.device) 
        else: 
            if args.loss_w_chamfer_GT_pred!=0 or args.loss_w_chamfer_pred_GT!=0:
                args.contour=0
                losses['loss_chamfer_GT_pred'], losses['loss_chamfer_pred_GT'] = diff_Hausdorff_loss.loss_Hausdorff_4_batches(
                    args.criterion_Hauss, outputs_uv_coord, sample_batched, args)
            else: 
                losses['loss_chamfer_GT_pred']= torch.zeros([], dtype=outputs_xyz_coord.dtype, 
                                                            requires_grad=False, device=args.device) 
                losses['loss_chamfer_pred_GT'] = torch.zeros([], dtype=outputs_xyz_coord.dtype, 
                                                                 requires_grad=False, device=args.device) 
            # Remove next line once all options for directional Hausdorff are working
#             losses['loss_chamfer'] = losses['loss_chamfer_GT_pred'] + losses['loss_chamfer_pred_GT']

        if args.n_outputs==1:
            if args.loss_w_chamfer_contour!=0:
                args.contour=1
                losses['loss_chamfer_contour'] = diff_Hausdorff_loss.loss_Hausdorff_4_batches(
                    args.criterion_Hauss, outputs_uv_coord, sample_batched, args)
            else: 
                losses['loss_chamfer_contour'] = torch.zeros([], dtype=outputs_xyz_coord.dtype,
                                                   requires_grad=False, device=args.device) 
        else: 
            if args.loss_w_chamfer_GTcontour_pred!=0 or args.loss_w_chamfer_pred_GTcontour!=0:
                args.contour=1
                losses['loss_chamfer_GTcontour_pred'], losses['loss_chamfer_pred_GTcontour'] = diff_Hausdorff_loss.loss_Hausdorff_4_batches(
                    args.criterion_Hauss, outputs_uv_coord, sample_batched, args)
            else: 
                losses['loss_chamfer_GTcontour_pred']= torch.zeros([], dtype=outputs_xyz_coord.dtype, 
                                                            requires_grad=False, device=args.device) 
                losses['loss_chamfer_pred_GTcontour'] = torch.zeros([], dtype=outputs_xyz_coord.dtype, 
                                                                 requires_grad=False, device=args.device) 
            # Remove next line once all options for directional Hausdorff are working
#             losses['loss_chamfer_contour'] = losses['loss_chamfer_GTcontour_pred'] + losses['loss_chamfer_pred_GTcontour']
        
        if args.loss_w_geo!=0: 
            losses['loss_geo'] = functions_train.create_loss_geo(args, outputs_xyz_coord, labels_xyz_coord)
        else: 
            losses['loss_geo'] = torch.zeros([], dtype=outputs_xyz_coord.dtype,
                                               requires_grad=False, device=args.device) 
            
        if args.loss_w_horizConsecEdges!=0: 
            losses['loss_horizConsecEdges'] = functions_train.create_loss_horizConsecEdges(args, outputs_xyz_coord, labels_xyz_coord)
        else: 
            losses['loss_horizConsecEdges'] = torch.zeros([], dtype=outputs_xyz_coord.dtype,
                                               requires_grad=False, device=args.device)
        if args.loss_w_verConsecEdges!=0: 
            losses['loss_verConsecEdges'] = functions_train.create_loss_verConsecEdges(args, outputs_xyz_coord, labels_xyz_coord)
        else: 
            losses['loss_verConsecEdges'] = torch.zeros([], dtype=outputs_xyz_coord.dtype,
                                               requires_grad=False, device=args.device)
        
        dtype = torch.float64 if args.dtype==1 else torch.float32
        with torch.no_grad():
            # MSE loss of predicted coordinates only, 
            # with no extra structure added such as distances between neighbouring vertices, normal vectors...
            if args.loss_w_uv==0:
                if args.predict_uv_or_xyz=='uv' or args.predict_uv_or_xyz=='uvy':
                    losses['loss_uv_coord_only'] = criterion(outputs_uv_coord, labels_uv_coord) 
                else:
                    losses['loss_uv_coord_only'] = -1 * torch.ones([1], dtype=dtype, device=args.device)
            if args.loss_w_xyz==0:
                if args.predict_uv_or_xyz in ['xyz', 'uvy', 'uvD']:
                    losses['loss_xyz_coord_only'] = criterion(outputs_xyz_coord, labels_xyz_coord) 
                else:
                    losses['loss_xyz_coord_only'] = -1 * torch.ones([1], dtype=dtype, device=args.device)
            if args.loss_w_D==0:
                if args.predict_uv_or_xyz=='uvD':
                    losses['loss_D_coord_only'] = criterion(outputs_D, labels_D) 
                else:
                    losses['loss_D_coord_only'] = -1 * torch.ones([1], dtype=dtype, device=args.device)

            if args.loss == 1: # my own MSE loss of predicted coordinates only
                if args.loss_w_uv==0:
                    if args.predict_uv_or_xyz=='uv' or args.predict_uv_or_xyz=='uvy':
                        losses['loss_myown_uv_coord_only'] = MSE_loss(outputs_uv_coord, labels_uv_coord, 
                                                            batch_size=args.batch_size, num_selected_vertices=2)
                    else:
                        losses['loss_myown_uv_coord_only'] = -1
                if args.loss_w_xyz==0:
                    if args.predict_uv_or_xyz=='xyz' or args.predict_uv_or_xyz=='uvy':
                        losses['loss_myown_xyz_coord_only'] = MSE_loss(outputs_xyz_coord, labels_xyz_coord,
                                                             batch_size=args.batch_size, num_selected_vertices=3)
                    else:
                        losses['loss_myown_xyz_coord_only'] = -1

        if phase == 'train_nonAnnot':
            for key in losses:
                if key in ['loss_uv_coord_only', 'loss_xyz_coord_only', 'loss_D_coord_only']:
                    losses[key]=losses[key].detach()
            
        if args.n_outputs==1:
            losses['loss'] = args.loss_factor*(args.loss_w_uv*losses['loss_uv_coord_only'] + args.loss_w_xyz*losses['loss_xyz_coord_only'] + args.loss_w_D*losses['loss_D_coord_only'] + args.loss_w_chamfer*losses['loss_chamfer'] + args.loss_w_chamfer_contour*losses['loss_chamfer_contour'] + args.loss_w_geo*losses['loss_geo'] + args.loss_w_horizConsecEdges*losses['loss_horizConsecEdges'] + args.loss_w_verConsecEdges*losses['loss_verConsecEdges'])
        else:
            losses['loss'] = args.loss_factor*(args.loss_w_uv*losses['loss_uv_coord_only'] + args.loss_w_xyz*losses['loss_xyz_coord_only'] + args.loss_w_D*losses['loss_D_coord_only'] + args.loss_w_chamfer_GT_pred*losses['loss_chamfer_GT_pred'] + args.loss_w_chamfer_pred_GT*losses['loss_chamfer_pred_GT'] + args.loss_w_chamfer_GTcontour_pred*losses['loss_chamfer_GTcontour_pred'] + args.loss_w_chamfer_pred_GTcontour*losses['loss_chamfer_pred_GTcontour'] + args.loss_w_geo*losses['loss_geo'] + args.loss_w_horizConsecEdges*losses['loss_horizConsecEdges'] + args.loss_w_verConsecEdges*losses['loss_verConsecEdges'])

        return remove_losses_we_dont_need(losses, args)

def initialize_run_loss(losses):
    run_loss=OrderedDict()
    for key in losses:
        run_loss[key] = 0.0
    return run_loss
            
def update_run_loss(losses, args, run_loss):           
    for key in run_loss:
        if key in ['loss_myown_uv_coord_only', 'loss_myown_xyz_coord_only']:
            run_loss[key] += losses[key]
        else:
            run_loss[key] += losses[key].item()
    #     # loss.item() is the average (at least with MSE) of 
    #     # the losses of the observations in the batch.
    #             At the end of the epoch, running_loss will be the sum of loss.item() for all the batches.
    #             Then, we will define the epoch loss as the average of the MSE loss of each observation,
    #             which equals 
    #             running_loss * args.batch_size / training_size,
    #             or equivalently, 
    #             running_loss / (number of batches in which the training data is split),
    #             or the analogue with validation set size if phase=='val' 
    return run_loss

def print_train_batch_loss(losses, args, epoch, batch_idx, dataset_sizes, phase):
    pretext = 'Train Epoch:' if phase == 'train' else 'Train (non annotated) Epoch:'
    
    n_batches = len(dataloaders[phase]) # number of batches in which the phase data is split
    phase_size = dataset_sizes[phase] # size of phase set
    
    for key in losses:
        if losses[key].item()!=-1.:
            print(pretext + ' {} [{}/{} ({:.0f}%)]'.format(
                epoch+1, batch_idx * args.batch_size, phase_size, 100. * batch_idx / n_batches), end='\t')
            if key=='loss':
                print(key + ': {:.6f}'.format(losses[key].item()/args.loss_factor))
            else:
                print(key + ': {:.6f}'.format(losses[key].item()))
    
    if args.loss == 1: # my own MSE loss
        print(pretext + ' {} [{}/{} ({:.0f}%)]\tMy Loss on uv_coord_only: {:.6f}'.format(
            epoch+1, batch_idx * args.batch_size, phase_size,
            100. * batch_idx / n_batches, losses['loss_myown_uv_coord_only']))
        print(pretext + ' {} [{}/{} ({:.0f}%)]\tMy Loss on xyz_coord_only: {:.6f}'.format(
            epoch+1, batch_idx * args.batch_size, phase_size,
            100. * batch_idx / n_batches, losses['loss_myown_xyz_coord_only']))
    
def save_train_batch_losses_to_tensorboard(epoch, dataloaders, batch_idx, losses, writer, phase):
    # Save training loss to tensorboardX
    n_batches = len(dataloaders[phase]) # number of batches in which the phase data is split
    niter = int(epoch*n_batches+batch_idx) 
    # niter is the number of iterations. 
    # In the epoch 0, it coincides with the batch_idx. 
    # In the epoch 1, it will be 750+args.batch_size (if training_size=3000, batch_size=4)

    for key in losses:
        if key=='loss':
            writer.add_scalar(phase + '/' + key + '_1perBatch', losses[key].item()/args.loss_factor, niter) 
        else:
            writer.add_scalar(phase + '/' + key + '_1perBatch', losses[key].item(), niter) 
    # If I return a loss which is not 1 number per batch but one per element in the batch,
    # then consider doing something else like
    # writer.add_scalar(phase + '/Loss', loss.data[0], niter) or
    # writer.add_scalar(phase + '/Loss', loss.data[0].item(), niter)
    # 'loss.data[0]' is a tensor with a sigle value, and that value is 'loss.item()'

def epoch_loss_and_print(run_loss, dataloaders, phase, args):
    """
    Compute epoch loss from running loss and number of batches.
    Print the losses and return them as text to be saved to file later.
    
    Recall: 
    dataset_sizes['train'] == training_size
    len(dataloaders['train']) == training_size / args.batch_size, i.e.,
    len(dataloaders['train']) == number of batches in which the training data is split
    inputs.size(0) == args.batch_size
    """
    num_batches = len(dataloaders[phase]) # number of batches in which the phase data is split
    text_to_print=''
    epoch_losses=OrderedDict()
    for key in run_loss:
        if key=='loss':
            epoch_losses[key] = run_loss[key] / (num_batches * args.loss_factor) 
        else:
            epoch_losses[key] = run_loss[key] / num_batches 
        if epoch_losses[key]!=-1.:
            text_to_print += phase + ' ' + key +': {0:.16f}'.format(epoch_losses[key]) + "\n"
    print(text_to_print)
    return epoch_losses, text_to_print
    
def save_epoch_losses_to_tensorboard(epoch, phase, writer, epoch_losses, args):
    niter = epoch+1
    for key in epoch_losses:
        writer.add_scalar(phase + '/' + key + '_1perEpoch', epoch_losses[key], niter)

def create_phases(args, epoch):
    """Each epoch has a training and validation phase"""
    if args.lengths_proportion_train_nonAnnotated is None:
        phases = ['train', 'val']
    elif args.lengths_proportion_train_nonAnnotated==1:
        phases = ['train_nonAnnot', 'val']
    else:
        if epoch==args.num_epochs-1 and args.lastEpochSupOnly==1:
            phases = ['train', 'val']
        elif epoch==args.num_epochs-1 and args.lastEpochSupOnly==2:
            phases = ['train', 'train_nonAnnot', 'train_nonAnnot', 'train_nonAnnot', 'val']
            # These 5 phases will use 
            # [model.train(), model.eval(), model.train(), model.eval(), model.eval()], respectively
            # and coorespond to the value of train_vs_eval_seq: 
            # [0, 1, 2, 3, 4], respectively
        else:
            phases = ['train', 'train_nonAnnot', 'val']
    return phases

def create_phases_4_evaluation(args, dataloaders):
    phases = ['train']
#     if args.lengths_proportion_train_nonAnnotated is not None:
#         phases.append('train_nonAnnot')
    phases.append('val')
    if (args.lengths_proportion_test is not None) and (args.lengths_proportion_test!=0):
        phases.append('test')
#     print(dataloaders.keys())
    return phases

def eval_model_setting(args, eval_only=0):
    """If eval_only==1, then train_model() will not train the model but 
    instead, it will only evaluate and save the losses.
    eval_model_setting() sets the parameters to make this change within train_model()."""
    if eval_only==1:
        args.num_epochs = 1
    return args
    
def train_model(model, optimizer, scheduler, saving_directory, writer, criterion,
               dataloaders, args, kwargs_normalize_labels_pred, eval_only=0):
    """Function to train (or only evaluate, if eval_only==1) the model"""
    since = time.time()
    train_vs_eval_seq = 0
    args=eval_model_setting(args, eval_only)
    text_to_print_eval_only=''
    
    for epoch in range(args.num_epochs):
        if eval_only==0:
            if epoch==args.num_epochs-1 and args.lastEpochSupOnly==2:
                print("\n"*2 + "LAST EPOCH")
            scheduler.step()
            args.epoch = epoch
            print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
            print('-' * 10)
            args.invert_distance_direction=1 if epoch < args.n_epochs_pred_GT else 0

            phases=create_phases(args, epoch)      
        else:
            phases=create_phases_4_evaluation(args, dataloaders)
            
        for phase in phases:
            if eval_only==0:
                if epoch==args.num_epochs-1 and args.lastEpochSupOnly==2:
                    if train_vs_eval_seq in [0, 2]:
                        print("\nTrain model")
                        model.train()
                    else:
                        print("\nEvaluate model")
                        model.eval()
                else:
                    if phase in ['train', 'train_nonAnnot']:
                        if args.testing_eval in [0,3]:
                            model.train() # Set model to training mode during training phase
                        elif args.testing_eval == 1:
                            model.eval() # Set model to evaluate mode. Hence, some possible BatchNorm or Dropout may be omitted.
                        elif args.testing_eval == 2: 
                            # Set model to training mode but freeze BatchNorm weights 
                            # The freezing is done beforehand, when defining the network architecture
                            model.train()   
                    else:
                        if args.testing_eval == 3:
                            model.train() # Set model to training mode during validation phase
                        else:
                            model.eval() # Set model to evaluate mode during validation phase
            else: 
                model.eval()

            # Iterate over data
            for batch_idx, sample_batched in enumerate(dataloaders[phase], 1):
                for key in ['towel_pixel_subsample', 'towel_pixel_contour_subsample']:
                    if key in sample_batched:
                        if args.dtype==0:
                            sample_batched[key] = sample_batched[key].to(args.device).float()
                        else:
                            sample_batched[key] = sample_batched[key].to(args.device).double()
                
                inputs = sample_batched['image'].to(args.device)
                
                labels, labels_uv_coord, labels_xyz_coord, labels_D = functions_train.create_labels_and_prediction(
                    args, sample_batched=sample_batched, **kwargs_normalize_labels_pred)

                # zero the parameter gradients
                if eval_only==0: optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(eval_only==0 and 
                                            phase in ['train', 'train_nonAnnot'] and train_vs_eval_seq in [0, 2]):
                    # Shape args.batch_size x (args.num_selected_vertices * args.num_coord_per_vertex)
                    outputs, outputs_uv_coord, outputs_xyz_coord, outputs_D = functions_train.create_labels_and_prediction(
                        args, sample_batched=sample_batched, GT_or_prediction='pred', model=model, inputs=inputs, 
                        **kwargs_normalize_labels_pred)

                    losses = losses_forward(
                        args, outputs, outputs_uv_coord, outputs_xyz_coord, labels, labels_uv_coord, labels_xyz_coord, criterion,
                        sample_batched, phase, outputs_D, labels_D)

                    if eval_only==0 and args.log_epochs!=0 and batch_idx==1 and epoch%args.log_epochs==0:
                        functions_train.save_predictions(phase, epoch+1, labels, outputs, loss, 
                                                         sample_batched_img_name=sample_batched['img_name'], 
                                                         saving_directory=saving_directory, n_round=args.round)

                    # backward + optimize only if in training phase
                    if eval_only==0 and phase in ['train', 'train_nonAnnot'] and train_vs_eval_seq in [0, 2]:
                        losses['loss'].backward()

                        # Update model coefficients after computing the loss for a batch
                        # (perform mini-batch gradient descent)
                        optimizer.step() 

                if batch_idx==1:
                    run_loss = initialize_run_loss(losses)
                run_loss = update_run_loss(losses, args, run_loss)

                if eval_only==0 and phase in ['train', 'train_nonAnnot'] and args.lastEpochSupOnly not in [1, 3]:
                    if batch_idx % args.log_interval == 0:
                        print_train_batch_loss(losses, args, epoch, batch_idx, dataset_sizes, phase)
                        save_train_batch_losses_to_tensorboard(epoch, dataloaders, batch_idx, losses, writer, phase)

            num_batches = len(dataloaders[phase]) # number of batches in which the phase data is split
            if num_batches!=0:
                # Average of the MSE of each observation
                epoch_losses, text_to_print = epoch_loss_and_print(run_loss, dataloaders, phase, args)
                if eval_only==1: 
                    text_to_print_eval_only += text_to_print + "\n"*2

                # Save epoch losses of any phase to tensorboard
                if eval_only==0:
                    if args.lastEpochSupOnly not in [1, 3]:
                        save_epoch_losses_to_tensorboard(epoch, phase, writer, epoch_losses, args)

                    if (isnan(epoch_losses['loss_xyz_coord_only']) == True) or (isnan(epoch_losses['loss_uv_coord_only']) == True) or (isnan(epoch_losses['loss_D_coord_only']) == True):
                        print("Not running more phases within the current epoch.")
                        break
            if eval_only==0:
                if epoch==args.num_epochs-1 and args.lastEpochSupOnly==2:
                    if phase=='train':
                        model_wts_last_epoch_before_NonAnnot = copy.deepcopy(model.state_dict())    
                    elif phase=='train_nonAnnot' and train_vs_eval_seq==1:
                        loss_before_last_nonAnnot = epoch_losses['loss_' + args.str_coord + '_coord_only']
                        print('\n' + args.str_coord + ' loss on train_nonAnnot before last self-supervised epoch:', end=' ')
                        print('{:4f}\n'.format(loss_before_last_nonAnnot), "\n")
                    elif phase=='train_nonAnnot' and train_vs_eval_seq==3:
                        loss_after_last_nonAnnot = epoch_losses['loss_' + args.str_coord + '_coord_only']
                        print('\n' + args.str_coord + ' loss on train_nonAnnot after last self-supervised epoch:', end=' ')
                        print('{:4f}\n'.format(loss_after_last_nonAnnot), "\n")
                        # Keep the best-performing model between running last epoch with only
                        # annotated data or with both annotated and non-annotated data
                        if loss_before_last_nonAnnot<loss_after_last_nonAnnot:
                            model.load_state_dict(model_wts_last_epoch_before_NonAnnot)
                            before_of_after_NonAnnot="\n"*2 + "Model performed better before the last use of non-annotated data" + "\n"*2
                        else:
                            before_of_after_NonAnnot="\n"*2 + "Model performed better after the last use of non-annotated data" + "\n"*2
                        print(before_of_after_NonAnnot)

                    train_vs_eval_seq += 1
                        
        if eval_only==0 and num_batches!=0:
            if (isnan(epoch_losses['loss_xyz_coord_only']) == True) or (isnan(epoch_losses['loss_uv_coord_only']) == True):
                print("No running more epochs.")
                break

    if eval_only==0:
        if num_batches!=0 and ((isnan(epoch_losses['loss_xyz_coord_only']) == True) or (isnan(epoch_losses['loss_uv_coord_only']) == True) or (isnan(epoch_losses['loss_D_coord_only']) == True)):
            print("NaN")
            last_loss_coord_only_val = float('Inf')
            save_Nan(saving_directory)
        else:
            # Save last coordinate-only validation loss
            last_loss_coord_only_val = epoch_losses['loss_' + args.str_coord + '_coord_only']

            print()

            time_elapsed = time.time() - since # in seconds
            days, hours, minutes, seconds = functions_train.seconds_to_dhms(time_elapsed)
            objective = 'Training' if eval_only==0 else 'Evaluation'
            text_to_save = objective + ' complete in {:.0f}d {:.0f}h {:.0f}m {:.0f}s\n'.format(days, hours, minutes, seconds) 
            text_to_save += 'Last ' + args.str_coord + ' validation loss: {:4f}\n'.format(last_loss_coord_only_val)
            text_to_save += "\n" + "Last validation losses:\n" + text_to_print
            if args.lastEpochSupOnly==2:
                text_to_save += before_of_after_NonAnnot
            print(text_to_save)
            file_name = os.path.join(saving_directory, 'time_and_best_losses.txt')
            with open(file_name, 'w') as x_file:
                x_file.write(text_to_save)
            # Save model
            #     if args.hyperpar_option==0 or args.neighb_dist_weight!=0 or args.normals!=0:
            #         save_models(saving_directory, best_model, n_round=args.round)
            if args.hyperpar_option==0 or args.save_all_models==1:
                save_models(saving_directory, model, n_round=args.round)

            # Save train/val/test loss of coordinates only
            if args.save_losses!=0:
                functions_train.evaluate_model(model=model, saving_directory=saving_directory, dataloaders=dataloaders, args=args)
        return model, last_loss_coord_only_val 
    else:
        time_elapsed = time.time() - since # in seconds
        days, hours, minutes, seconds = functions_train.seconds_to_dhms(time_elapsed)
        text_to_save = 'Training complete in {:.0f}d {:.0f}h {:.0f}m {:.0f}s\n'.format(days, hours, minutes, seconds) 
        text_to_save += "\n" + text_to_print_eval_only
        file_name = os.path.join(saving_directory, 'time_and_evaluated_losses_' + args.sequence_name + '.txt')
        with open(file_name, 'w') as x_file:
            x_file.write(text_to_save)

    







###
### Optimize hyperparameters function
###
def optimize_hyperpar(args, model,
                      saving_directory, criterion=None,
                      hyperpar_values = 
                      {'lr': [0.01, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                       'neighb_dist_weight': [100, 10, 1, 1/3, 0.01]}):
    """
    Optimize the hyperparameters

    Input:
    hyperpar_values is a dictionary where each key is the name of a hyperparameter name
    and the associated value is the list of values to try on that hyperparameter.

    Output:
    I will combine the values of the different hyperparameters at random, rather than grid-wise (Cartesian product style). 
    The hyperparameter with the highest number n of values to try will see all its values tries exactly once, 
    and the rest will be used as many times we need to end up going over a total of n runs of training.
    """
    since = time.time()

    print('Before shuffling:', hyperpar_values)
    i=0
    for key in hyperpar_values:
        random.seed(i)
        i+=1
        random.shuffle(hyperpar_values[key])

    print('After shuffling: ', hyperpar_values)
    highest_num_values_to_try = max([len(value) for value in hyperpar_values.values()])
    print('highest_num_values_to_try:', highest_num_values_to_try)
    text_to_save = ""
    best_model_wts_exists=0
    for i in range(highest_num_values_to_try):
        # Directory for tensorboard results
        hyperpars = ''
        if 'lr' in hyperpar_values:
            args.lr = hyperpar_values['lr'][i%len(hyperpar_values['lr'])]
            hyperpars += 'lr'+str(args.lr)+'_'
        if 'momentum' in hyperpar_values:
            args.momentum = hyperpar_values['momentum'][i%len(hyperpar_values['momentum'])]
            hyperpars += 'mom'+str(args.momentum)+'_'
        if 'gamma' in hyperpar_values:
            args.gamma = hyperpar_values['gamma'][i%len(hyperpar_values['gamma'])]
            hyperpars += 'gamma'+str(args.gamma)+'_'
        if 'step_size' in hyperpar_values:
            args.step_size = hyperpar_values['step_size'][i%len(hyperpar_values['step_size'])]
            hyperpars += 'stepSize'+str(args.step_size)+'_'
        if 'neighb_dist_weight' in hyperpar_values:
            args.neighb_dist_weight = hyperpar_values['neighb_dist_weight'][i%len(hyperpar_values['neighb_dist_weight'])]
            hyperpars += 'Wgeo'+str(args.neighb_dist_weight)+'_'
        if 'w_uv' in hyperpar_values:
            args.w_uv = hyperpar_values['w_uv'][i%len(hyperpar_values['w_uv'])]
            hyperpars += 'Wuv'+str(args.w_uv)+'_'
        if 'w_D' in hyperpar_values:
            args.w_D = hyperpar_values['w_D'][i%len(hyperpar_values['w_D'])]
            hyperpars += 'WD'+str(args.w_D)+'_'
        if 'w_xyz' in hyperpar_values:
            args.w_xyz = hyperpar_values['w_xyz'][i%len(hyperpar_values['w_xyz'])]
            hyperpars += 'Wxyz'+str(args.w_xyz)+'_'
        if 'w_chamfer_GT_pred' in hyperpar_values:
            args.w_chamfer_GT_pred = hyperpar_values['w_chamfer_GT_pred'][i%len(hyperpar_values['w_chamfer_GT_pred'])]
            hyperpars += 'WchamferGTpred'+str(args.w_chamfer_GT_pred)+'_'
        if 'w_chamfer_pred_GT' in hyperpar_values:
            args.w_chamfer_pred_GT = hyperpar_values['w_chamfer_pred_GT'][i%len(hyperpar_values['w_chamfer_pred_GT'])]
            hyperpars += 'WchamferpredGT'+str(args.w_chamfer_pred_GT)+'_'
        if 'w_chamfer_GTcontour_pred' in hyperpar_values:
            args.w_chamfer_GTcontour_pred = hyperpar_values['w_chamfer_GTcontour_pred'][i%len(hyperpar_values['w_chamfer_GTcontour_pred'])]
            hyperpars += 'WchamferGTcontourpred'+str(args.w_chamfer_GTcontour_pred)+'_'
        if 'w_chamfer_pred_GTcontour' in hyperpar_values:
            args.w_chamfer_pred_GTcontour = hyperpar_values['w_chamfer_pred_GTcontour'][i%len(hyperpar_values['w_chamfer_pred_GTcontour'])]
            hyperpars += 'WchamferpredGTcontour'+str(args.w_chamfer_pred_GTcontour)+'_'

        # New loss weights
        if 'loss_w_chamfer_GT_pred' in hyperpar_values:
            args.loss_w_chamfer_GT_pred = hyperpar_values['loss_w_chamfer_GT_pred'][i%len(hyperpar_values['loss_w_chamfer_GT_pred'])]
            hyperpars += 'WchamferGTpred'+str(args.loss_w_chamfer_GT_pred)+'_'
        if 'loss_w_chamfer_pred_GT' in hyperpar_values:
            args.loss_w_chamfer_pred_GT = hyperpar_values['loss_w_chamfer_pred_GT'][i%len(hyperpar_values['loss_w_chamfer_pred_GT'])]
            hyperpars += 'WchamferpredGT'+str(args.loss_w_chamfer_pred_GT)+'_'
        if 'loss_w_chamfer_GTcontour_pred' in hyperpar_values:
            args.loss_w_chamfer_GTcontour_pred = hyperpar_values['loss_w_chamfer_GTcontour_pred'][i%len(hyperpar_values['loss_w_chamfer_GTcontour_pred'])]
            hyperpars += 'WchamferGTcontourpred'+str(args.loss_w_chamfer_GTcontour_pred)+'_'
        if 'loss_w_chamfer_pred_GTcontour' in hyperpar_values:
            args.loss_w_chamfer_pred_GTcontour = hyperpar_values['loss_w_chamfer_pred_GTcontour'][i%len(hyperpar_values['loss_w_chamfer_pred_GTcontour'])]
            hyperpars += 'WchamferpredGTcontour'+str(args.loss_w_chamfer_pred_GTcontour)+'_'
        if 'loss_w_geo' in hyperpar_values:
            args.loss_w_geo = hyperpar_values['loss_w_geo'][i%len(hyperpar_values['loss_w_geo'])]
            hyperpars += 'Wgeo'+str(args.loss_w_geo)+'_'
        if 'loss_w_horizConsecEdges' in hyperpar_values:
            args.loss_w_horizConsecEdges = hyperpar_values['loss_w_horizConsecEdges'][i%len(hyperpar_values['loss_w_horizConsecEdges'])]
            hyperpars += 'WhorizEdges'+str(args.loss_w_horizConsecEdges)+'_'
        if 'loss_w_verConsecEdges' in hyperpar_values:
            args.loss_w_verConsecEdges = hyperpar_values['loss_w_verConsecEdges'][i%len(hyperpar_values['loss_w_verConsecEdges'])]
            hyperpars += 'WverEdges'+str(args.loss_w_verConsecEdges)+'_'
        if 'loss_w_uv' in hyperpar_values:
            args.loss_w_uv = hyperpar_values['loss_w_uv'][i%len(hyperpar_values['loss_w_uv'])]
            hyperpars += 'Wuv'+str(args.loss_w_uv)+'_'
        if 'loss_w_D' in hyperpar_values:
            args.loss_w_D = hyperpar_values['loss_w_D'][i%len(hyperpar_values['loss_w_D'])]
            hyperpars += 'WD'+str(args.loss_w_D)+'_'
        if 'loss_w_xyz' in hyperpar_values:
            args.loss_w_xyz = hyperpar_values['loss_w_xyz'][i%len(hyperpar_values['loss_w_xyz'])]
            hyperpars += 'Wxyz'+str(args.loss_w_xyz)+'_'
        if 'loss_factor' in hyperpar_values:
            args.loss_factor = hyperpar_values['loss_factor'][i%len(hyperpar_values['loss_factor'])]
            hyperpars += 'lossF'+str(int(args.loss_factor))+'_'

#             l=len(hyperpars)
        if hyperpars[-1]=='_': hyperpars=hyperpars[:-1]

        # Create the directory for tensorboard results
        current_saving_directory = os.path.join(saving_directory, hyperpars)
        print('Training model in:', current_saving_directory)
        writer = SummaryWriter(current_saving_directory) # This creates the directory if it doesn't exist
        
        args = functions_data_processing.weight_processing(args)

        if args.frozen_resnet==1: # Only the parameters of the final layer are being optimized
            optimizer_conv = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
        else: # All parameters are being optimized
            optimizer_conv = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Multiply learning rate by gamma every step_size epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.step_size, gamma=args.gamma)

        if criterion is not None:
            model_after_last_epoch, last_loss_coord_only_val = train_model(
                model=model, optimizer=optimizer_conv, scheduler=exp_lr_scheduler,
                saving_directory=current_saving_directory, writer=writer, criterion=criterion,
                dataloaders=dataloaders, args=args, kwargs_normalize_labels_pred=kwargs_normalize_labels_pred)
        # Add current validation loss to be saved later for comparison between hyperparameter choices
        text_to_save += hyperpars + ': {:4f}\n'.format(last_loss_coord_only_val)

        # Keep the model with the best validation loss so far
        if (isnan(last_loss_coord_only_val) == False) and (i==0 or last_loss_coord_only_val < best_loss_val):
            best_loss_val = last_loss_coord_only_val
            best_model_wts = copy.deepcopy(model_after_last_epoch.state_dict()) 
            best_hyperpars = hyperpars
            best_model_wts_exists=1
        elif (isnan(last_loss_coord_only_val) == True) and i==0:
            best_loss_val = float('Inf')
            best_hyperpars = hyperpars
        print('\nBest ' + args.str_coord + ' validation loss so far: {:4f}\n'.format(best_loss_val), "\n")
        print('Best hyperparameters so far: ' + best_hyperpars + '\n')

    time_elapsed = time.time() - since # in seconds
    days, hours, minutes, seconds = functions_train.seconds_to_dhms(time_elapsed)
       
    # Save the model with the best validation loss so far
    if best_model_wts_exists==1:
        model_after_last_epoch.load_state_dict(best_model_wts)
        save_models(saving_directory=saving_directory, model=model_after_last_epoch, n_round = args.round)
   
    text_to_save = 'Hyperparameter optimization complete in {:.0f}d {:.0f}h {:.0f}m {:.0f}s\n'.format(days, hours, minutes, seconds) 
    if best_model_wts_exists==1:
        text_to_save += 'Best ' + args.str_coord + ' validation loss: {:4f}\n'.format(best_loss_val)
        text_to_save += 'Best hyperparameters: ' + best_hyperpars + '\n'
    if args.round!=1:
        text_to_save += 'Round ' + str(args.round-1) + ' model:\n'
        text_to_save += args.round1_model_dir
    file_name = 'time_and_best_losses_opt.txt' if args.round==1 else 'time_and_best_losses_opt_rd'+str(args.round)+'.txt'
    file_name = os.path.join(saving_directory, file_name)
    print(text_to_save)
    with open(file_name, 'w') as x_file:
        x_file.write(text_to_save)

        
        
        
        
        


###
### Create the network
###
if __name__=='__main__':
    if args.round==1:
        # Load a pretrained ResNet
        if args.resnet_version == 18:
    #         model = torchvision.models.resnet18(pretrained=True) # No dropout
            # using my code for introducing dropout and alternative activation functions to ResNet18:
            model = resnet_including_dropout.resnet18(pretrained=True, dropout_p=args.dropout_p, relu_alternative=args.relu_alternative) 

        elif args.resnet_version == 152:
            model = torchvision.models.resnet152(pretrained=True)

        # If args.frozen_resnet==1: The parameters of the ResNet will not be optimized
        # If args.frozen_resnet==0: The parameters of the ResNet will be optimized
        if args.frozen_resnet==1: # Use ResNet as fixed feature extractor only. I.e., freeze its parameters during training
            # Freeze all the ResNet network 
            # by setting ``requires_grad == False``, 
            # so that the gradients are not computed in ``backward()``.
            #
            # More on this, here: <br>
            # <http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>. 
            for param in model.parameters():
                param.requires_grad = False
        else: # Finetune the ResNet. I.e., do not freeze the parameters, but train with them
            for param in model.parameters():
                param.requires_grad = True

        if args.verbose==1: 
            print('Architecture of this ResNet before changing the final FC layer:')
            print(model.parameters)

        # # This does NOT show the total number of parameters:
        # num_parameters_model = sum(1 for _ in model.parameters())
        # print('ResNet' + str(args.resnet_version) + ' does NOT consist of ' + str(num_parameters_model) + ' parameters.')

        num_ftrs = model.fc.in_features # number of inputs in the last layer of the ResNet
        if args.verbose==1: 
            print('Number of features extracted by ResNet' + str(args.resnet_version) + ':', num_ftrs)
            print()
            # Number of features extracted by ResNet18: 512
            # Number of features extracted by ResNet152: 2048

        # Change the last layer.
        # It will still be a fully connected later with num_ftrs inputs, 
        # but we will change the number of outputs.
        # Parameters of newly constructed modules have requires_grad=True by default
        #
        # To predict the uv, xyz or uvycoordinates of the selected vertices      

        if args.verbose==1:
            print("BEFORE CHANGING OUTPUT LAYER")
            print("weights of conv1 layer:", model.state_dict()['conv1.weight'][0, 0], "\n")
            print("weights of FC layer:", model.state_dict()['fc.weight'], "\n")
        model.fc = nn.Linear(num_ftrs, args.num_selected_vertices*args.num_coord_per_vertex) 
        if args.verbose==1:
            print("AFTER CHANGING OUTPUT LAYER")
            print("weights of conv1 layer:", model.state_dict()['conv1.weight'][0, 0], "\n")
            print("weights of FC layer:", model.state_dict()['fc.weight'], "\n")
            print("Note that the output FC layer weights change but the weights of other layers do not\n")
    #     model(inputs) has the form:
    #     [u_0, v_0, ..., u_i, v_i], if args.predict_uv_or_xyz == 'uv'
    #     [x_0, y_0, z_0, ..., x_i, y_i, z_i], if args.predict_uv_or_xyz == 'xyz'
    #     [u_0, v_0, y_0, ..., u_i, v_i, y_i], if args.predict_uv_or_xyz == 'uvy'

        if args.verbose==1: 
            print('Architecture of this ResNet after changing the final FC layer:')
            print(model.parameters, "\n")
    else:
        model_filename = os.path.join(args.round1_model_dir, 'model.pt')
    #     model = torch.load(model_filename, map_location='cpu')
        model = torch.load(model_filename, map_location=args.device)
        
        if args.freeze_all_but_last_layer==1:
            # Number of layers
            num_layers = 0
            for child in model.children():
                num_layers+=1
            print("\n"*3, "Number of layers:", num_layers, "\n"*3) # 11

            # Freeze parameters up to (but not including) layer lt
            print("\nFreezing all layers but the last...\n")
            lt=num_layers
            cntr=0
            for child in model.children():
                cntr+=1
                if cntr < lt:
                    print(child)
                    for param in child.parameters():
                        param.requires_grad = False

            # Show which parameters have gradient tracking
            if args.verbose==1:
                print("\n"*5, "params with NO grad", "\n"*5)
                for param in model.parameters():    
                    if param.requires_grad == False:
                        print(param)
                print("\n"*5, "params with grad", "\n"*5)
                for param in model.parameters():    
                    if param.requires_grad == True:
                        print(param)

    
    model = model.to(args.device)

    # dtype of the weights of the net (default: torch.float, which is the same as torch.float32)
    if args.dtype==1: model.double()
    # for param in model.parameters():
    #     print(param.dtype)

    # ### Trying to add Dropout. Now I include it directly when creating the network.
    # if args.dropout_p!=0:
    #     import dropout_code

    if args.verbose==1:
        functions_train.show_architecture_and_forward_pass(model, dataloaders, args, kwargs_normalize=kwargs_normalize_labels_pred)  

    # Visualize weights
    # print("weights")
    # for key, value in model.state_dict().items():
    #     print(key, value, "\n")
    
    
###
### Set training options
###
if __name__=='__main__':
    criterion = nn.MSELoss()

    if args.frozen_resnet==1: # Only the parameters of the final layer are being optimized
        optimizer_conv = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    else: # All parameters are being optimized
        optimizer_conv = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Multiply learning rate by gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.step_size, gamma=args.gamma)

    if args.testing_eval == 2: # Freeze some of the batch normalization parameters learned within the ResNet during training
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
    #                 if args.verbose==1:
    #                     print('m.parameters():')
    #                     print(m.parameters())
                if args.verbose==1:
                    i=0
                    print(i)
                for p in m.parameters():
                    p.requires_grad = False
                    if args.verbose==1:
                        i+=1
                        print(i)
    #                     if args.verbose==1:
    #                         print('p:')
    #                         print(p)
                        # The prints show that there are about two values p within m.parameters() for each value of m

    # # Show network architecture
    # if args.verbose==1:
    #     print('Architecture of the ResNet with the last fully connected layer adapted:')
    #     print(model)




###
### Train and evaluate
###
if __name__=='__main__':
    if args.hyperpar_option==0:
        # Set tensorboard directory
        writer = SummaryWriter(saving_directory) # This creates the directory if it doesn't exist

        args = functions_data_processing.weight_processing(args)

        print('Training model in:', saving_directory)
        if args.num_epochs==0:
            save_models(saving_directory, model)
        elif args.testing_loss_computations==0:
                model_after_last_epoch, last_loss_coord_only_val = train_model(
                    model=model, optimizer=optimizer_conv, scheduler=exp_lr_scheduler, saving_directory=saving_directory, 
                    writer = writer, criterion=criterion, dataloaders=dataloaders, args=args,
                    kwargs_normalize_labels_pred=kwargs_normalize_labels_pred)
        elif args.testing_loss_computations==1:
            functions_train.check_loss_computations(model, optimizer_conv,
                                                    exp_lr_scheduler, saving_directory=saving_directory, writer=writer, 
                                                    testing_eval=args.testing_eval, predict_uv_or_xyz=args.predict_uv_or_xyz, 
                                                    dataloaders=dataloaders, device=args.device, batch_size=args.batch_size,
                                                    num_selected_vertices=args.num_selected_vertices,
                                                    choice_of_loss=args.loss, dataset_sizes=dataset_sizes,
                                                    num_epochs=args.num_epochs, criterion=criterion)
    else:
        hyperpar_values = functions_train.hyperpars_to_try(args.hyperpar_option)        

        optimize_hyperpar(args=args, model=model, saving_directory=saving_directory, 
                          criterion=criterion, hyperpar_values=hyperpar_values)