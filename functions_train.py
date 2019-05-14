
# coding: utf-8

# **Author**: `Francisco Belchí <frbegu@gmail.com>, <https://github.com/KikoBelchi/2d_to_3d>`_



###
### Imports
###
from collections import OrderedDict
import numpy as np
import os
import pickle
import random
import scipy.spatial
import time
import torch
import torch.nn as nn

import binary_image
import data_loading
import distance_matrix_4_tensors
import find_and_draw_contours
import functions_data_processing
import loss_functions
import submesh
import train
   

###
### Directory where tensorboard visualizations and the trained models will sit
###
def trained_model_directory(args):
    model_directory = '' if args.predict_uv_or_xyz=='xyz' else args.predict_uv_or_xyz
    model_directory += 'Epochs' + str(args.num_epochs)
    model_directory += '_' + args.sequence_name + args.dataset_number
    if args.texture_type!='train_non-text' and args.texture_type!='':
        model_directory += '_' + args.texture_type.split('_')[1]
    elif args.texture_type=='':
        model_directory += '_allText'
    
    if args.reordered_dataset == 0:
        model_directory += '_' + str(args.num_selected_vertices) + 'v'
    elif args.reordered_dataset == 1:
        model_directory += '_' + str(args.submesh_num_vertices_vertical) + 'x' + str(args.submesh_num_vertices_horizontal) + 'mesh'

    if args.camera_coordinates == 1:
        model_directory += '_cameraCoord'
    else:
        model_directory += '_worldCoord'

    if args.normals == 1:
        model_directory += '_normals'
    else:
        model_directory += '_noNormals'
        
    model_directory += '_ResNet' + str(args.resnet_version)
    if args.relu_alternative!='relu':
        model_directory += '_' + args.relu_alternative
    
    if args.crop_centre_or_ROI==0: # centre crop
        model_directory += '_centreCrop'
    elif args.crop_centre_or_ROI==1: # Squared box containing the towel
        model_directory += '_ROI'
    elif args.crop_centre_or_ROI==2: # Rectangular box containing the towel
        model_directory += '_rectROI'
    elif args.crop_centre_or_ROI==3: # No crop
        model_directory += '_noCrop'
    
    if args.batch_size!=4:
        model_directory += '_batchSize' + str(args.batch_size)
        
    if args.frozen_resnet==1:
        model_directory += '_notFinetuned'
    
    if args.hyperpar_option==0:
        if args.lr!=0.001:
            model_directory += '_lr' + str(args.lr)
        if args.momentum!=0.9:
            model_directory += '_mom' + str(args.momentum)
        if args.gamma!=0.1:
            model_directory += '_gam' + str(args.gamma)
        if args.step_size!=7:
            model_directory += '_stepSize' + str(args.step_size)
        # Uncomment this to run visualization of models trained before 01/02/19
    #     if args.w_coord!=1:
    #         model_directory += '_coord' +str(args.w_coord)
        # Comment this to run visualization of models trained before 01/02/19
    if args.loss_weights==0:
        if args.w_uv!=0:
            model_directory += '_Wuv' +str(args.w_uv)
        if args.w_xyz!=0:
            model_directory += '_Wxyz' +str(args.w_xyz)
        if args.neighb_dist_weight!=0:
            model_directory += '_Wgeo' +str(args.neighb_dist_weight)
#             model_directory += '_neigbDistW' +str(args.neighb_dist_weight)
    else:
        if args.loss_w_uv!=0:
            model_directory += '_Wuv' +str(args.loss_w_uv)
        if args.loss_w_xyz!=0:
            model_directory += '_Wxyz' +str(args.loss_w_xyz)
        if args.loss_w_D!=0:
            model_directory += '_WD' +str(args.loss_w_D)         
        if args.loss_w_geo!=0:
            model_directory += '_Wgeo' +str(args.loss_w_geo)
        if args.loss_w_horizConsecEdges!=0:
            model_directory += '_WhorizEdges' +str(args.loss_w_horizConsecEdges)
        if args.loss_w_verConsecEdges!=0:
            model_directory += '_WverEdges' +str(args.loss_w_verConsecEdges)
        
    if args.loss_diff_Hauss!=0:
        model_directory += '_diff'
    else: 
        model_directory += '_'
    hausdorff_chamfer = 'Haus' if args.hausdorff==1 else 'Chamf'
    model_directory += hausdorff_chamfer
    if args.n_outputs==2: model_directory += 'IndepW' # independent weights
    if args.loss_weights==0 and (args.w_chamfer_GT_pred!=0 or args.w_chamfer_pred_GT!=0 or args.w_chamfer_GTcontour_pred!=0 or args.w_chamfer_pred_GTcontour!=0):
        model_directory += '_' + str(args.w_chamfer_GT_pred) + '_' + str(args.w_chamfer_pred_GT)
        model_directory += '_' + str(args.w_chamfer_GTcontour_pred) + '_' + str(args.w_chamfer_pred_GTcontour)
    if args.loss_weights==1 and (args.loss_w_chamfer_GT_pred!=0 or args.loss_w_chamfer_pred_GT!=0 or args.loss_w_chamfer_GTcontour_pred!=0 or args.loss_w_chamfer_pred_GTcontour!=0):
        model_directory += '_' + str(args.loss_w_chamfer_GT_pred) + '_' + str(args.loss_w_chamfer_pred_GT)
        model_directory += '_' + str(args.loss_w_chamfer_GTcontour_pred) + '_' + str(args.loss_w_chamfer_pred_GTcontour)
            
    if args.loss_weights==1: model_directory += '_lossW'
    if args.loss_factor!=1.: model_directory += '_lossF' + str(int(args.loss_factor)) 
    if args.normWeights1!=1: model_directory += 'unnor'
        
    if args.uv_normalization==1:
        model_directory += '_uvNorm'
    if args.uv_normalization==2:
        model_directory += '_uvNorm224'
    if args.new_uv_norm!=0:
        model_directory += '_newuvNorm' + str(args.new_uv_norm)
#     if not(args.lengths_proportion_test is None):
#         model_directory += '_alsoTest'
    if args.lengths_proportion_train!=0.8:
        model_directory += '_train' + str(args.lengths_proportion_train)
    if args.lengths_proportion_train_nonAnnotated is not None:
        model_directory += '_nonAnn' + str(args.lengths_proportion_train_nonAnnotated)
#         if args.train_w_annot_only==1 and args.round!=1:
        if args.train_w_annot_only==1:
            pass
        else:
            model_directory += '_both'
    if args.testing_eval!=0:
        model_directory += '_eval' + str(args.testing_eval)
    if args.hyperpar_option!=0:
        model_directory += '_hyper' + str(args.hyperpar_option)
    if args.unsupervised==1:
        model_directory += '_unsup'  
    if args.uv_loss_only==1:
        model_directory += '_2Dloss'
        
    if args.subsample_ratio!=1:
        model_directory += '_subsample' + str(args.subsample_ratio)
    if args.subsample_ratio_contour!=1:
        model_directory += '_subsCont' + str(args.subsample_ratio_contour)
        
    model_directory += '' if args.round==1 else '_rd'+str(args.round)
    model_directory += '' if args.permutation_dir is None else '_permOutLayer'
    
    if args.GTxyz_from_uvy==1: model_directory += '_GTfromuvy'
    if args.dropout_p!=0: model_directory += '_DO' + str(args.dropout_p)
        
    if args.GTuv!=0:  model_directory += '_GTuv'
        
    if args.append_to_model_dir!=None:
        model_directory += args.append_to_model_dir
    if args.dtype==1:
        model_directory += "_double"
    return model_directory

def save_predictions(phase, epoch, labels, outputs, loss, sample_batched_img_name, saving_directory='.', n_round=1):       
    postfix = '_' + str(phase) + '_epoch' + str(epoch)
#     postfix += '' if n_round==1 else '_rd'+str(n_round)
    postfix += '.pt'
    
    # Save ground truth
    filename = os.path.join(saving_directory, 'labels' + postfix)
    torch.save(labels, filename)
    
    # Save prediction
    filename = os.path.join(saving_directory, 'outputs' + postfix)
    torch.save(outputs, filename)
    
    # Save loss
    filename = os.path.join(saving_directory, 'loss' + postfix)
    torch.save(loss, filename)
    
    # Save list of image names
    filename = os.path.join(saving_directory, 'img_name' + postfix)
    with open(filename, 'wb') as fp:
        pickle.dump(sample_batched_img_name, fp)
    
def load_predictions(phase, epoch, saving_directory='.'):       
    postfix = '_' + str(phase) + '_epoch' + str(epoch) + '.pt'
    
    # Load ground truth
    filename = os.path.join(saving_directory, 'labels' + postfix)
    labels = torch.load(filename, map_location='cpu')
    
    # Load prediction
    filename = os.path.join(saving_directory, 'outputs' + postfix)
    outputs = torch.load(filename, map_location='cpu')
    
    # Load loss
    filename = os.path.join(saving_directory, 'loss' + postfix)
    loss = torch.load(filename, map_location='cpu')
    
    # Load list of image names
    filename = os.path.join(saving_directory, 'img_name' + postfix)
    with open(filename, 'rb') as fp:
        sample_batched_img_name = pickle.load(fp)
    
    return labels, outputs, loss, sample_batched_img_name

def check_loss_computations(model, optimizer, scheduler, saving_directory, writer, testing_eval,
                            predict_uv_or_xyz, dataloaders, device, batch_size, num_selected_vertices,
                            choice_of_loss, dataset_sizes,
                            num_epochs=25, criterion=None):
    """ Function to check whether the loss computations are correct"""
    since = time.time()

    best_loss_val = 10000000.0 # Initialize with a large number
    best_loss_train = 10000000.0 # Initialize with a large number

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print()
            print('phase:', phase)
            if phase == 'train':
                scheduler.step()
                if testing_eval == 0:
                    # Set model to training mode
                    model.train()
                elif testing_eval == 1:
                    # Set model to evaluate mode. Hence, some possible BatchNorm or Dropout may be omitted.
                    model.eval()   
                elif testing_eval == 2: 
                    # Set model to training mode but freeze BatchNorm weights 
                    # The freezing is done beforehand, when defining the network architecture
                    model.train()   
            else:
                # Set model to evaluate mode
                model.eval()   

            running_loss = 0.0
            running_loss_myown = 0.0
#             running_corrects = 0

            # Iterate over data
            for batch_idx, sample_batched in enumerate(dataloaders[phase], 1):
                inputs = sample_batched['image']
                inputs = inputs.to(device)
                if predict_uv_or_xyz=='xyz':
                    if normals == 1:
                        # Shape batch_size x (num_selected_vertices*2) x 3:
                        labels = torch.cat((sample_batched['xyz'], sample_batched['normal_coordinates']), 1) 
                        # Shape batch_size x (num_selected_vertices * 6):
                        labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices*2) 
                    else:
                        # Shape batch_size x num_selected_vertices x 3:
                        labels = sample_batched['xyz'] 
                        # Shape batch_size x (num_selected_vertices * 3):
                        labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices) 
                else:
                    # Shape batch_size x num_selected_vertices x 2:
                    labels = sample_batched['uv']
                    # Shape batch_size x (num_selected_vertices * 2):
                    labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices, num_coord_per_vertex=2) 
                labels = labels.to(device)
                if args.dtype==1: labels=labels.double()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):


                    ###
                    ### The line for testing the losses:
                    ###
                    outputs = labels+2




                    if choice_of_loss == 1: # my own MSE loss
                        loss_myown = loss_functions.MSE_loss(labels, outputs, batch_size=batch_size, num_selected_vertices=num_selected_vertices)
                        print('loss_myown:', loss_myown)
                    # torch.nn MSELoss
                    loss = criterion(outputs, labels) 
                    print('loss:', loss)

                if choice_of_loss == 1: # my own MSE loss
                    running_loss_myown += loss_myown
                    print('running_loss_myown:', running_loss_myown)
                running_loss += loss.item()
                print('running_loss:', running_loss)
#                 # loss.item() is the average (at least with MSE) of 
#                 # the losses of the observations in the batch.
#                         At the end of the epoch, running_loss will be the sum of loss.item() for all the batches.
#                         Then, we will define the epoch loss as the average of the MSE loss of each observation,
#                         which equals 
#                         running_loss * batch_size / training_size,
#                         or equivalently, 
#                         running_loss / (number of batches in which the training data is split),
#                         or the analogue with validation set size if phase=='val' 

            print('len(dataloaders[phase]):', len(dataloaders[phase]))
            print('inputs.size(0), which should be the batch_size:', inputs.size(0))
            print("dataset_sizes['train'] / batch_size:", dataset_sizes['train'] / inputs.size(0))
            epoch_loss = running_loss / len(dataloaders[phase]) # Average of the MSE of each observation
            print('epoch_loss:', epoch_loss)
            if choice_of_loss == 1: # my own MSE loss
                epoch_loss_myown = running_loss_myown / len(dataloaders[phase]) # Average of the MSE of each observation
                print('epoch_loss_myown:', epoch_loss_myown)
#                         Remember: 
#                         dataset_sizes['train'] == training_size
#                         len(dataloaders['train']) == training_size / batch_size, i.e.,
#                         len(dataloaders['train']) == number of batches in which the training data is split
#                         inputs.size(0) == batch_size

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if choice_of_loss == 1: # my own MSE loss
                print('{} My Loss: {:.4f}'.format(phase, epoch_loss_myown))
#                 I removed epoch_acc from printing

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


###
### Predict the output and loss of a batch
###
# def predict_and_loss_old(model, sample_batched, normals, num_selected_vertices, device, choice_of_loss=0, verbose=0,
#                      predict_uv_or_xyz='xyz'):
#     if predict_uv_or_xyz=='xyz':
#         num_coord_per_vertex=3
#     else:
#         num_coord_per_vertex=2
        
#     with torch.no_grad():
#         inputs = sample_batched['image']
#         inputs = inputs.to(device)
#         outputs = model(inputs)
#         batch_size = sample_batched['image'].shape[0]

#         if verbose==1:
#             print('Output of the batch:')
#             print(outputs)
#             print("batch_size=", sample_batched['image'].shape[0])
            
#         if predict_uv_or_xyz=='xyz':
#             if normals == 1:
#                 # Shape batch_size x (num_selected_vertices*2) x 3
#                 labels = torch.cat((sample_batched['xyz'], sample_batched['normal_coordinates']), 1) 

#                 # Shape batch_size x (num_selected_vertices * 6)
#                 labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices*2) 
#             else:
#                 # Shape batch_size x num_selected_vertices x 3
#                 labels = sample_batched['xyz'] 

#                 # Shape batch_size x (num_selected_vertices * 3)
#                 labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices) 
#         else:
#             # Shape batch_size x num_selected_vertices x 2
#             labels = sample_batched['uv']

#             # Shape batch_size x (num_selected_vertices * 2)
#             labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices, num_coord_per_vertex=2) 

#         labels = labels.to(device)
#         if args.dtype==1: labels=labels.double()

#     #     criterion = nn.MSELoss(reduce=False)
#         criterion = nn.MSELoss()
#     #     print(criterion.reduction)
#         # Since 'criterion.reduction == 'elementwise_mean'', 
#         # the resulting loss below is the average of the losses of the batch, 
#         # instead of 1 loss per element in the batch.

#         loss = criterion(outputs, labels)
#         if choice_of_loss == 1: # my own MSE loss
#             loss_myown = loss_functions.MSE_loss(labels, outputs, batch_size=batch_size, num_selected_vertices=num_selected_vertices)
        
#         print('Loss of the batch:')
#         print(loss)
#     #     print(loss.item())
#     #     print(loss.data)
#     #     print(loss.data[0])
#         if choice_of_loss == 1: # my own MSE loss
#             print('My Loss of the batch:')
#             print(loss_myown)
#         if normals == 1:
#             # Shape back to batch_size x (num_selected_vertices*2) x num_coord_per_vertex
#             labels = functions_data_processing.reshape_labels_back(labels, batch_size, num_selected_vertices*2, num_coord_per_vertex=num_coord_per_vertex) 
#             outputs = functions_data_processing.reshape_labels_back(outputs, batch_size, num_selected_vertices*2, num_coord_per_vertex=num_coord_per_vertex) 
#         else:
#             # Shape back to batch_size x num_selected_vertices x num_coord_per_vertex
#             labels = functions_data_processing.reshape_labels_back(labels, batch_size, num_selected_vertices, num_coord_per_vertex=num_coord_per_vertex) 
#             outputs = functions_data_processing.reshape_labels_back(outputs, batch_size, num_selected_vertices, num_coord_per_vertex=num_coord_per_vertex) 
#         return labels, outputs

# if __name__ == '__main__':
#     print('Output and Loss on a validation batch.')
#     sample_batched = next(iter(dataloaders['val']))
#     predict_and_loss(model = model_best_wrt_val_set, sample_batched = sample_batched, 
#                  normals=normals, num_selected_vertices=num_selected_vertices)
#     predict_and_loss(model = model_after_last_epoch, sample_batched = sample_batched, 
#                  normals=normals, num_selected_vertices=num_selected_vertices)

def predict_and_loss(model, sample_batched, args, kwargs_normalize):
    with torch.no_grad():
        inputs = sample_batched['image'].to(args.device)
        labels, labels_uv_coord, labels_xyz_coord, labels_D = create_labels_and_prediction(
                        args, sample_batched=sample_batched, **kwargs_normalize)
        outputs, outputs_uv_coord, outputs_xyz_coord, outputs_D = create_labels_and_prediction(
            args, sample_batched=sample_batched, GT_or_prediction='pred', model=model, inputs=inputs, **kwargs_normalize)
        if args.verbose==1:
            print('Labels of the batch:')
            print(labels, labels_uv_coord, labels_xyz_coord, labels_D)
            print('Output of the batch:')
            print(outputs, outputs_uv_coord, outputs_xyz_coord, outputs_D)
        criterion = nn.MSELoss()
#         loss = criterion(outputs, labels)
#         print('Loss of the batch:')
#         print(loss)
        if args.dropout_p==0:
            if 'uv' in args.predict_uv_or_xyz:
                loss_uv_coord_only = criterion(outputs_uv_coord, labels_uv_coord) 
                print('Loss_uv_coord_only of the batch:')
                print(loss_uv_coord_only)
            if 'y' in args.predict_uv_or_xyz or 'D' in args.predict_uv_or_xyz:
                loss_xyz_coord_only = criterion(outputs_xyz_coord, labels_xyz_coord) 
                print('Loss_xyz_coord_only of the batch:')
                print(loss_xyz_coord_only)
            if 'D' in args.predict_uv_or_xyz:
                loss_D = criterion(outputs_D, labels_D) 
                print('Loss_D of the batch:')
                print(loss_D)
    return labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord, labels_D, outputs_D
    
    
def directional_chamfer_for_numpy(pt_cld_a, pt_cld_b, args=None):
    """ For every point x in pt_cld_a, compute d_x = d(x, pt_cld_b).
    Return the sum of d_x over all x in pt_cld_a. 
    
    pt_cld_a and pt_cld_b are numpy arrays or torch tensors of shapes Mx2 and Nx2.
    
    CAVEAT: Regardless of the format of the input point clouds, the output is a numpy array of a single element.
    type=<class 'numpy.float64'>
    """
    if args is not None:
        if args.hausdorff==0:
            return np.sum(scipy.spatial.distance_matrix(pt_cld_a, pt_cld_b).min(axis=1))
        elif args.hausdorff==1:
            return np.max(scipy.spatial.distance_matrix(pt_cld_a, pt_cld_b).min(axis=1))
        
def directional_chamfer(pt_cld_a, pt_cld_b, args=None):
    """ 
    Like directional_chamfer_for_numpy(), but using tensors as input and output and throughout the process.
    For every point x in pt_cld_a, compute d_x = d(x, pt_cld_b).
    Return the sum of d_x over all x in pt_cld_a. 
    
    pt_cld_a and pt_cld_b are tensor tensors of shapes Mx2 and Nx2.
    
    The output is a tensor of a single element.
    """
    if args is not None:
        if args.hausdorff==0:
            return torch.sum(torch.min(distance_matrix_4_tensors.distance_matrix(pt_cld_a, pt_cld_b, dtype=args.dtype), dim=1)[0])
        elif args.hausdorff==1:
            return torch.max(torch.min(distance_matrix_4_tensors.distance_matrix(pt_cld_a, pt_cld_b, dtype=args.dtype), dim=1)[0])

def towel_pixels(image_path, args):
    """ Get a point cloud of uv coordinates from either the towel pixels (if args.contour==0) or the contour of such area (if args.contour==1). The input is the path of the image from which to extract the pixels. 
    
    The output has shape (M, 2). """
#             if uv_towel is None:
    # Binary image: 0 --> background and shade; 255 --> towel pixels
    binary_TowelOnly = binary_image.binary_img_from_towel_igm_name(image_path, verbose=0, sequence_name=args.sequence_name)

    if args.contour==0:
        # uv pixels occupied by towel in the RGB image
        uv_towel = binary_image.find_uv_from_binary_image(binary_TowelOnly, verbose=0, numpy_or_tensor=1, device=args.device, dtype=args.dtype) # I should not need to append .to(args.device).double(), since it's already done within find_uv_from_binary_image()
    else:
        # uv of contour of pixels occupied by towel in the RGB image
        uv_towel = find_and_draw_contours.full_tensor_of_contour_pixels_from_binary(
            binary_TowelOnly, device=args.device, verbose=0, numpy_or_tensor=1, dtype=args.dtype)
    return uv_towel

def sample_of_towel_pixels(uv_towel, args, random_seed=1):
    """ Input: a point cloud of uv coordinates from either the towel pixels (if args.contour==0) or the contour of such area (if args.contour==1). 
    Output: a subsample of this point cloud.
    
    The output has shape (M, 2). """
    if args.contour==0:
        # uv pixels occupied by towel in the RGB image
        subsample_ratio=args.subsample_ratio
        subsample_size=args.subsample_size
    else:
        # uv of contour of pixels occupied by towel in the RGB image
        subsample_ratio=args.subsample_ratio_contour
        subsample_size=args.subsample_size_contour
    if args.GTtowelPixel == 1:
        subsample_size = min(subsample_size, args.num_selected_vertices)

    if args.towelPixelSample==1:
        n_pixels_in_subsample = subsample_size 
        random.seed(random_seed)
    elif subsample_ratio!=1:
        n_pixels_in_subsample = max(min(args.num_selected_vertices, uv_towel.shape[0]), round(uv_towel.shape[0]/subsample_ratio))
        random.seed(args.num_epochs+1) 
        # By changing args.num_epochs by the current epoch, 
        # I can use a different seed at each epoch
        # to get a different random subsample at every epoch,
        # which can be a good option in order to avoid the vertices going too close to some specific pixels.
        
    if subsample_ratio!=1 or args.towelPixelSample==1:
        subsample_idx = random.sample(range(uv_towel.shape[0]), n_pixels_in_subsample)
#                 subsample_idx = [i for i in range(uv_towel.shape[0]) if i%args.subsample_ratio==0]
        uv_towel = uv_towel[subsample_idx]
    
    if args.verbose==1:
        if args.contour==0:
            print("uv_towel_subsample_shape: ", end='')
        else:
            print("uv_towel_contour_subsample_shape: ", end='')
        print(uv_towel.shape)
    return uv_towel
#     pt_cld_GT = functions_data_processing.xy_from_x_y_tensor(uv_towel[:,0], uv_towel[:,1]) # Shape Mx2
#     print(uv_towel==pt_cld_GT)
#     return pt_cld_GT
    


def chamfer_directional_loss(args, uv=None, X_world=None, Y_world=None, Z_world=None, rounding=0,
                             image_path=None, GT_or_prediction='pred', verbose=1, uv_towel=None):
    """ Output: sum the distance of each towel pixel to the set of predicted uv vertices. 
    The output is this sum as a tensor multiplied by the given weight. 
    
    If a set uv of pixels is not entered, and rather, a set of XYZ world coordinates is given, 
    then we first project the predicted world coordinates to uv coordinates.
    If rounding==1, the projected uv coordinates will be rounded to the nearest integer.
    
    If args.invert_distance_direction=1, then the distance is computed in the opposite direction, 
    i.e, we sum the distance of each predicted uv vertices to the set of towel pixels.
    
    If contour==1, instead of using the whole set of towel pixels for comparison, we will use its contour.
    """
    if args.camera_coordinates!=0:
        print('\n\n\n\nYou must use world coordinates\n\n\n\n')
    else:
        if GT_or_prediction=='GT':
            if args.dtype==1:             
                return torch.zeros([1,0]).to(args.device).double()
            else: 
                return torch.zeros([1,0]).to(args.device)
        else:
            pt_cld_GT = sample_of_towel_pixels(image_path, args) # Shape Mx2

            if X_world is not None:
                # Predicted World coordinates --> Pixel coordinates
                u_proj, v_proj = functions_data_processing.world_to_pixel_coordinates_whole_set_tensor(
                    X_world, Y_world, Z_world, args.Camera_proj_matrix, rounding=rounding, args=args)
                pt_cld_proj = functions_data_processing.xy_from_x_y_tensor(u_proj, v_proj) # Shape Nx2
            else: pt_cld_proj=uv
                
            if verbose==1:
                import functions_plot
                import matplotlib.pyplot as plt
                # Plot uv on RGB
                fig=functions_plot.plot_RGB_and_landmarks(u_visible=pt_cld_proj[:,0], v_visible=pt_cld_proj[:,1],
                                                          image_path=image_path)
                plt.title("Projected uv from predicted world coordinates")
                fig=functions_plot.plot_RGB_and_landmarks(u_visible=pt_cld_GT[:,0], v_visible=pt_cld_GT[:,1],
                                                          image_path=image_path)
                plt.title("uv occupied by the towel obtained from the RGB alone")
                plt.show()
                
            if args.contour==0:
                w_GT_pred = args.w_chamfer_GT_pred
                w_pred_GT = args.w_chamfer_pred_GT
            else:
                w_GT_pred = args.w_chamfer_GTcontour_pred
                w_pred_GT = args.w_chamfer_pred_GTcontour
        
            if args.invert_distance_direction==0:
                if args.dtype==1:             
                    return w_GT_pred*torch.tensor(
                        directional_chamfer(pt_cld_a=pt_cld_GT, pt_cld_b=pt_cld_proj, 
                                            args=args)).to(args.device).double().view([1])
                else:
                    return w_GT_pred*torch.tensor(
                        directional_chamfer(pt_cld_a=pt_cld_GT, pt_cld_b=pt_cld_proj, 
                                            args=args)).to(args.device).float().view([1])
            if args.invert_distance_direction==1:
                if args.dtype==1:             
                    return w_pred_GT*torch.tensor(
                        directional_chamfer(pt_cld_a=pt_cld_proj, pt_cld_b=pt_cld_GT, 
                                            args=args)).to(args.device).double().view([1])
                else:
                    return w_pred_GT*torch.tensor(
                        directional_chamfer(pt_cld_a=pt_cld_proj, pt_cld_b=pt_cld_GT, 
                                            args=args)).to(args.device).float().view([1])

if __name__=='__main__':
    # Run this by using 
    # python functions_train.py --camera-coordinates 0 --w-chamfer-GT-pred 1 --subsample-ratio 1000
    import data_loading
    args = functions_data_processing.parser()
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)
    transformed_dataset = data_loading.instanciate_dataset(args)
    args.template = transformed_dataset[0]
    # args.template['xyz'] is a numpy array of shape num_vertices x 3
    world_coord = torch.from_numpy(args.template['xyz']).to(args.device)
    if args.dtype==1: world_coord=world_coord.double()           
    # Predicted
#     uv_towel = args.template['uv_towel'] if args.unsupervised==1 else None
    uv_towel=None
    ch_loss = chamfer_directional_loss(args, 
                             X_world=world_coord[:,0], Y_world=world_coord[:,1], Z_world=world_coord[:,2], 
                             rounding=0,
                             image_path=args.template['img_name'], verbose=1, uv_towel=uv_towel)
    print(ch_loss)

    # Ground truth
    ch_loss = chamfer_directional_loss(args, 
                             X_world=world_coord[:,0], Y_world=world_coord[:,1], Z_world=world_coord[:,2], 
                             rounding=0,
                             image_path=args.template['img_name'], GT_or_prediction='GT', verbose=1)
    print(ch_loss)
       
def chamfer_directional_loss_batch(args, GT_or_prediction, tensors_coord_only=None, sample_batched=None, verbose=1):
    """ Apply chamfer_directional_loss() on a batch. """
    chamfer_GT_pred = torch.zeros([args.batch_size, 1], requires_grad=False).to(args.device)
    if args.dtype==1: chamfer_GT_pred=chamfer_GT_pred.double()           
    if GT_or_prediction=='pred':
        if args.predict_uv_or_xyz=='xyz':
            for i in range(args.batch_size):
                chamfer_GT_pred[i]=chamfer_directional_loss(
                    args, X_world=tensors_coord_only[i,:,0], Y_world=tensors_coord_only[i,:,1], 
                    Z_world=tensors_coord_only[i,:,2], rounding=0,
                    image_path=sample_batched['img_name'][i], verbose=verbose)
        else:
            for i in range(args.batch_size):
#                     uv_towel = sample_batched['uv_towel'][i] if args.unsupervised==1 else None
                uv_towel=None
                chamfer_GT_pred[i]=chamfer_directional_loss(
                    args, uv=tensors_coord_only[i,:,:], rounding=0,
                    image_path=sample_batched['img_name'][i], verbose=verbose, uv_towel=uv_towel)
    return chamfer_GT_pred
    
if __name__=='__main__':
    # Run this by using 
    # python functions_train.py --camera-coordinates 0 --w-chamfer-GT-pred 1 --subsample-ratio 1000
    import data_loading
    args = functions_data_processing.parser()
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)
    transformed_dataset = data_loading.instanciate_dataset(args)
    transformed_dataset_parts, dataset_sizes, dataloaders = data_loading.random_split_notMixingSequences( 
        dataset=transformed_dataset, args=args)
    sample_batched=next(iter(dataloaders['train']))
    # CAVEAT: tensors_coord_only should be the predicted xyz, but here I will use the GT xyz,
    # to check that chamfer_directional_loss_batch() works 
    tensors_coord_only = sample_batched[args.str_coord] 
    # Predicted
    ch_loss = chamfer_directional_loss_batch(args, 'pred', tensors_coord_only, sample_batched)
    print(ch_loss)
    # Ground Trurh
    ch_loss = chamfer_directional_loss_batch(args, 'GT', tensors_coord_only, sample_batched)
    print(ch_loss)
    
def bidirectional_chamfer(pt_cld_a, pt_cld_b, args):
    """ 
    For every point x in pt_cld_a, compute d_x = d(x, pt_cld_b).
    Return the sum of d_x over all x in pt_cld_a, multiplied by a weight.
    
    For every point x in pt_cld_b, compute d_x = d(x, pt_cld_a).
    Return the sum of d_x over all x in pt_cld_b, multiplied by a weight.
    
    pt_cld_a and pt_cld_b are tensor tensors of shapes Mx2 and Nx2.
    
    The output are two tensors of a single element each.
    
    We do this instead of calling chamfer_directional twice so that we only need to compute the distance matrix once.
    """
    if args.contour==0:
        w_GT_pred = args.w_chamfer_GT_pred
        w_pred_GT = args.w_chamfer_pred_GT
    else:
        w_GT_pred = args.w_chamfer_GTcontour_pred
        w_pred_GT = args.w_chamfer_pred_GTcontour
        
    D = distance_matrix_4_tensors.distance_matrix(pt_cld_a, pt_cld_b, dtype=args.dtype)
    if args.hausdorff==0:
        GT_pred = w_GT_pred*torch.sum(torch.min(D, dim=1)[0]).to(args.device).view([1])
        pred_GT = w_pred_GT*torch.sum(torch.min(D, dim=0)[0]).to(args.device).view([1])
    elif args.hausdorff==1:
        GT_pred = w_GT_pred*torch.max(torch.min(D, dim=1)[0]).to(args.device).view([1])
        pred_GT = w_pred_GT*torch.max(torch.min(D, dim=0)[0]).to(args.device).view([1])
    
    if args.dtype==1: GT_pred, pred_GT = GT_pred.double(), pred_GT.double() 
    elif args.dtype==0: GT_pred, pred_GT = GT_pred.float(), pred_GT.float() 
    
    return GT_pred, pred_GT
        
def chamfer_bidirectional_loss(args, uv=None, X_world=None, Y_world=None, Z_world=None, rounding=0,
                             image_path=None, GT_or_prediction='pred', verbose=1, uv_towel=None):
    """ Outputs: 
    - sum the distance of each towel pixel to the set of predicted uv vertices. 
    The 1st output is this sum as a tensor multiplied by the given weight. 
    - sum the distance of each predicted uv vertices to the set of towel pixels. 
    The 2nd output is this sum as a tensor multiplied by the given weight. 
    If a set uv of pixels is not entered, and rather, a set of XYZ world coordinates is given, 
    then we first project the predicted world coordinates to uv coordinates.
    If rounding==1, the projected uv coordinates will be rounded to the nearest integer.
    
    We do this instead of calling chamfer_directional_loss twice so that we only need to compute the distance matrix once.
    
    If contour==1, instead of using the whole set of towel pixels for comparison, we will use its contour.
    """
    if args.camera_coordinates!=0:
        print('\n\n\n\nYou must use world coordinates\n\n\n\n')
    else:
        if GT_or_prediction=='GT':
            if args.dtype==1: 
                return torch.zeros([1,0]).to(args.device).double(), torch.zeros([1,0]).to(args.device).double()
            else:
                return torch.zeros([1,0]).to(args.device), torch.zeros([1,0]).to(args.device)
        else:
            pt_cld_GT = sample_of_towel_pixels(image_path, args) # Shape Mx2

            if X_world is not None:
                # Predicted World coordinates --> Pixel coordinates
                u_proj, v_proj = functions_data_processing.world_to_pixel_coordinates_whole_set_tensor(
                    X_world, Y_world, Z_world, args.Camera_proj_matrix, rounding=rounding, args=args)
                pt_cld_proj = functions_data_processing.xy_from_x_y_tensor(u_proj, v_proj) # Shape Nx2
            else: pt_cld_proj=uv
                
            return bidirectional_chamfer(pt_cld_a=pt_cld_GT, pt_cld_b=pt_cld_proj, args=args)

def chamfer_bidirectional_loss_batch(args, GT_or_prediction, tensors_coord_only=None, sample_batched=None, verbose=1):
    """ Apply chamfer_bidirectional_loss() on a batch. """
    chamfer_GT_pred = torch.zeros([args.batch_size, 1], requires_grad=False).to(args.device)
    chamfer_pred_GT = torch.zeros([args.batch_size, 1], requires_grad=False).to(args.device)
    if args.dtype==1: chamfer_GT_pred, chamfer_pred_GT = chamfer_GT_pred.double() , chamfer_pred_GT.double() 

    if GT_or_prediction=='pred':
        if args.predict_uv_or_xyz=='xyz':
            for i in range(args.batch_size):
                chamfer_GT_pred[i], chamfer_pred_GT[i]=chamfer_bidirectional_loss(
                    args, X_world=tensors_coord_only[i,:,0], Y_world=tensors_coord_only[i,:,1], 
                    Z_world=tensors_coord_only[i,:,2], rounding=0,
                    image_path=sample_batched['img_name'][i], verbose=verbose)
        else:
            for i in range(args.batch_size):
#                     uv_towel = sample_batched['uv_towel'][i] if args.unsupervised==1 else None
                uv_towel=None
                chamfer_GT_pred[i], chamfer_pred_GT[i]=chamfer_bidirectional_loss(
                    args, uv=tensors_coord_only[i,:,:], rounding=0,
                    image_path=sample_batched['img_name'][i], verbose=verbose, uv_towel=uv_towel)
    return chamfer_GT_pred, chamfer_pred_GT
    
def chamfer_all_options(args, GT_or_prediction, tensors_coord_only, sample_batched):
    """ Example of use within create_labels_and_prediction():
        >>> tensor_parts['chamfer_GT_pred'], tensor_parts['chamfer_pred_GT'] = chamfer_all_options(
        args, GT_or_prediction, tensors_coord_only, sample_batched)
    """
    if args.contour==0:
        w_GT_pred = args.w_chamfer_GT_pred
        w_pred_GT = args.w_chamfer_pred_GT
    else:
        w_GT_pred = args.w_chamfer_GTcontour_pred
        w_pred_GT = args.w_chamfer_pred_GTcontour
        
    # Chamfer: None
    if w_GT_pred==0 and w_pred_GT==0:
        if args.dtype==1: 
            return torch.Tensor().to(args.device).double(), torch.Tensor().to(args.device).double() 
        else:
            return torch.Tensor().to(args.device), torch.Tensor().to(args.device)

    # Chamfer: ONLY towel pixels --> uv prediction
    if w_GT_pred!=0 and w_pred_GT==0:
        args.invert_distance_direction=0
        if args.dtype==1: 
            return chamfer_directional_loss_batch(args, GT_or_prediction, tensors_coord_only, sample_batched, 
                                              args.verbose), torch.Tensor().to(args.device).double() 
        else:
            return chamfer_directional_loss_batch(args, GT_or_prediction, tensors_coord_only, sample_batched, 
                                              args.verbose), torch.Tensor().to(args.device)

    # Chamfer: ONLY uv prediction --> towel pixels
    if w_pred_GT!=0 and w_GT_pred==0:
        args.invert_distance_direction=1
        if args.dtype==1: 
            return chamfer_directional_loss_batch(args, GT_or_prediction, tensors_coord_only, sample_batched, 
                                              args.verbose), torch.Tensor().to(args.device).double() 
        else:
            return chamfer_directional_loss_batch(args, GT_or_prediction, tensors_coord_only, sample_batched, 
                                              args.verbose), torch.Tensor().to(args.device)

    # Chamfer: towel pixels <--> uv prediction
    if w_pred_GT!=0 and w_GT_pred!=0:
        return chamfer_bidirectional_loss_batch(args, GT_or_prediction, tensors_coord_only, sample_batched, args.verbose)

def prediction(model, inputs, args):
    """ This produces the prediction of the model, given the value of args.new_uv_norm, 
    which specifies some possible (un)normalization necessary to be applied to the output of the ResNet,
    to keep the output of the ResNet giving values between 0 and 1 and still obtain the unconstrained predictions we want. """
    if args.dtype==1: inputs=inputs.double()
    outputs = model(inputs)
    #     model(inputs) has the form:
    #     [u_0, v_0, ..., u_i, v_i], if args.predict_uv_or_xyz == 'uv'
    #     [x_0, y_0, z_0, ..., x_i, y_i, z_i], if args.predict_uv_or_xyz == 'xyz'
    #     [u_0, v_0, y_0, ..., u_i, v_i, y_i], if args.predict_uv_or_xyz == 'uvy'
    if args.new_uv_norm==1 and args.predict_uv_or_xyz in ['uv', 'uvy']:
        outputs = functions_data_processing.reshape_labels_back(
            outputs, args.batch_size, args.num_selected_vertices, args.num_coord_per_vertex)
        print(outputs[0,0,0])
        outputs[:,:,0], outputs[:,:,1] = functions_data_processing.unnormalize_uv_01_tensor(u=outputs[:,:,0],v=outputs[:,:,1])
        print(outputs[0,0,0], "\n")
        outputs = functions_data_processing.reshape_labels(
            outputs, args.batch_size, args.num_selected_vertices,  args.num_coord_per_vertex)
        
    if args.permutation_dir is not None:
        outputs[:,:]=outputs[:,args.permutation_4_output_layer]
        
    return outputs
    
def coord_only(args, sample_batched, GT_or_prediction, model, inputs):   
    """
    Compute the GT or predicted uv or xyz coordinates of a batch.
    The output is a tensor of shape (args.batch_size, args.num_selected_vertices, args.num_coord_per_vertex)
    """
    if GT_or_prediction=='GT': 
        tensors_coord_only = sample_batched[args.str_coord]
    else:
        outputs = prediction(model, inputs, args)
        tensors_coord_only = functions_data_processing.reshape_labels_back(
            outputs, args.batch_size, args.num_selected_vertices, args.num_coord_per_vertex)
        if args.print_prediction==1:
#             print('model(inputs)[0,0].item():', model(inputs)[0,0].item())
            print('model(inputs)[0,0].item():', tensors_coord_only[0,0,0].item()) # avoid recomputing model(inputs)
#             print(sample_batched['img_name'][0])
    tensors_coord_only=tensors_coord_only.to(args.device)
    if args.dtype==1: tensors_coord_only=tensors_coord_only.double() 
    # Shape (batch_size, args.num_selected_vertices, args.num_coord_per_vertex)
    return tensors_coord_only

def coord_only_uvy(args, sample_batched, GT_or_prediction, model, inputs):   
    """
    Compute the GT or predicted uv and xyz coordinates of a batch from the model which outputs uv and y coordinates.
    The output consists of 
    - uv_coord: a tensor of shape (args.batch_size, args.num_selected_vertices, 2)
    - xyz_coord: a tensor of shape (args.batch_size, args.num_selected_vertices, 3)
    """
    if GT_or_prediction=='GT': 
        uv_coord = sample_batched['uv'].to(args.device)
        xyz_coord = sample_batched['xyz'].to(args.device)
    else:
        if args.verbose==1:
            print("model.state_dict().keys():", model.state_dict().keys(), "\n") # Note that the Dropout layer does not appear, not even if the method is not downloaded as Pretrained
#             print("weights of 1st conv layer:", model.state_dict()['conv1.weight'], "\n")
            print("weights of FC layer:", model.state_dict()['fc.weight'], "\n")
    #         print("weights of FC layer bias?:", model.state_dict()['fc.bias'], "\n")
            print("model(inputs):", model(inputs), "\n")
        
        outputs = prediction(model, inputs, args)
        tensors_coord_only = functions_data_processing.reshape_labels_back(
            outputs, args.batch_size, args.num_selected_vertices, args.num_coord_per_vertex) # Shape (batch_size, args.num_selected_vertices, 3)

        uv_coord = tensors_coord_only[:,:,0:2].to(args.device)
        if args.GTuv!=0:
            uv_coord_GT = sample_batched['uv'].to(args.device)
        y_coord =  tensors_coord_only[:,:,2].to(args.device)
        
        # uvy--> xyz
        if args.GTuv==0:
            xyz_coord = functions_data_processing.pixel_to_world_of_batch_of_clouds_knowing_Y(
            uv_coord=uv_coord, C=args.Camera_proj_matrix_tensor, C_inv_tensor=args.Camera_proj_matrix_inv_tensor, 
            Y_world_GT = y_coord, args=args, xyz_world_GT=None, verbose=0)  
        else:
            xyz_coord = functions_data_processing.pixel_to_world_of_batch_of_clouds_knowing_Y(
            uv_coord=uv_coord_GT, C=args.Camera_proj_matrix_tensor, C_inv_tensor=args.Camera_proj_matrix_inv_tensor, 
            Y_world_GT = y_coord, args=args, xyz_world_GT=None, verbose=0)  
        if args.print_prediction==1:
#             print('model(inputs)[0,0].item():', model(inputs)[0,0].item())
            print('Possibly unnormalized model(inputs)[0,0].item():', tensors_coord_only[0,0,0].item()) # avoid recomputing model(inputs)
#             print(sample_batched['img_name'][0])
    if args.dtype==1: uv_coord, xyz_coord = uv_coord.double(), xyz_coord.double() 
    return uv_coord, xyz_coord

def coord_only_uvD(args, sample_batched, GT_or_prediction, model, inputs):   
    """
    Compute the GT or predicted uv (pixel) and D (depth) coordinates of a batch from the model which outputs uv+D coordinates.
    The output consists of 
    - uv_coord: a tensor of shape (args.batch_size, args.num_selected_vertices, 2)
    - D: a tensor of shape (args.batch_size, args.num_selected_vertices, 1)
    """
    if GT_or_prediction=='GT': 
        uv_coord = sample_batched['uv'].to(args.device)
        D = sample_batched['D'].to(args.device)
    else:
        if args.verbose==1:
            print("model.state_dict().keys():", model.state_dict().keys(), "\n") # Note that the Dropout layer does not appear, not even if the method is not downloaded as Pretrained
#             print("weights of 1st conv layer:", model.state_dict()['conv1.weight'], "\n")
            print("weights of FC layer:", model.state_dict()['fc.weight'], "\n")
    #         print("weights of FC layer bias?:", model.state_dict()['fc.bias'], "\n")
            print("model(inputs):", model(inputs), "\n")
        
        outputs = prediction(model, inputs, args)
        tensors_coord_only = functions_data_processing.reshape_labels_back(
            outputs, args.batch_size, args.num_selected_vertices, args.num_coord_per_vertex) # Shape (batch_size, args.num_selected_vertices, 3)

        uv_coord = tensors_coord_only[:,:,0:2].to(args.device)
        if args.GTuv!=0:
            uv_coord_GT = sample_batched['uv'].to(args.device)
        D =  tensors_coord_only[:,:,2].to(args.device).view(uv_coord.shape[0], uv_coord.shape[1], 1)
        
#         # See uvy--> xyz from coord_only_uvy()

        if args.print_prediction==1:
#             print('model(inputs)[0,0].item():', model(inputs)[0,0].item())
            print('Possibly unnormalized model(inputs)[0,0].item():', tensors_coord_only[0,0,0].item()) # avoid recomputing model(inputs)
#             print(sample_batched['img_name'][0])
    if args.dtype==1: uv_coord, D = uv_coord.double(), D.double() 
    return uv_coord, D

def combine_all_in_1_tensor(args, sample_batched, GT_or_prediction, model, inputs, uv_coord, xyz_coord):
    # Reshape from (batch_size, args.num_selected_vertices, args.num_coord_per_vertex) to 
    # (batch_size, args.num_selected_vertices* args.num_coord_per_vertex). 
    # tensor_parts['coord'][j] has the form [x_0, y_0, z_0, ..., x_i, y_i, z_i] or [u_0, v_0, ..., u_i, v_i].
    if args.w_uv==0:
        tensor_parts = {'uv': torch.Tensor().to(args.device)} # Empty tensor
    else:
        tensor_parts = {'uv': functions_data_processing.reshape_labels(
            uv_coord.contiguous(), args.batch_size, args.num_selected_vertices, 2)} 
    if args.w_xyz==0:
        tensor_parts['xyz']= torch.Tensor().to(args.device)
    else:
        tensor_parts['xyz']=functions_data_processing.reshape_labels(
            xyz_coord, args.batch_size, args.num_selected_vertices, 3) 
    tensor_parts['uv']=args.w_uv*tensor_parts['uv']
    tensor_parts['xyz']=args.w_xyz*tensor_parts['xyz']
    if args.dtype==1: tensor_parts['uv'], tensor_parts['xyz'] = tensor_parts['uv'].double() , tensor_parts['xyz'].double() 

    # Normal orientations
    # This currently only works for args.normals==0
    # If I want to use the case args.normals==1, I need to make the normal vector computation
    if args.normals == 1:
        # tensor_parts['normals'] has the form [nx_0, ny_0, nz_0, ..., nx_i, ny_i, nz_i]
        print('\n!!!Case to be considered in the code yet!!!\n')
    else:
        tensor_parts['normals'] = torch.Tensor() # Empty tensor
    tensor_parts['normals'] = tensor_parts['normals'].to(args.device)
    if args.dtype==1: tensor_parts['normals'] = tensor_parts['normals'].double()

    # Geodesic distance between adjacent vertices
    if args.predict_uv_or_xyz=='xyz' or args.predict_uv_or_xyz=='uvy':
        # If GT_or_prediction=='pred', we compute the distances between the predicted vertices
        # If GT_or_prediction=='GT', then
        #      - If args.unsupervised==1, we use the distance between vertices of args.template
        #      - If args.unsupervised==0, we use the distance between vertices from GT corresponding to each input
        if args.unsupervised==1 and GT_or_prediction=='GT': # Load vertices of args.template
            xyz_coord_template = torch.zeros(xyz_coord.shape, dtype=xyz_coord.dtype,
                                         requires_grad=False).to(args.device)
            for i in range(args.batch_size):
                xyz_coord_template[i,:,:] = torch.from_numpy(args.template['xyz']).to(args.device)
                if args.dtype==1: xyz_coord_template[i,:,:] = xyz_coord_template[i,:,:].double()
            if args.verbose==1: print("args.template['img_name']:", args.template['img_name'])
            xyz_submesh_batch = xyz_coord_template
        else: xyz_submesh_batch = xyz_coord
    elif args.predict_uv_or_xyz=='uv': 
        xyz_submesh_batch = uv_coord

    if args.neighb_dist_weight!=0:
        tensor_parts['neighbour_dist'] = args.neighb_dist_weight * submesh.squared_l2_distance_of_adjacent_vertices_of_submesh_batch(
            args, Vertex_coordinates_submesh_batch=xyz_submesh_batch)
    else: tensor_parts['neighbour_dist'] = torch.Tensor() # Empty tensor
    tensor_parts['neighbour_dist'] = tensor_parts['neighbour_dist'].to(args.device)
    if args.dtype==1: tensor_parts['neighbour_dist'] = tensor_parts['neighbour_dist'].double()
    elif args.dtype==0: tensor_parts['neighbour_dist'] = tensor_parts['neighbour_dist'].float()
    # Tensor of shape (batch_size, num_selected_vertices)

    # Chamfer - specifying whether we already have uv or whether we have to get them from xyz
    tensors_coord_only = xyz_coord if args.predict_uv_or_xyz=='xyz' else uv_coord

    # Chamfer on the towel pixels
    args.contour=0
    tensor_parts['chamfer_GT_pred'], tensor_parts['chamfer_pred_GT'] = chamfer_all_options(
        args, GT_or_prediction, tensors_coord_only, sample_batched)

    # Chamfer on the contour of the towel pixels
    args.contour=1
    tensor_parts['chamfer_contourGT_pred'], tensor_parts['chamfer_pred_GTcontour'] = chamfer_all_options(
        args, GT_or_prediction, tensors_coord_only, sample_batched)

    tensors = torch.cat((tensor_parts['uv'], tensor_parts['xyz'], tensor_parts['normals'], tensor_parts['neighbour_dist'],
                         tensor_parts['chamfer_GT_pred'], tensor_parts['chamfer_pred_GT'], 
                         tensor_parts['chamfer_contourGT_pred'], tensor_parts['chamfer_pred_GTcontour']), 1)
    return tensors

def create_labels_and_prediction(args, sample_batched, GT_or_prediction='GT', model=None, inputs=None, 
                                 normalize_xyz_min=[0,0,0], normalize_xyz_max=[1,1,1],
                                 normalize_D_min=0, normalize_D_max=1):
    """     
    Input:
    - inputs: the RGB image to predict from.
    - sample_batched: a batch obtained from a dataloader.
    - If args.unsupervised==1, then args.template contains the element from the dataset which acts as the template.
    
    If GT_or_prediction=='GT':
        Output:
        - tensors: 2D tensor of labels whose 1st dim is batch_size. 
        It is the tensor which must be compared to the prediction in the loss of the batch sample_batched
        for which the gradient is computed. 
        This tensor may include xyz or uv vertex coordinates and/or normal vectors and/or distance between neighbouring vertices, 
        depending on the input arguments.
        - tensors_coord_only: tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex) 
        containing the ground truth xyz or uv cocordinates of all the vertices in the batch.
    If GT_or_prediction=='pred':
        Output:
        - tensors: 2D tensor of predictions whose 1st dim is batch_size. 
        It is the tensor which must be compared to the ground truth in the loss of the batch sample_batched
        for which the gradient is computed. 
        This tensor may include xyz or uv vertex coordinates and/or normal vectors and/or distance between neighbouring vertices, 
        depending on the input arguments.
        - uv_coord: tensor of shape (batch_size, num_selected_vertices, 2) 
        containing the predicted uv cocordinates of all the vertices in the batch.
        - xyz_coord: tensor of shape (batch_size, num_selected_vertices, 3) 
        containing the predicted xyz cocordinates of all the vertices in the batch.
    """
    # Predicted/GT coordinates
    if args.predict_uv_or_xyz=='uv':
        uv_coord = coord_only(args, sample_batched, GT_or_prediction, model, inputs)
        xyz_coord = torch.Tensor().to(args.device) # Empty tensor
        D = torch.Tensor().to(args.device) # Empty tensor
    elif args.predict_uv_or_xyz=='xyz':
        xyz_coord = coord_only(args, sample_batched, GT_or_prediction, model, inputs)    
        uv_coord = torch.Tensor().to(args.device)
        D = torch.Tensor().to(args.device) # Empty tensor
    elif args.predict_uv_or_xyz=='uvy':
        uv_coord, xyz_coord = coord_only_uvy(args, sample_batched, GT_or_prediction, model, inputs)   
        D = torch.Tensor().to(args.device) # Empty tensor
    elif args.predict_uv_or_xyz=='uvD':
        uv_coord, D = coord_only_uvD(args, sample_batched, GT_or_prediction, model, inputs)  
        xyz_coord=functions_data_processing.normalized_uvD_to_normalized_xyz_batch(
            uv_coord, D, args, normalize_D_min, normalize_D_max, normalize_xyz_min, normalize_xyz_max, f_u=args.f_u, f_v=args.f_v)
            
    if args.dtype==1: uv_coord, xyz_coord, D = uv_coord.double() , xyz_coord.double(), D.double()
        
    if args.loss_weights==1:
        tensors = torch.Tensor() # Empty tensor
        tensors = tensors.to(args.device)
        if args.dtype==1: tensors = tensors.double()
    else:
        tensors = combine_all_in_1_tensor(args, sample_batched, GT_or_prediction, model, inputs, uv_coord, xyz_coord)

    return tensors, uv_coord, xyz_coord, D

# def evaluate_model(model, saving_directory, dataloaders, args, criterion = nn.MSELoss(), kwargs_normalize_labels_pred=None):
#     """Function to compute and save the loss (or losses) of the training, validation and test set of a model."""
#     with torch.no_grad():
#         text_to_save = 'Loss of the different datasets:\n' 
#         phases = train.create_phases_4_evaluation(args)
        
#         for phase in phases:
#             text_to_save += '\n' + phase + '\n' 
#             model.train() if args.testing_eval == 3 else model.eval()
#             running_loss_myown_uv_coord_only = 0
#             running_loss_myown_xyz_coord_only = 0
#             running_loss = 0
#             running_loss_uv_coord_only = 0
#             running_loss_xyz_coord_only = 0
            
#             # Iterate over data
#             for batch_idx, sample_batched in enumerate(dataloaders[phase], 1):
#                 inputs = sample_batched['image']
#                 inputs = inputs.to(args.device)

#                 labels, labels_uv_coord, labels_xyz_coord = create_labels_and_prediction(
#                     args, sample_batched=sample_batched, **kwargs_normalize_labels_pred)

#                 # Shape args.batch_size x (args.num_selected_vertices * args.num_coord_per_vertex)
#                 outputs, outputs_uv_coord, outputs_xyz_coord = create_labels_and_prediction(
#                     args, sample_batched=sample_batched, GT_or_prediction='pred', model=model, inputs=inputs, **kwargs_normalize_labels_pred)

#                 # torch.nn MSELoss
#                 loss = criterion(outputs, labels)

#                 # MSE loss of predicted coordinates only, 
#                 # with no extra structure added such as distances between neighbouring vertices, normal vectors...           
#                 if args.predict_uv_or_xyz=='uv' or args.predict_uv_or_xyz=='uvy':
#                     loss_uv_coord_only = criterion(outputs_uv_coord, labels_uv_coord) 
#                 else:
#                     loss_uv_coord_only = -1*torch.ones([1], dtype=torch.float64, device=args.device)
#                 if args.predict_uv_or_xyz=='xyz' or args.predict_uv_or_xyz=='uvy':
#                     loss_xyz_coord_only = criterion(outputs_xyz_coord, labels_xyz_coord) 
#                 else:
#                     loss_xyz_coord_only = -1*torch.ones([1], dtype=torch.float64, device=args.device)

#                 if args.loss == 1: # my own MSE loss of predicted coordinates only
#                     if args.predict_uv_or_xyz=='uv' or args.predict_uv_or_xyz=='uvy':
#                         loss_myown_uv_coord_only = MSE_loss(outputs_uv_coord, labels_uv_coord, 
#                                                             batch_size=args.batch_size, num_selected_vertices=2)
#                     else:
#                         loss_myown_uv_coord_only = -1 
#                     if args.predict_uv_or_xyz=='xyz' or args.predict_uv_or_xyz=='uvy':
#                         loss_myown_xyz_coord_only = MSE_loss(outputs_xyz_coord, labels_xyz_coord,
#                                                              batch_size=args.batch_size, num_selected_vertices=3)
#                     else:
#                         loss_myown_xyz_coord_only = -1 

#                 if args.loss == 1: # my own MSE loss
#                     running_loss_myown_uv_coord_only += loss_myown_uv_coord_only
#                     running_loss_myown_xyz_coord_only += loss_myown_xyz_coord_only
#                 running_loss += loss.item()
#                 running_loss_uv_coord_only += loss_uv_coord_only.item()
#                 running_loss_xyz_coord_only += loss_xyz_coord_only.item()
# #                 # loss.item() is the average (at least with MSE) of 
# #                 # the losses of the observations in the batch.
# #                         At the end of the phase, running_loss will be the sum of loss.item() for all the batches.
# #                         Then, we will define the epoch loss as the average of the MSE loss of each observation,
# #                         which equals 
# #                         running_loss * args.batch_size / training_size,
# #                         or equivalently, 
# #                         running_loss / (number of batches in which the training data is split),
# #                         or the analogue with validation set size if phase=='val' 

#             # Average of the MSE of each observation
#             epoch_loss = running_loss / len(dataloaders[phase]) 
#             epoch_loss_uv_coord_only = running_loss_uv_coord_only / len(dataloaders[phase]) 
#             epoch_loss_xyz_coord_only = running_loss_xyz_coord_only / len(dataloaders[phase]) 
#             if args.loss == 1: # my own MSE loss
#                 epoch_loss_myown_uv_coord_only = running_loss_myown_uv_coord_only / len(dataloaders[phase])
#                 epoch_loss_myown_xyz_coord_only = running_loss_myown_xyz_coord_only / len(dataloaders[phase])
# #                         Remember: 
# #                         dataset_sizes['train'] == training_size
# #                         len(dataloaders['train']) == training_size / args.batch_size, i.e.,
# #                         len(dataloaders['train']) == number of batches in which the training data is split
# #                         inputs.size(0) == args.batch_size

#             text_to_save += 'Loss: {:4f}\n'.format(epoch_loss)
#             text_to_save += 'Loss_uv_coord_only: {:4f}\n'.format(epoch_loss_uv_coord_only)
#             text_to_save += 'Loss_xyz_coord_only: {:4f}\n'.format(epoch_loss_xyz_coord_only)
#             if args.loss == 1: # my own MSE loss
#                 text_to_save += 'My Loss on uv_coord_only: {:4f}\n'.format(epoch_loss_myown_uv_coord_only)
#                 text_to_save += 'My Loss on uv_coord_only: {:4f}\n'.format(epoch_loss_myown_uv_coord_only)
#         print('\n'+text_to_save+'\n')

#         file_name = os.path.join(saving_directory, 'final_losses_train_val_test.txt')
#         print("Saving train/val/test loss in " + file_name)
#         with open(file_name, 'w') as x_file:
#             x_file.write(text_to_save)

def evaluate_model(model, saving_directory, dataloaders, args, criterion = nn.MSELoss(), kwargs_normalize_labels_pred=None):
    """Function to compute and save the loss (or losses) of the training, validation and test set of a model."""
    optimizer, scheduler, writer = None, None, None
    train.train_model(model, optimizer, scheduler, saving_directory, writer, criterion,
               dataloaders, args, kwargs_normalize_labels_pred=kwargs_normalize_labels_pred, eval_only=1)
            
def show_architecture_and_forward_pass(model_conv, dataloaders, args, kwargs_normalize):
    print('Architecture of this ResNet after changing the final FC layer:')
    print(model_conv.parameters, "\n")

    # Pick the first training batch
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)
    if args.verbose==1:
        print("sample_batched['image'].shape[0], batch_size:", sample_batched['image'].shape[0], args.batch_size)

    if args.verbose==1:
        print('Output and Loss on the first training batch.')
        print('\nImage names of batch:', sample_batched['img_name'])
        print()

    # Predict the output and loss of the shown batch
    model_conv.train()
    labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord, labels_D, outputs_D = predict_and_loss(
        model = model_conv, sample_batched = sample_batched, args=args, kwargs_normalize=kwargs_normalize)
    print("output_uv of first element in the batch:")
    print(outputs_uv_coord[0,:,:])
    
def create_loss_geo(args, outputs_xyz_coord, labels_xyz_coord):
    """     
    Input:
    - If args.unsupervised==1, then args.template contains the element from the dataset which acts as the template.
    - outputs_xyz_coord: tensor of shape (args.batch_size, args.num_selected_vertices, 3) with the predicted xyz
    - labels_xyz_coord: tensor of shape (args.batch_size, args.num_selected_vertices, 3) with the GT xyz, only needed if args.unsupervised==0

    Output: 
    For each pair of adjacent vertices in the mesh, compute their Euclidean pair-wise distance in the prediction and in GT or in the template.
    With these 2 lists of distances (one for prediction and one for GT or the template), compute MSE loss and return it.
    """

    # Compute distances between predicted vertices
    outputs_geo = submesh.squared_l2_distance_of_adjacent_vertices_of_submesh_batch(
        args, Vertex_coordinates_submesh_batch=outputs_xyz_coord)
    
    if args.unsupervised==0: # compare against distances between GT vertices
        labels_geo = submesh.squared_l2_distance_of_adjacent_vertices_of_submesh_batch(
        args, Vertex_coordinates_submesh_batch=labels_xyz_coord)
    else: # compare against distances between vertices of args.template
        # Load vertices of args.template
        xyz_coord_template = torch.zeros(outputs_xyz_coord.shape, dtype=outputs_xyz_coord.dtype,
                                     requires_grad=False).to(args.device)
        for i in range(args.batch_size):
            xyz_coord_template[i,:,:] = torch.from_numpy(args.template['xyz']).to(args.device)
            if args.dtype==1: xyz_coord_template[i,:,:] = xyz_coord_template[i,:,:].double()
        if args.verbose==1: print("args.template['img_name']:", args.template['img_name'])
        
        labels_geo = submesh.squared_l2_distance_of_adjacent_vertices_of_submesh_batch(
        args, Vertex_coordinates_submesh_batch=xyz_coord_template).to(args.device)

    criterion = nn.MSELoss()
    return criterion(outputs_geo, labels_geo) # name it loss_geo  

def create_loss_horizConsecEdges(args, outputs_xyz_coord, labels_xyz_coord):
    """     
    Input:
    - If args.unsupervised==1, then we use as Ground Truth that horizontally consecutive edges are parallel (which represents what happens in the template).
    - If args.unsupervised==0, then we use as Ground Truth the true sin of the angle between horizontally consecutive edges.
    - outputs_xyz_coord: tensor of shape (args.batch_size, args.num_selected_vertices, 3) with the predicted xyz
    - labels_xyz_coord: tensor of shape (args.batch_size, args.num_selected_vertices, 3) with the GT xyz, only needed if args.unsupervised==0

    Output: 
    loss_horizConsecEdges: For each pair of horizontally consecutive edges in the mesh, compute the sine of their angle in the prediction and in GT or in the template.
    With these 2 lists of sines (one for prediction and one for GT or the template), compute MSE loss and return it.
    """

    # Compute distances between predicted vertices
    outputs_horizConsecEdges = submesh.sin_3horizConsec_vert_from_submesh_batch(
        args, Vertex_coordinates_submesh_batch=outputs_xyz_coord)
    
    if args.unsupervised==0: # compare against angles between GT edges
        labels_horizConsecEdges = submesh.sin_3horizConsec_vert_from_submesh_batch(
        args, Vertex_coordinates_submesh_batch=labels_xyz_coord)
    else: # compare against angles between vertices of args.template
        # Assume horizontally consecutive edges are parallel (which represents what happens in the template)
        n_pairs = submesh.n_horiz_consec_edges_of_submesh(args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal)
        labels_horizConsecEdges = torch.zeros([args.batch_size, n_pairs], dtype=outputs_xyz_coord.dtype,
                                     requires_grad=False, device=args.device)
        
    criterion = nn.MSELoss()
    return criterion(outputs_horizConsecEdges, labels_horizConsecEdges) # name it loss_horizConsecEdges  

def create_loss_verConsecEdges(args, outputs_xyz_coord, labels_xyz_coord):
    """     
    Input:
    - If args.unsupervised==1, then we use as Ground Truth that vertically consecutive edges are parallel (which represents what happens in the template).
    - If args.unsupervised==0, then we use as Ground Truth the true sin of the angle between vertically consecutive edges.
    - outputs_xyz_coord: tensor of shape (args.batch_size, args.num_selected_vertices, 3) with the predicted xyz
    - labels_xyz_coord: tensor of shape (args.batch_size, args.num_selected_vertices, 3) with the GT xyz, only needed if args.unsupervised==0

    Output: 
    loss_verConsecEdges: For each pair of vertically consecutive edges in the mesh, compute the sine of their angle in the prediction and in GT or in the template.
    With these 2 lists of sines (one for prediction and one for GT or the template), compute MSE loss and return it.
    """

    # Compute distances between predicted vertices
    outputs_verConsecEdges = submesh.sin_3verConsec_vert_from_submesh_batch(
        args, Vertex_coordinates_submesh_batch=outputs_xyz_coord)
    
    if args.unsupervised==0: # compare against angles between GT edges
        labels_verConsecEdges = submesh.sin_3verConsec_vert_from_submesh_batch(
        args, Vertex_coordinates_submesh_batch=labels_xyz_coord)
    else: # compare against angles between vertices of args.template
        # Assume vertically consecutive edges are parallel (which represents what happens in the template)
        n_pairs = submesh.n_ver_consec_edges_of_submesh(args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal)
        labels_verConsecEdges = torch.zeros([args.batch_size, n_pairs], dtype=outputs_xyz_coord.dtype,
                                     requires_grad=False, device=args.device)
        
    criterion = nn.MSELoss()
    return criterion(outputs_verConsecEdges, labels_verConsecEdges) # name it loss_verConsecEdges  

def hyperpars_to_try(hyperpar_option):
    """ Return dictionary of hyperparameters to try, sorted by key"""
    if hyperpar_option==1:
        hyperpar_values = {'lr': [0.01, 0.005, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                       'neighb_dist_weight': [10**10, 10**5, 10, 1, 0.3, 0.01]}
    elif hyperpar_option==2:
        hyperpar_values = {'lr': [0.01, 0.005, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                       'neighb_dist_weight': [10**8, 10**6, 10**4, 10**3, 10**2]}
    elif hyperpar_option==3:
        hyperpar_values = {'lr': [0.01, 0.005, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                       'neighb_dist_weight': [10**8, 10**6]}
    elif hyperpar_option==4: 
        # From this option on, neighb_dist_weight is a float between 0 and 1 which representa a ratio:
        # the distance of every pair of adjacent vertices in a submesh multiplied by neighb_dist_weight
        # and the rest of the loss is multiplied by 1-neighb_dist_weight
        hyperpar_values = {'lr': [0.01, 0.005, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                       'neighb_dist_weight': [0.9, 0.7, 0.5, 0.3, 0.1]}
    elif hyperpar_option==5: 
        hyperpar_values = {'lr': [0.01], 'momentum': [0.9], 'gamma': [0.1],
                       'neighb_dist_weight': [0.9, 0.7, 0.5, 0.3, 0.1]}
    elif hyperpar_option==6: 
        neigh_list = np.linspace(0.0, 1.0, num=21).tolist()
        hyperpar_values = {'lr': [0.01], 'momentum': [0.9], 'gamma': [0.1],
                       'neighb_dist_weight': neigh_list}
    elif hyperpar_option==7: 
        neigh_list = np.linspace(0.0, 1.0, num=11).tolist()
        hyperpar_values = {'lr': [0.01], 'momentum': [0.9], 'gamma': [0.1],
                       'neighb_dist_weight': neigh_list}
    elif hyperpar_option==8: 
        neigh_list = [0.0, 0.2, 0.4, 0.6, 0.8]
        hyperpar_values = {'lr': [0.01], 'momentum': [0.9], 'gamma': [0.1],
                       'neighb_dist_weight': neigh_list}
    elif hyperpar_option==9: 
        w_chamfer_GT_pred_list = [0.01, 0.1, 1., 10]
        hyperpar_values = {'lr': [0.01], 'momentum': [0.9], 'gamma': [0.1],
                       'neighb_dist_weight': [0.8], 'w_chamfer_GT_pred': w_chamfer_GT_pred_list}
        # CAVEAT: Change of format of loss weights:
        # Up to 29/01/19 (hyperpar_option==9), w_coord = 1-neighb_dist_weight and w_chamfer_GT_pred was an absolute value, 
        # as big as wanted.
        # From 30/01/19 on (hyperpar_option==10), all the weights will be normalized to sum up to 1. 
        # E.g., w_coord=1, neighb_dist_weight=4, w_chamfer_GT_pred=5 
        # After normalization: w_coord=0.1, neighb_dist_weight=0.4, w_chamfer_GT_pred=0.5.
        # See functions_data_processing.normalize_weights(), which is used in 
        # functions_train.create_labels_and_prediction()
    if hyperpar_option==10:
        hyperpar_values = {'lr': [0.01, 0.005, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.1, 0.01], 
                          'neighb_dist_weight': [0]}
    if hyperpar_option==11:
        hyperpar_values = {'lr': [0.01, 0.005], 'momentum': [0.1, 0.3], 'gamma': [0.1, 0.01], 
                          'neighb_dist_weight': [0]}
    if hyperpar_option==12:
        if args.predict_uv_or_xyz == 'uvy':
            hyperpar_values = {'lr': [0.01, 0.005], 'momentum': [0.3, 0.9], 'gamma': [0.1, 0.01], 
                          'neighb_dist_weight': [0, 5, 10], 'w_uv': [1], 'w_xyz': [1, 5, 10]}
        else: print("\n"*10 + "In order to use hyperpar_option 12, you need predict_uv_or_xyz 'uvy'" + "\n"*10)
    if hyperpar_option==13:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==1:
            hyperpar_values = {'lr': [0.01, 0.005], 'momentum': [0.3, 0.9], 'gamma': [0.1, 0.01], 
                               'neighb_dist_weight': [1, 4, 8, 12], 
                               'w_chamfer_GT_pred': [0, 3], 'w_chamfer_pred_GTcontour': [0, 3],
                               'w_chamfer_GTcontour_pred': [1, 4, 8, 12], 'w_chamfer_pred_GTcontour': [1, 4, 8, 12]}
        else: print("\n"*10 + "In order to use hyperpar_option 13, you need predict_uv_or_xyz 'uvy' and unsupervised on" + "\n"*10)
    if hyperpar_option==14:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==1:
            hyperpar_values = {'lr': [0.01, 0.005], 'momentum': [0.3, 0.9], 'gamma': [0.1, 0.01], 
                               'neighb_dist_weight': [10, 100], 
                               'w_chamfer_GTcontour_pred': [1, 4], 'w_chamfer_pred_GTcontour': [1, 4]}
        else: print("\n"*10 + "In order to use hyperpar_option 14, you need predict_uv_or_xyz 'uvy' and unsupervised on" + "\n"*10)
    if hyperpar_option==15:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==1:
            hyperpar_values = {'lr': [0.01, 0.005], 'momentum': [0.3, 0.9], 'gamma': [0.1, 0.01], 
                               'neighb_dist_weight': [10, 100, 500], 
                               'w_chamfer_GTcontour_pred': [1], 'w_chamfer_pred_GTcontour': [5, 10, 20]}
        else: print("\n"*10 + "In order to use hyperpar_option 15, you need predict_uv_or_xyz 'uvy' and unsupervised on" + "\n"*10)
    if hyperpar_option==16:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==1:
            hyperpar_values = {'lr': [0.01, 0.005], 'momentum': [0.3, 0.9], 'gamma': [0.1, 0.01], 
                               'neighb_dist_weight': [1, 2, 5, 10], 
                               'w_chamfer_GTcontour_pred': [1], 'w_chamfer_pred_GTcontour': [1, 2]}
        else: print("\n"*10 + "In order to use hyperpar_option 16, you need predict_uv_or_xyz 'uvy' and unsupervised on" + "\n"*10)
    if hyperpar_option==17: # Good for DS11 8x12 supervised uvy 
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==0:
            hyperpar_values = {'lr': [0.01, 0.005, 0.01, 0.01], 'momentum': [0.3, 0.9, 0.5, 0.8], 'gamma': [0.1, 0.01, 0.1, 0.01], 
                               'neighb_dist_weight': [1, 2, 5, 10], 
                               'w_uv': [1], 'w_xyz': [1, 2, 5, 10]}
        else: print("\n"*10 + "In order to use hyperpar_option 17, you need predict_uv_or_xyz 'uvy' and unsupervised off" + "\n"*10)
    if hyperpar_option==18:
        if args.dataset_number == '15':
            lr_list = [10**-x for x in range(0, 5, 1)] + [10**-x for x in range(5, 48, 7)]
            hyperpar_values = {'lr': lr_list, 'momentum': [0.1], 'gamma': [0.1], 
                               'neighb_dist_weight': [1, 2, 5, 10], 
                               'w_uv': [1], 'w_xyz': [1, 2, 5, 10]}
        else: print("\n"*10 + "In order to use hyperpar_option 18, you need to use dataset_number 15" + "\n"*10)
    if hyperpar_option==19:
        if args.dataset_number == '15':
            hyperpar_values = {'lr': [5, 0.01, 0.005, 0.01]*2, 'momentum': [0.3, 0.9, 0.5, 0.8]*2, 'gamma': [0.1, 0.01, 0.1, 0.01]*2, 
                               'neighb_dist_weight': [1, 2, 5, 10]*2, 
                               'w_uv': [1], 'w_xyz': [1, 2, 5, 10]*2}
        else: print("\n"*10 + "In order to use hyperpar_option 19, you need to use dataset_number 15" + "\n"*10)
    if hyperpar_option==20:
        if args.dataset_number == '15':
            hyperpar_values = {'lr': [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]*4, 'momentum': [0.3, 0.9, 0.5, 0.8]*8, 
                               'gamma': [0.1, 0.01]*16, 'neighb_dist_weight': [1, 2, 5, 10]*8,
                               'w_uv': [1], 'w_xyz': [1, 2, 5, 10]*8}
        else: print("\n"*10 + "In order to use hyperpar_option 20, you need to use dataset_number 15" + "\n"*10)
    if hyperpar_option==21:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==0:
            hyperpar_values = {'lr': [0.01, 0.005]*4, 'momentum': [0.3, 0.9, 0.5, 0.8]*2, 'gamma': [0.1, 0.01]*4, 
                               'neighb_dist_weight': [1, 2, 5, 10]*2, 
                               'w_uv': [1]*8, 'w_xyz': [1, 2, 5, 10]*2}
        else: print("\n"*10 + "In order to use hyperpar_option 21, you need predict_uv_or_xyz 'uvy' and unsupervised off" + "\n"*10)
    if hyperpar_option==22:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==1:
            hyperpar_values = {'lr': [0.1, 0.01, 0.01, 0.005]*4, 'momentum': [0.3, 0.9, 0.5, 0.8]*4, 'gamma': [0.1, 0.01]*8, 
                               'neighb_dist_weight': [1, 4, 8, 12]*4, 
                               'w_chamfer_GT_pred': [1, 2, 4, 8]*4, 'w_chamfer_pred_GTcontour': [1, 2, 4, 8]*4,
                               'w_chamfer_GTcontour_pred': [1, 2, 4, 8]*4, 'w_chamfer_pred_GTcontour': [1, 2, 4, 8]*4}
        else: print("\n"*10 + "In order to use hyperpar_option 22, you need predict_uv_or_xyz 'uvy' and unsupervised on" + "\n"*10)
    if hyperpar_option==23:
        if args.predict_uv_or_xyz == 'uvy' and args.unsupervised==1:
            hyperpar_values = {'lr': [0.1, 0.01, 0.01, 0.005], 'momentum': [0.3, 0.9, 0.5, 0.8], 'gamma': [0.1, 0.01]*2, 
                               'neighb_dist_weight': [1, 4, 8, 12], 
                               'w_chamfer_GT_pred': [1, 2, 4, 8], 'w_chamfer_pred_GTcontour': [1, 2, 4, 8],
                               'w_chamfer_GTcontour_pred': [1, 2, 4, 8], 'w_chamfer_pred_GTcontour': [1, 2, 4, 8]}
        else: print("\n"*10 + "In order to use hyperpar_option 23, you need predict_uv_or_xyz 'uvy' and unsupervised on" + "\n"*10)
    elif hyperpar_option==24: # for supervised uv prediction
        hyperpar_values = {'lr': [0.01, 0.005, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                       'neighb_dist_weight': [0, 1, 2]}
    elif hyperpar_option==25: # 1st time using new loss weights; for unsupervised uvy prediction
        hyperpar_values = {'lr': [100, 10, 0.1, 0.01, 0.001], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                           'loss_w_chamfer_GT_pred': [1], 'loss_w_chamfer_pred_GT': [1], 'loss_w_geo': [2, 100, 200, 400, 800]}
    elif hyperpar_option==26: # using new loss weights; for unsupervised uvy prediction
        hyperpar_values = {'lr': [1, 0.1, 0.01, 0.001, 0.1, 0.1, 0.1], 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01],
                           'loss_w_chamfer_GT_pred': [1], 'loss_w_chamfer_pred_GT': [1], 'loss_w_geo': [2, 40, 80, 100, 200, 400, 800]}
    elif hyperpar_option==27: # to try 1 loss at a time at performing depth prediction only
        lr_list = [10**-x for x in range(0, 22, 3)] + [10**x for x in range(2, 22, 3)]
        hyperpar_values = {'lr': lr_list, 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01]}
    elif hyperpar_option==28: # try different learning rates
        lr_list = [10**-x for x in range(0, 22, 4)] + [0.1, 0.01, 10] + [10**x for x in range(2, 22, 4)]
        hyperpar_values = {'lr': lr_list, 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01], 'step_size':[2, 4, 6]}
    elif hyperpar_option==29: # try different learning rates
        lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] 
        hyperpar_values = {'lr': lr_list, 'momentum': [0.3, 0.6, 0.9], 'gamma': [0.9, 0.5, 0.1, 0.05, 0.01], 'step_size':[2, 4, 6]}
    elif hyperpar_option==30: # try different weights
        weights_to_try = [1, 3, 5, 8, 20, 100, 500]*6
        lr_list = [0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.01] * 6 
        hyperpar_values = {'lr': lr_list, 'loss_w_uv': [1], 'loss_w_xyz': weights_to_try, 
                           'loss_w_chamfer_GT_pred': weights_to_try, 'loss_w_chamfer_pred_GT': weights_to_try, 
                           'loss_w_chamfer_GTcontour_pred': weights_to_try, 'loss_w_geo': weights_to_try, 
                           'loss_w_horizConsecEdges': weights_to_try}
    elif hyperpar_option==31: # try different weights
        weights_to_try = [1, 3, 5, 8, 20, 100, 500]*6
        hyperpar_values = {'loss_w_uv': [1], 'loss_w_xyz': weights_to_try, 
                           'loss_w_chamfer_GT_pred': weights_to_try, 'loss_w_chamfer_pred_GT': weights_to_try, 
                           'loss_w_chamfer_GTcontour_pred': weights_to_try, 'loss_w_geo': weights_to_try, 
                           'loss_w_horizConsecEdges': weights_to_try}
    elif hyperpar_option==32: # try different weights
        print("\n"*20 + "Do not repeat weights as in hyper32. Rather, pick similar but not the same values, instead." + "\n"*20)
        weights_to_try = [5, 20, 500]*6
        hyperpar_values = {'loss_w_uv': [1], 'loss_w_xyz': weights_to_try, 
                           'loss_w_chamfer_GT_pred': weights_to_try, 'loss_w_chamfer_pred_GT': weights_to_try, 
                           'loss_w_chamfer_GTcontour_pred': weights_to_try, 'loss_w_geo': weights_to_try, 
                           'loss_w_horizConsecEdges': weights_to_try}
    elif hyperpar_option==33: # try different weights
        def value_to_list(x):
            llista=[x, 5*x, 10*x, 50*x, 100*x, 200*x, 500*x, 1000*x, 5000*x]
            return [int(m) for m in llista]
        weights_uv = value_to_list(885/800)
        weights_xyz = value_to_list(885/0.1)
        weights_chamfer_GT_pred = value_to_list(885/24)
        weights_chamfer_pred_GT = value_to_list(885/24)
        weights_chamfer_GTcontour_pred = value_to_list(885/37)
        weights_geo = value_to_list(885/0.09)
        weights_horizConsecEdges = value_to_list(885/0.1)
        hyperpar_values = {'loss_w_uv': weights_uv, 'loss_w_xyz': weights_xyz, 
                           'loss_w_chamfer_GT_pred': weights_chamfer_GT_pred, 'loss_w_chamfer_pred_GT': weights_chamfer_pred_GT, 
                           'loss_w_chamfer_GTcontour_pred': weights_chamfer_GTcontour_pred, 'loss_w_geo': weights_geo, 
                           'loss_w_horizConsecEdges': weights_horizConsecEdges}
    elif hyperpar_option==34: # try different weights
        def value_to_list(x):
            llista=[x, 2*x, 5*x, 10*x]
            return [int(m) for m in llista]
        weights_uv = value_to_list(885/800)
        weights_xyz = value_to_list(885/0.1)
        weights_chamfer_GT_pred = value_to_list(885/24)
        weights_chamfer_pred_GT = value_to_list(885/24)
        weights_chamfer_GTcontour_pred = value_to_list(885/37)
        weights_geo = value_to_list(885/0.09)
        weights_horizConsecEdges = value_to_list(885/0.1)
        hyperpar_values = {'loss_w_uv': weights_uv, 'loss_w_xyz': weights_xyz, 
                           'loss_w_chamfer_GT_pred': weights_chamfer_GT_pred, 'loss_w_chamfer_pred_GT': weights_chamfer_pred_GT, 
                           'loss_w_chamfer_GTcontour_pred': weights_chamfer_GTcontour_pred, 'loss_w_geo': weights_geo, 
                           'loss_w_horizConsecEdges': weights_horizConsecEdges}
    elif hyperpar_option==35: # try different weights
        def value_to_list(x):
            llista=[x, 2*x, 3*x, 4*x, 5*x, 6*x, 10*x, 20*x, 50*x]
            return [int(m) for m in llista]
        weights_uv = value_to_list(885/800)
        weights_xyz = value_to_list(885/0.1)
        weights_chamfer_GT_pred = value_to_list(885/24)
        weights_chamfer_pred_GT = value_to_list(885/24)
        weights_chamfer_GTcontour_pred = value_to_list(885/37)
        weights_geo = value_to_list(885/0.09)
        weights_horizConsecEdges = value_to_list(885/0.1)
        hyperpar_values = {'loss_w_uv': weights_uv, 'loss_w_xyz': weights_xyz, 
                           'loss_w_chamfer_GT_pred': weights_chamfer_GT_pred, 'loss_w_chamfer_pred_GT': weights_chamfer_pred_GT, 
                           'loss_w_chamfer_GTcontour_pred': weights_chamfer_GTcontour_pred, 'loss_w_geo': weights_geo, 
                           'loss_w_horizConsecEdges': weights_horizConsecEdges}
    elif hyperpar_option==36: # try different weights and lr
        def value_to_list(x):
            llista=[x, 2*x, 3*x, 4*x, 5*x, 6*x, 10*x, 20*x, 50*x]
            return [int(m) for m in llista]
        weights_uv = value_to_list(885/800)
        weights_xyz = value_to_list(885/0.1)
        weights_chamfer_GT_pred = value_to_list(885/24)
        weights_chamfer_pred_GT = value_to_list(885/24)
        weights_chamfer_GTcontour_pred = value_to_list(885/37)
        weights_geo = value_to_list(885/0.09)
        weights_horizConsecEdges = value_to_list(885/0.1)
        hyperpar_values = {'lr': [100, 10, 1, 0.1, 0.01, 0.01, 0.01, 0.005, 0.001], 
                           'step_size': [2, 4, 6],
                           'loss_w_uv': weights_uv, 'loss_w_xyz': weights_xyz, 
                           'loss_w_chamfer_GT_pred': weights_chamfer_GT_pred, 'loss_w_chamfer_pred_GT': weights_chamfer_pred_GT, 
                           'loss_w_chamfer_GTcontour_pred': weights_chamfer_GTcontour_pred, 'loss_w_geo': weights_geo, 
                           'loss_w_horizConsecEdges': weights_horizConsecEdges}
    elif hyperpar_option==37: # Hausdorff weights
        hyperpar_values = {'lr': [1, 0.8, 0.7, 0.5, 0.2, 0.1, 0.01], 
                           'step_size': [2, 3, 4, 5, 6],
                           'loss_w_chamfer_GT_pred': [1, 2, 3, 4, 5, 6, 7, 8], 
                           'loss_w_chamfer_pred_GT': [1, 2, 3, 4, 5, 6, 7, 8], 
                           'loss_w_chamfer_GTcontour_pred': [1, 2, 3, 4, 5, 6, 7, 8], 
                           'gamma': [0.1, 0.3, 0.5]}
    elif hyperpar_option==38: # uv-xyz
        hyperpar_values = {'lr': [1, 0.8, 0.7, 0.5, 0.2, 0.1, 0.01], 
                           'step_size': [2, 3, 4, 5, 6],
                           'loss_w_uv': [1], 
                           'loss_w_xyz': [1, 5, 10, 50, 100, 500, 1000, 5000], 
                           'gamma': [0.1, 0.3, 0.5]}
    elif hyperpar_option==39: # uv-xyz
        hyperpar_values = {'lr': [0.05, 0.01], 
                           'step_size': [4, 6],
                           'loss_w_uv': [1], 
                           'loss_w_xyz': [1, 5, 10, 50, 100, 500, 1000, 5000], 
                           'gamma': [0.1, 0.3, 0.5]}
    elif hyperpar_option==40: # uv-xyz
#         I chose the following hyperparameters to create hyper40 from the observations of hyper39
#         2x3: Best hyper from hyper39: lr0.01_gamma0.3_stepSize4_Wuv1_Wxyz500
#         8x12: Best hyper from hyper39: lr0.01_gamma0.3_stepSize4_Wuv1_Wxyz100

        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02], 
                           'step_size': [3, 4, 4, 4, 4, 4, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_xyz': [100, 200, 300, 400, 500, 600, 700, 800], 
                           'gamma': [0.1, 0.3, 0.5]}
    elif hyperpar_option==41: # uv-Hausdorff 2D only
        # 41>42>43, where x>y means that x produce a better model than y
        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01], 
                           'step_size': [3, 4, 4, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [20, 50, 100, 200, 400], 
                           'loss_w_chamfer_pred_GT': [20, 50, 100, 200, 400], 
                           'loss_w_chamfer_GTcontour_pred': [20, 50, 100, 200, 400], 
                           'gamma': [0.2, 0.3, 0.3, 0.3, 0.4]}
    elif hyperpar_option==42: # 8x12: uv-Hausdorff 2D only
        # 41>42>43, where x>y means that x produce a better model than y
        hyperpar_values = {'lr': [0.1, 0.05, 0.03, 0.02, 0.01], 
                           'step_size': [4, 5, 5, 5, 6],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [50, 200, 400, 600, 800], 
                           'loss_w_chamfer_pred_GT': [50, 200, 400, 600, 800], 
                           'loss_w_chamfer_GTcontour_pred': [20, 50, 50, 100, 200], 
                           'gamma': [0.3, 0.3, 0.3, 0.4, 0.4]}
    elif hyperpar_option==43: # 8x12: uv-Hausdorff 2D only
        # 41>42>43, where x>y means that x produce a better model than y
        hyperpar_values = {'lr': [0.05, 0.03, 0.02, 0.01], 
                           'step_size': [4, 5, 5, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [50, 100, 150, 200], 
                           'loss_w_chamfer_pred_GT': [600, 800, 1000, 1200], 
                           'loss_w_chamfer_GTcontour_pred': [100, 200, 400, 600], 
                           'gamma': [0.3, 0.3, 0.4, 0.4, 0.4]}
    elif hyperpar_option==44: # 8x12: Hausdorff vs geo (unsupervised)
        hyperpar_values = {'loss_w_geo': [10, 100, 500, 1000, 5000, 10000]}
    elif hyperpar_option==45: # 8x12: best combination from hyper40
        hyperpar_values = {'lr': [0.01], 'gamma': [0.3], 'step_size': [4],
                           'loss_w_uv': [1], 'loss_w_xyz': [100]} 
    elif hyperpar_option==46: # all losses except for 3D self-supervised
        weights_Hausdorff = [20, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700]
        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01], 
                           'step_size': [3, 4, 4, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': weights_Hausdorff,
                           'loss_w_chamfer_pred_GT': weights_Hausdorff,
                           'loss_w_chamfer_GTcontour_pred': weights_Hausdorff,
                           'loss_w_xyz': [100, 200, 300, 500, 700, 1000, 1500, 2000, 5000, 10000], 
                           'gamma': [0.2, 0.3, 0.3, 0.3, 0.4]}
    elif hyperpar_option==47: # all losses except for angles
        # CAVEAT: Running it like this, all three Hausdorff weights will be the same in every run.
        weights_Hausdorff = [20, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700]
        # Try hihger values, because 5ann+27nonAnn gives almost constant uv predictions
        # Make them different for the different Hausdorff weights too
        
        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01], 
                           'step_size': [3, 4, 4, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': weights_Hausdorff,
                           'loss_w_chamfer_pred_GT': weights_Hausdorff,
                           'loss_w_chamfer_GTcontour_pred': weights_Hausdorff,
                           'loss_w_xyz': [500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 9000, 10000, 15000], 
                           'loss_w_geo': [10, 100, 500, 1000, 2000, 5000, 7000, 8000, 10000, 12000, 15000, 30000],
                           'gamma': [0.2, 0.3, 0.3, 0.3, 0.4]}
    elif hyperpar_option==48: # all losses except for angles
        # Changes from hyper47:
        # Try hihger values, because 5ann+27nonAnn gives almost constant uv predictions
        # Make them different for the different Hausdorff weights too
        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01], 
                           'step_size': [3, 4, 4, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [150, 200, 250, 300, 350, 400, 500, 600, 700],
                           'loss_w_chamfer_pred_GT': [150, 200, 250, 300, 350, 400, 500, 600, 700],
                           'loss_w_chamfer_GTcontour_pred': [150, 200, 250, 300, 350, 400, 500, 600, 700],
                           'loss_w_xyz': [500, 700, 1000, 1500, 2000, 3000, 4000, 8000, 9000, 10000], 
                           'loss_w_geo': [500, 1000, 2000, 5000, 7000, 10000, 15000],
                           'gamma': [0.2, 0.3, 0.3, 0.3, 0.4]}
    elif hyperpar_option==49: # all losses except for angles
        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01], 
                           'step_size': [3, 3, 3, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400],
                           'loss_w_chamfer_pred_GT': [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400],
                           'loss_w_chamfer_GTcontour_pred': [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400],
                           'loss_w_xyz': [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400], 
                           'loss_w_geo': [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400],
                           'loss_w_horizConsecEdges': [0, 1, 2, 5],
                           'gamma': [0.2, 0.3, 0.3, 0.3, 0.4]}
    elif hyperpar_option==50: # all losses 
        hyperpar_values = {'lr': [0.02, 0.01, 0.01, 0.01, 0.01], 
                           'step_size': [3, 3, 3, 4, 5],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'loss_w_chamfer_pred_GT': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'loss_w_chamfer_GTcontour_pred': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'loss_w_xyz': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                           'loss_w_geo': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'loss_w_horizConsecEdges': [0, 1, 2, 5],
                           'gamma': [0.2, 0.3, 0.3, 0.3, 0.4]}
    elif hyperpar_option==51: # slight modification of best values from hyper50
#         Best value from hper50:
#         lr0.01_gamma0.3_stepSize4_WchamferGTpred3_WchamferpredGT4_WchamferGTcontourpred1_Wgeo4_WhorizEdges1_Wuv1_Wxyz6
        hyperpar_values = {'lr': [0.01], 
                           'step_size': [4],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [3],
                           'loss_w_chamfer_pred_GT': [4],
                           'loss_w_chamfer_GTcontour_pred': [2, 3], # different from the best value so far: 1
                           'loss_w_xyz': [6], 
                           'loss_w_geo': [4],
                           'loss_w_horizConsecEdges': [2, 5], # different from the best value so far: 1
                           'gamma': [0.3]}
    elif hyperpar_option==52: # best values from hyper50 including angles between vertically adjacent edges
#         Best value from hper50:
#         lr0.01_gamma0.3_stepSize4_WchamferGTpred3_WchamferpredGT4_WchamferGTcontourpred1_Wgeo4_WhorizEdges1_Wuv1_Wxyz6
        hyperpar_values = {'lr': [0.01], 
                           'step_size': [4],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [3],
                           'loss_w_chamfer_pred_GT': [4],
                           'loss_w_chamfer_GTcontour_pred': [1],
                           'loss_w_xyz': [6], 
                           'loss_w_geo': [4],
                           'loss_w_horizConsecEdges': [1],
                           'loss_w_verConsecEdges': [1],
                           'gamma': [0.3]}
    elif hyperpar_option==53: # slight modification of hpyer52 to allow for different values of horizontal and vertical angle weights to be tested
        weight_list = [0.5, 0.1]
        hyperpar_values = {'lr': [0.01], 
                           'step_size': [4],
                           'loss_w_uv': [1], 
                           'loss_w_chamfer_GT_pred': [3],
                           'loss_w_chamfer_pred_GT': [4],
                           'loss_w_chamfer_GTcontour_pred': [1],
                           'loss_w_xyz': [6], 
                           'loss_w_geo': [4],
                           'loss_w_horizConsecEdges': weight_list,
                           'loss_w_verConsecEdges': weight_list,
                           'gamma': [0.3]}
    elif hyperpar_option==54: # trying different hyperparameters when uv_normalization is on and only uv loss is used
        hyperpar_values = {'lr': [0.01, 0.005, 0.001, 0.0005, 0.0001]*3, 
                           'step_size': [4, 4, 5, 6, 7]*3,
                           'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                           'loss_factor':[1., 10., 100.]}
    elif hyperpar_option==55: # Normalization: uv, D; Losses: uv, D
        hyperpar_values = {'lr': [0.001, 0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001], 
                           'step_size': [3, 4, 4, 5, 6, 6, 7],
                           'gamma': [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                           'loss_w_uv': [1, 2, 3, 5, 7, 9, 15],
                           'loss_w_D': [1, 2, 3, 5, 7, 9, 15]}
    elif hyperpar_option==56: # Normalization: uv, D; Losses: uv, D
        # hyper 55 Loss keeps decreasing --> in hyper 56, try hyper similar to best values of hyper 55: 
        # lr0.0005_gamma0.8_stepSize4_Wuv15, with different loss_w_D as well.
        hyperpar_values = {'lr': [0.0007, 0.0006, 0.0005, 0.0004], 
                           'step_size': [3, 4, 5, 6],
                           'gamma': [0.7, 0.8, 0.9],
                           'loss_w_uv': [5, 10, 15, 20],
                           'loss_w_D': [1, 5, 10, 15]}
    elif hyperpar_option==57: # Normalization: uv, D; Losses: uv, D
        # hyper 56 Loss keeps decreasing --> in hyper 57, try hyper similar to best values of hyper 56:
        #  lr0.0006_gamma0.7_stepSize4_Wuv5_WD5 
        #  lr0.0005_gamma0.8_stepSize6_Wuv20_WD10 
        # ratios of loss_w_uv/loss_w_D=15 are too large
        hyperpar_values = {'lr': [0.0006, 0.00058, 0.00054, 0.0005], 
                           'step_size': [4, 5, 5, 6],
                           'gamma': [0.7, 0.73, 0.76, 0.8],
                           'loss_w_uv': [1, 2, 3, 5],
                           'loss_w_D': [1, 2, 3, 5]}
    elif hyperpar_option==58: # Normalization: uv, D, xyz; Losses: uv, D, geo
        # like hyper 57, plus loss_w_geo
        hyperpar_values = {'lr': [0.0006, 0.00058, 0.00054, 0.0005], 
                           'step_size': [4, 5, 5, 6],
                           'gamma': [0.7, 0.73, 0.76, 0.8],
                           'loss_w_uv': [1, 2, 3, 5],
                           'loss_w_D': [1, 2, 3, 5],
                           'loss_w_geo': [1, 2, 3, 5]}
    elif hyperpar_option==59: # Normalization: uv, D, xyz; Losses: uv, D, angles(horiz & vert)
        # like hyper 58, changing loss_w_geo by loss_w_horizConsecEdges and loss_w_verConsecEdges
        hyperpar_values = {'lr': [0.0006, 0.00058, 0.00054, 0.0005], 
                           'step_size': [4, 5, 5, 6],
                           'gamma': [0.7, 0.73, 0.76, 0.8],
                           'loss_w_uv': [1, 2, 3, 5],
                           'loss_w_D': [1, 2, 3, 5],
                           'loss_w_horizConsecEdges': [1, 2],
                           'loss_w_verConsecEdges': [1, 2]}
    elif hyperpar_option==60: # Normalization: uv, D, xyz; Losses: uv, D, Hausdorff, geo, angles(horiz & vert)
        # like hyper 59, adding geo and hausdorff
        hyperpar_values = {'lr': [0.0007, 0.0006, 0.0006, 0.0005]*2, 
                           'step_size': [4, 4, 5, 6]*2,
                           'gamma': [0.7, 0.75, 0.8, 0.85]*2,
                           'loss_w_uv': [1, 5, 8, 10]*2,
                           'loss_w_D': [1, 2, 3, 5]*2,
                           'loss_w_chamfer_GT_pred': [1, 2, 3, 4]*2,
                           'loss_w_chamfer_pred_GT': [1, 2, 3, 4]*2,
                           'loss_w_chamfer_GTcontour_pred': [1, 2, 3, 4]*2,
                           'loss_w_geo': [1, 2, 1, 2]*2,
                           'loss_w_horizConsecEdges': [1, 2, 1, 2]*2,
                           'loss_w_verConsecEdges': [1, 2, 1, 2]*2}
    elif hyperpar_option==61: # Finetuning on tshirt: uv, D, xyz; Losses: uv, D, Hausdorff, geo
        # Similar to hyper60, without angles
        hyperpar_values = {'lr': [0.0007, 0.0004, 0.0002, 0.0001]*2, 
                           'step_size': [3, 4, 5, 6]*2,
                           'gamma': [0.7, 0.75, 0.8, 0.85]*2,
                           'loss_w_uv': [5, 6, 7, 8, 9, 10, 11, 12],
                           'loss_w_D': [2, 3, 4, 5]*2,
                           'loss_w_chamfer_GT_pred': [1, 2, 3, 4]*2,
                           'loss_w_chamfer_pred_GT': [1, 2, 3, 4]*2,
                           'loss_w_chamfer_GTcontour_pred': [1, 2, 3, 4]*2,
                           'loss_w_geo': [1, 2, 1, 2]*2}   
        # The following values won for tshirt finetuning on 0.99 of training set being non-annotated
        # lr0.0004_gamma0.8_stepSize6_WchamferGTpred1_WchamferpredGT4_WchamferGTcontourpred4_Wgeo1_Wuv6_WD4
    elif hyperpar_option==62: # Normalization: uv, D, xyz; Losses: uv, D, Hausdorff, geo
        # similar to best of hyper 60, without angles(horiz & vert), more weight to uv parts
        # where we consider best of hyper60:
        # lr0.0005_gamma0.8_stepSize4_WchamferGTpred1_WchamferpredGT4_WchamferGTcontourpred4_Wgeo1_WhorizEdges2_WverEdges2_Wuv5_WD3
        hyperpar_values = {'lr': [0.0005, 0.0005, 0.0004, 0.0003]*2, 
                           'step_size': [3, 4, 5, 6]*2,
                           'gamma': [0.7, 0.75, 0.8, 0.85]*2,
                           'loss_w_uv': [5, 5, 8, 10]*2,
                           'loss_w_D': [1, 2, 2, 3, 3, 4, 4, 5],
                           'loss_w_chamfer_GT_pred': [1, 2, 3, 4]*2,
                           'loss_w_chamfer_pred_GT': [3, 4, 5, 6]*2,
                           'loss_w_chamfer_GTcontour_pred': [3, 4, 5, 6]*2,
                           'loss_w_geo': [1, 2, 1, 2]*2}
    elif hyperpar_option==63: # Normalization: uv, D, xyz; Losses: uv, D
        # similar to best of hyper 63, without Hausdorff and geo losses
        # recall that we consider best of hyper60:
        # lr0.0005_gamma0.8_stepSize4_WchamferGTpred1_WchamferpredGT4_WchamferGTcontourpred4_Wgeo1_WhorizEdges2_WverEdges2_Wuv5_WD3
        hyperpar_values = {'lr': [0.0006, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0003, 0.0003], 
                           'step_size': [3, 4, 5, 6]*2,
                           'gamma': [0.7, 0.75, 0.8, 0.85]*2,
                           'loss_w_uv': [1, 3, 5, 8, 10, 20, 40, 80],
                           'loss_w_D': [1]}
        
    return OrderedDict(sorted(hyperpar_values.items(), key=lambda t: t[0]))

from datetime import datetime, timedelta

def seconds_to_dhms(sec):
    sec = timedelta(seconds=round(sec))
    d = datetime(1,1,1) + sec
    days, hours, minutes, seconds = d.day-1, d.hour, d.minute, d.second
    return days, hours, minutes, seconds

def create_dataloaders(transformed_dataset, args):
    """
    For TowelWall datasets, we use random_split_notMixingSequences because
    mixing all frames from all sequences together and then randomly picking some 
    for training/val/test may make all sets contain extremelly similar observations, 
    since contiguous frames of a same video sequence can be very similar.
    Hence, we keep all frames of a sequence in the same set, out of the sets train/val/test.
    
    # So, for TowelWall datastes, randomly splitting video sequences into train/val/(test), 
    # and then randomizing the ordering of all frames of all sequences in 'train' and doing the same with 'val' and 'test'.
    # Every random choice has a seed for reproducibility.

    Indices of transformed_dataset_parts[0] will consist of range(training_size)
    Indices of transformed_dataset_parts[1] will consist of range(validation_size)
    
    The previous approach not separating by video sequences consisted simply of this train/val split:
    random_seed = 1 # For reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    [transformed_dataset_parts[0], transformed_dataset_parts[1]] = \
    torch.utils.data.dataset.random_split(transformed_dataset, [training_size, validation_size])
    """
    if args.sequence_name == 'TowelWall':
        transformed_dataset_parts, dataset_sizes, dataloaders = data_loading.random_split_notMixingSequences( 
            dataset=transformed_dataset, args=args)
    elif args.sequence_name in ['DeepCloth', 'kinect_tshirt', 'kinect_paper']:
        transformed_dataset_parts, dataset_sizes, dataloaders = data_loading.random_split_DeepCloth( 
            dataset=transformed_dataset, args=args)
    return transformed_dataset_parts, dataset_sizes, dataloaders
    
def print_split_info(transformed_dataset, transformed_dataset_parts, dataset_sizes, dataloaders, verbose=0):
    print('Length of transformed_dataset:', len(transformed_dataset))
    print('Length of each part:', dataset_sizes)
    # Check the indexing system works with this split
    if verbose==1:
        print()
        print('First indices of training part of the permutation:', transformed_dataset_parts[0].indices[0:5])
        idx_in_training_set = 0
        idx_in_whole_dataset = transformed_dataset_parts[0].indices[idx_in_training_set]
        sample_from_training = transformed_dataset_parts[0][idx_in_training_set]
        sample_from_whole_dataset = transformed_dataset[idx_in_whole_dataset]
        print('Address of image in transformed_dataset_parts[0][' + str(idx_in_training_set) + ']:', sample_from_training['img_name'])
        print('idx_in_whole_dataset = transformed_dataset_parts[0].indices[' + 
              str(idx_in_training_set) + '] = ' + str(idx_in_whole_dataset))
        print('Address of image in transformed_dataset[' + str(idx_in_whole_dataset) + ']:', sample_from_whole_dataset['img_name'])