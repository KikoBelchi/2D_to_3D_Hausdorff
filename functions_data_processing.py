import argparse
import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv
import os
import pandas as pd
import pickle
from skimage import io
import torch
import torch.nn as nn
from torchvision import transforms


import data_loading
import diff_Hausdorff_loss
import functions_train

    
###
### Parser for data loading and training arguments
###
def parser():
    parser = argparse.ArgumentParser(description='Data loading, training and visualization arguments')
    
    # Dataset
    parser.add_argument('--sequence-name', type=str, default='TowelWall',
                        help="The dataset used is either 'Renders' + sequence_name + str(dataset_number), or sequence_name + str(dataset_number) if --sequence-name is 'DeepCloth' or 'kinect_tshirt' or 'kinect_paper'. (default: 'TowelWall').")
    parser.add_argument('--dataset-number', type=int, default=7, 
                        help="The dataset used is the one in 'Renders' + sequence_name + str(dataset_number) or in  sequence_name + str(dataset_number) if --sequence-name is 'DeepCloth'. (default: 7)")
    parser.add_argument('--texture-type', type=str, default='train_non-text',
                        help="DeepCloth2 consists of data with 4 different types of textures: 'train_low-text', 'train_non-text', 'train_struct-text' or 'train_text'. If one these optins is chosen, then only that data will be loaded in the dataset class. If the entered string is the empty string '', then all of them will be used to load a bigger dataset.")
    parser.add_argument('--towelPixelSample', type=int, default=0, 
                        help="--towelPixelSample 0 means the towel pixel (contour or not) subsample is computed every time a loss is computed. --towelPixelSample 1 means the towel pixel (contour or not) subsample is computed within the dataset class. In particular, we will not do different subsamples at every epoch in this case. --towelPixelSample 2 means the towel pixel (contour or not) subsample is computed before training, saved to a file and they are loaded from the file within the dataset class. In particular, we will not do different subsamples at every epoch in this case.")
    parser.add_argument('--GTtowelPixel', type=int, default=0, 
                        help="If --GTtowelPixel==0 (default), the set of towel pixels and the contour thereof is computed using thresholds on the input images. If --GTtowelPixel==1, the set of towel pixels and the contour thereof is computed using GT uv.")
    
    # Submesh
    parser.add_argument('--reordered-dataset', type=int, default=1, 
                        help='If reordered_dataset == 0, we only use the first --vertices vertices (in the order provided by Blender) from the 3D reconstruction. If reordered_dataset == 1, we only use submesh_num_vertices_vertical x submesh_num_vertices_horizontal vertices from the 3D reconstruction. (default: 0)')
    parser.add_argument('--num-selected-vertices', type=int, default=6, 
                        help='Number of vertices to use. It selects the first ones returned by blender. Number between 6 and 5356 with the current dataset')
    parser.add_argument('--submesh-num-vertices-vertical', type=int, default=2, 
                        help="submesh_num_vertices_vertical (it can be used if reordered_dataset == 1): number of vertices in the vertical direction which want to be selected. The vertices will be chosen so that the distance between any consecutive vertices is the same. (default: 2) It must be an integer in the interval [2, 52]. If --sequence-name is 'DeepCloth', then it must be an integer in [2, 9].")
    parser.add_argument('--submesh-num-vertices-horizontal', type=int, default=3, 
                        help="submesh_num_vertices_horizontal (it can be used if reordered_dataset == 1): number of vertices in the horizontal direction which want to be selected. The vertices will be chosen so that the distance between any consecutive vertices is the same. (default: 2). It must be an integer in the interval [2, 103]. If --sequence-name is 'DeepCloth', then it must be an integer in [2, 9].")

    # Training
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='input batch size for training (default: 4). During visualization, in visualize_predictions.py, dataloaders work with args.batch_size_to_show, and args.batch_size is only used for loading the trained model.')
    parser.add_argument('--num-epochs', type=int, default=30, 
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)') # Try 0.01
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9). 0<momentum<1') # Try 0.5
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help="--lr wil be multiplited by --gamma every --step-size epochs (default: 0.1)")
    parser.add_argument('--step-size', type=int, default=7, 
                        help="--lr wil be multiplited by --gamma every --step-size epochs (default: 7)")            
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-epochs', type=int, default=0, 
                        help='if --log-epochs is non-zero, the prediction of the first training and validation batches will be saved at each epoch mutiple of --log-epochs (0-indexing epochs). If --log-epochs==0, (default), no predictions will be saved.')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='number of workers to train (default: 4)')
    parser.add_argument('--hyperpar-option', type=int, default=0, 
                        help='--hyperpar-option is an integer which selects which combination of hyperparameters to try during training. --hyperpar-option==0 performs no hyperparameter optimization and simply trains with 1 choice of hyperparameters.')
    parser.add_argument('--save-all-models', type=int, default=0, 
                        help='When optimizing hyperparameters, setting this to 1 will save the model of each set of hyperparameters tried. If kept to 0, it will only save the best-performing model.')
    parser.add_argument('--round', type=int, default=1, 
                        help='In round 1, train a model (e.g., using only vertex coordinates in the loss). In round 2, use the model trained in round 1 as initialization, for using a different loss (e.g., include distance between vertices). (default: 1)')
    parser.add_argument('--round1-model-dir', type=str,
                       default='uvEpochs1_TowelWall8_2x3mesh_cameraCoord_noNormals_ResNet18_rectROI_lr0.01_mom0.3_gam0.5_uvNorm',
                       help="directory containing model trained in round 1.")
    parser.add_argument('--train-w-annot-only', type=int, default=0,
                        help="If there is a split of the training set into annotated and non-annotated data but we only want to train with the annotated data of that split, set this to 1.")
 
    # Architecture
    parser.add_argument('--resnet-version', type=int, default=18, 
                        help='ResNet version to use as feature extractor (default: 18, which generates 512 features)')
                        # Number of features extracted by ResNet18: 512
                        # Number of features extracted by ResNet152: 2048
    parser.add_argument('--frozen-resnet', type=int, default=0, 
                        help='frozen_resnet==1 means the weights from ResNet are frozen, so that we use the features obtained from a pretrained ResNet and only train with the weights of the last fully connected layer appended to the ResNet. frozen_resnet==0 (default) means that we are performing some finetuning, i.e., we include the weights from the ResNet in the training process.')
    parser.add_argument('--freeze-all-but-last-layer', type=int, default=0, 
                        help='1 means that only the last layer will be optimised. 0 means all layers will. This should be used when running round 2 training and wanting to finetune a round 1 model. If we are dealing with round 1, the parameter to choose to optimize the last layer only is --frozen-resnet')

#     parser.add_argument('--best-or-last', type=int, default=0, 
#                         help='best_or_last==0 keeps the model with the best validation loss during training. best_or_last==1 keeps the model obtained at the last epoch, regardless of whether it gives the best validation loss over the training period or not (default: 0)')
    parser.add_argument('--dropout-p', type=float, default=0,
                       help="Probability of turning a weight into 0 during Dropout in training mode. If 0 (default), then no Dropout is applied.")
    parser.add_argument('--permutation-dir', type=str, default=None,
                       help="directory containing the permutation to apply to the output layer of the network. (default: None) This only makes sense after a first round of unsupervised uv(y) prediction.")
    parser.add_argument('--relu-alternative', type=str, default='relu',
                       help="subsitute of all appearances of ReLu within the ResNet. Possible values include 'relu' (default), 'elu', 'leaky'. Do not try 'prelu', since this changed the architecture of the ResNet in a way that does not allow the pretrained weights of the net to be loaded.")
    
    # Operations on the data
    parser.add_argument('--crop-centre-or-ROI', type=int, default=0, 
                        help='Cropping option: crop_centre_or_ROI==0: centre crop. crop_centre_or_ROI==1: Squared box containing the towel. crop_centre_or_ROI==2: Rectangular box containing the towel. crop_centre_or_ROI==3: no crop (default: 0)')            
    parser.add_argument('--camera-coordinates', type=int, default=1, 
                help='camera_coordinates == 1 uses camera coordinates. camera_coordinates == 0 uses world coordinates')
    parser.add_argument('--GTxyz-from-uvy', type=int, default=0, 
                help='--GTxyz-from-uvy==1: Ground Truth xyz world coordinates are computed from the Ground Truth uv pixel coordinates and the GT y world coordinate, in float32 format. --GTxyz-from-uvy == 0 (default): Ground truth xyz world coordinates are those directly reported by Blender')
    parser.add_argument('--GTuv', type=int, default=0, 
                help='If --GTuv==0, the predicted uv is used for predictions. If --GTuv==1, the GT uv is used as uv prediction, and then we only need to predict depth (default: 0)')  
    parser.add_argument('--save-tensor', type=int, default=0, help="When running save_towel_pixel_subsamples.py, it had the following effect. save_tensor==1: Save in torch tensor format. save_tensor==0: Save in numpy array format. save_tensor==2: Do not save and simply plot. When running on any script which needs to instanciate a dataset class which needs to load the towel pixel subsamples, this indicates whether they are loaded from/as numpy arrays (0) or as tensors (1).")
    parser.add_argument('--f-u', type=float, default=600.172, help="focal length f_u used for converting between coordinate systems.")
    parser.add_argument('--f-v', type=float, default=600.172, help="focal length f_v used for converting between coordinate systems.")

    # Split training/validation/test
    parser.add_argument('--random-seed-to-choose-video-sequences', type=int, default=1, 
                        help='random seed to choose which video sequences the training set will consist of. (default: 1)')
    parser.add_argument('--random-seed-to-shuffle-training-frames', type=int, default=2, 
                        help='random seed to shuffle the frames of all video sequences within the training set. (default: 2)')
    parser.add_argument('--random-seed-to-shuffle-validation-frames', type=int, default=3, 
                        help='random seed to shuffle the frames of all video sequences within the validation set. (default: 3)')
    parser.add_argument('--random-seed-to-shuffle-test-frames', type=int, default=4, 
                        help='random seed to shuffle the frames of all video sequences within the test set. (default: 4)')
    parser.add_argument('--lengths-proportion-test', type=float, default=None,
                       help="lengths_proportion_train (float between 0 and 1): percentatge of the dataset to be set for training (default: 0.8). lengths_proportion_test (float between 0 and 1 or None): If lengths_proportion_test is None, the split is only made into train and validation as follows: lengths_proportion_train for training, and the rest for validation. If lengths_proportion_test is not None, then the split is done with: lengths_proportion_train for training, lengths_proportion_test for test and the rest for validation. Given the RenderTowelWall datasets and the small amount of hyperparameters to optimize, we can do a 0.8/0.1/0.1 train/val/test split. lengths_proportion_train_nonAnnotated (float between 0 and 1, default: None): when there is a part of the training set for which annotations are used and a part for which the annotations are not used, lengths_proportion_train_nonAnnotated is the percentatge of the training set for which no annotations will be used")
    parser.add_argument('--lengths-proportion-train', type=float, default=0.8,
                        help="lengths_proportion_train (float between 0 and 1): percentatge of the dataset to be set for training (default: 0.8). lengths_proportion_test (float between 0 and 1 or None): If lengths_proportion_test is None, the split is only made into train and validation as follows: lengths_proportion_train for training, and the rest for validation. If lengths_proportion_test is not None, then the split is done with: lengths_proportion_train for training, lengths_proportion_test for test and the rest for validation. Given the RenderTowelWall datasets and the small amount of hyperparameters to optimize, we can do a 0.8/0.1/0.1 train/val/test split. lengths_proportion_train_nonAnnotated (float between 0 and 1, default: None): when there is a part of the training set for which annotations are used and a part for which the annotations are not used, lengths_proportion_train_nonAnnotated is the percentatge of the training set for which no annotations will be used")
    parser.add_argument('--lengths-proportion-train-4visuals', type=float, default=None,
                        help="lengths_proportion_train (float between 0 and 1) to use when visualizing the predictions of a model which may have been trained with a different training set ratio than the one we want to use for visualization. This allows the comparison of models trained on different training set sizes.")
    parser.add_argument('--lengths-proportion-test-4visuals', type=float, default=None,
                        help="lengths_proportion_test (float between 0 and 1) to use when visualizing the predictions of a model which may have been trained with a different training set ratio than the one we want to use for visualization. This allows the comparison of models trained on different training set sizes.")
    parser.add_argument('--lengths-proportion-train-nonAnnotated', type=float, default=None,
                       help="lengths_proportion_train (float between 0 and 1): percentatge of the dataset to be set for training (default: 0.8). lengths_proportion_test (float between 0 and 1 or None): If lengths_proportion_test is None, the split is only made into train and validation as follows: lengths_proportion_train for training, and the rest for validation. If lengths_proportion_test is not None, then the split is done with: lengths_proportion_train for training, lengths_proportion_test for test and the rest for validation. Given the RenderTowelWall datasets and the small amount of hyperparameters to optimize, we can do a 0.8/0.1/0.1 train/val/test split. lengths_proportion_train_nonAnnotated (float between 0 and 1, default: None): when there is a part of the training set for which annotations are used and a part for which the annotations are not used, lengths_proportion_train_nonAnnotated is the percentatge of the training set for which no annotations will be used")
    parser.add_argument('--lastEpochSupOnly', type=int, default=0,
                       help="if the training set is split into annotated and non-annotated parts, keeping --lastEpochSupOnly to 0 (default) will use both annotated and non-annotated training data in the last epoch, setting --lastEpochSupOnly to 1 will make sure that in the last epoch, only the annotated part of the training set is used, setting --lastEpochSupOnly to 2 will make sure that in the last epoch, the best model between using in the last epoch only annotated data or both annotated and non-annotated data is chosen, where these two options are evaluated on the non_annotated training set, temporarily, which should be changed.")
    
    # Normalizations
    parser.add_argument('--normalization', type=int, default=2, 
                        help='normalization==1 means the barycenter of the vertices is sent to the origin and a rescaling is then applied so that the furthest vertex from the origin is at distance 1. normalization==2 (default) means no normalization. normalization==3 means 0<=x, y, z<=1 normalization is performed, using the min and max values of x, y, z in the training set.')
    parser.add_argument('--boxplots-on', type=int, default=0, help="If --boxplot-on==1, plot boxplot showing the min and max of x, y, z coordinates of all vertices within the training set (computed w or w/o outliers, depending of the choice in the code). Otherwise, do not plot the boxplot.")    
    parser.add_argument('--D-normalization', type=int, default=2, 
                        help='By analogy with --normalization, D_normalization==2 (default) means no normalization. D_normalization==3 means 0<=Depth<=1 normalization is performed, using the min and max values of Depth in the training set.')
    parser.add_argument('--normalize-datasetClass-D-min', type=float, default=None, 
                        help='--normalize-datasetClass-D-min is the min value of D to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-D-max', type=float, default=None, 
                        help='--normalize-datasetClass-D-max is the max value of D to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-x-min', type=float, default=None, 
                        help='--normalize-datasetClass-x-min is the min value of x to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-y-min', type=float, default=None, 
                        help='--normalize-datasetClass-y-min is the min value of y to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-z-min', type=float, default=None, 
                        help='--normalize-datasetClass-z-min is the min value of z to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-x-max', type=float, default=None, 
                        help='--normalize-datasetClass-x-max is the max value of x to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-y-max', type=float, default=None, 
                        help='--normalize-datasetClass-y-max is the max value of y to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-datasetClass-z-max', type=float, default=None, 
                        help='--normalize-datasetClass-z-max is the max value of z to be used in the creation of the dataset class, if we want to externally enforce this number.')
    parser.add_argument('--normalize-labelsPred-D-min', type=float, default=None, 
                        help='--normalize-labelsPred-D-min is the min value of D to be used in the creation of the labels and predictions, if we want to externally enforce this number. ')
    parser.add_argument('--normalize-labelsPred-D-max', type=float, default=None, 
                        help='--normalize-labelsPred-D-max is the max value of D to be used in the creation of the labels and predictions, if we want to externally enforce this number. ')
    parser.add_argument('--normalize-labelsPred-x-min', type=float, default=None, 
                        help='--normalize-labelsPred-x-min is the min value of x to be used in the creation of the labels and predictions, if we want to externally enforce this number.')
    parser.add_argument('--normalize-labelsPred-y-min', type=float, default=None, 
                        help='--normalize-labelsPred-y-min is the min value of y to be used in the creation of the labels and predictions, if we want to externally enforce this number.')
    parser.add_argument('--normalize-labelsPred-z-min', type=float, default=None, 
                        help='--normalize-labelsPred-z-min is the min value of z to be used in the creation of the labels and predictions, if we want to externally enforce this number.')
    parser.add_argument('--normalize-labelsPred-x-max', type=float, default=None, 
                        help='--normalize-labelsPred-x-max is the max value of x to be used in the creation of the labels and predictions, if we want to externally enforce this number.')
    parser.add_argument('--normalize-labelsPred-y-max', type=float, default=None, 
                        help='--normalize-labelsPred-y-max is the max value of y to be used in the creation of the labels and predictions, if we want to externally enforce this number.')
    parser.add_argument('--normalize-labelsPred-z-max', type=float, default=None, 
                        help='--normalize-labelsPred-z-max is the max value of z to be used in the creation of the labels and predictions, if we want to externally enforce this number.')
    parser.add_argument('--uv-normalization', type=int, default=0, 
                        help='--uv-normalization==1 normalizes the u, v pixel coordinates to 0<=u, v<=1. --uv-normalization==0 does not perform this normalization. --uv-normalization==2 normalizes the u, v pixel coordinates to 0<=u, v<=223. (default: 0)')
    parser.add_argument('--new-uv-norm', type=int, default=0, 
                        help='--new-uv-norm==1 expects the u and v outputs of the ResNet to be approximately within the interval [0,1], and then it multiplies the values by the width-1 and height-1 of the original picture, respectively, before computing the loss, in the sense of outputs=model(inputs), outputs=unnormalize(outputs). --new-uv-norm==0 does not perform any of the above. (default: 0)')
    parser.add_argument('--normalize-distance-adj-vertices', type=int, default=1, 
                        help='Relevant only if --neighbour-dist!="no". --normalize-distance-adj-vertices == 1 divides the distance between adjacent vertices by sqrt(D), where D is the number of coordinates of the vertices (2 or 3) or divide the square of this distance by D. --normalize-distance-adj-vertices == 0, does not. (default: 1)')
    
    parser.add_argument('--verbose', type=int, default=0, 
                        help='verbose==1 will print comments about the loss, etc. To turn them off, set verbose=0 (default: 0)')
    parser.add_argument('--find-xyz-norm-given-normalized-uvD', type=int, default=0,
                        help="This should only be 1 if we know the normalization parameters for u, v and D, and want to obtain the normalization parameters from the xyz obtained by (in progress...; not used yet)")    
    
    parser.add_argument('--testing-eval', type=int, default=0, 
                        help="--testing-eval==1 sets the model to evaluation mode instead of to training mode, just to compare the results on predictions on train and validation sets during training. In this mode, some possible BatchNorm or Dropout may be omitted while training. --testing-eval==0 will not do this, and hence it will set the model to training mode while training. --testing-eval==2 will freeze only the batch normalization within the ResNet to check whether that is the only thing that model.eval() is doing when setting that option during training. --testing-eval==3 sets the model to training mode during the validation phases of training and it will compute the final prediction on training, validation and test sets using model.train() mode on. (default: 0)")
    
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help="--gpu-device is a number from 0 to the number of gpus in your machine -1, and it tells the computations to be performed on that gpu; (these are 0-indexed) (default: 0)")
    parser.add_argument('--predict-uv-or-xyz', type=str, default='xyz',
                        help="--predict-uv-or-xyz=='xyz' predicts xyz coordinates of 3D reconstruction. --predict-uv-or-xyz=='uv' predicts uv coordinates (pixel location) of the selected mesh vertices. --predict-uv-or-xyz=='uvy' predicts uv coordinates (pixel location) of the selected mesh vertices and also its y world coordinate (Z camera coordinate with our RT matrix), from which we infer xyz world coordinates too. --predict-uv-or-xyz=='uvD' predicts uv coordinates (pixel location) of the selected mesh vertices and also their depth. The option 'uvD' should be used only with dataste DeeoCloth2, and not with RendersTowelWallxx. (default: 'xyz')")
    
    # Modifications to the loss
    parser.add_argument('--loss', type=int, default=0, 
                        help="torch.nn's MSELoss is used for the gradient descent computation during training. If loss == 1, my own MSE loss at each step of the training and prediction is also reported (but its gradient is not computed). If loss==0, this extra loss is not reported. (default: 0)")
    parser.add_argument('--loss-factor', type=float, default=1., help="the batch loss is multiplied by this factor to help training, but the epoch loss is divided by this value so that regardless of the loss-factor, we can compute losses. The reported batch losses on screen, on file and on tensorboard are not multiplied by this factor, again, so that we can compare performance on different loss-factor values.")
    parser.add_argument('--uv-loss-only', type=int, default=0, 
                        help="If --predict_uv_or_xyz=='uvy', then the following holds: if --uv-loss-only==0, both uv and xyz parts are used in the loss. If --uv-loss-only==1, only the uv part (and not the xyz part) is used in the loss. This is mostly useful in the unsupervised setting, if we want to first make the uv move towards the towel in the first round, and then add depth predictions (default: 0) We could actually get rid of this variable, as when --predict_uv_or_xyz=='uvy', setting --uv-loss-only to 1 is equivalent to setting --loss-w-uv to 1 and all other loss weights to 0. ")
    parser.add_argument('--loss-diff-Hauss', type=int, default=0, 
                        help="If 1, the only used loss will be the differentiable Hausdorff (or Chamfer) loss. (default: 0) Deprecated, do not use. Now, if --loss-weights==1, then the 'differentiable Hausdorff loss' is used, rather than the one computed manually.")
    parser.add_argument('--loss-weights', type=int, default=1, 
                        help="If 1, the weights are applied to the summands of the computed losses. In this case, every summand loss is also reported. If 0, the weights are applied to the tensors from which the loss is computed. (default: 1)")
    parser.add_argument('--testing-loss-computations', type=int, default=0,
                        help="--testing-loss-computations==1 will compute the losses of the prediction outputs=labels+2, rather than on outputs=model(inputs). --testing-loss-computations==0 (default) will not.")
    parser.add_argument('--squared', type=int, default=1, 
                help='This is only relevant if --neighb-dist-weight!=0. In this case, --squared == 1 uses the squared of the Euclidean distance between adjacent vertices in the loss, whereas --squared == 0 uses the Euclidean distance between adjacent vertices in the loss.')
    parser.add_argument('--hausdorff', type=int, default=0,
                        help="--hausdorff 0 uses Chamfer distance in the loss. --hausdorff 1 uses Hausdorff distance (default: 0) This applies only if --w-chamfer-GT-pred!=0.")
    parser.add_argument('--subsample-ratio', type=int, default=1,
                        help="when --w-chamfer-GT-pred!=0, the set of towel pixels in the RGB image will be subsampled. We will select 1 of every --subsample-ratio pixels. Recommended value: between 50 and 75 (default: 1, which means no subsampling). A value of -1 indicates the following: when making sure that every uv prediction is close to a towel pixel, we will use a number of args.num_selected_vertices of the towel pixels; when making sure that every towel pixel is close to a uv prediction, we wil luse a number args.num_selected_vertices//2 of the towel pixels plus a number args.num_selected_vertices//2 of the towel pixel contour. Adding this extra weight to the contour compensates for the fact that there are less points around the contour and therefore, the algorithm would try to keep uv prediction in the interior of the towel, away from the contour.")
    parser.add_argument('--subsample-ratio-contour', type=int, default=1,
                        help="when --w-chamfer-GT-pred!=0, the contour of the towel pixels in the RGB image will be subsampled. We will select 1 of every --subsample-ratio-contour pixels. (default: 1, which means no subsampling).")
    parser.add_argument('--subsample-size', type=int, default=6,
                        help="when some Hausdorff computation is needed, a subset of as many as --subsample-size towel pixels in the RGB image will be selected.")
    parser.add_argument('--subsample-size-contour', type=int, default=6,
                        help="when some Hausdorff computation is needed, a subset of as many as --subsample-size-contour pixels from the contour of the towel region in the RGB image will be selected.")
    parser.add_argument('--normals', type=int, default=0, 
                help='normals == 1 uses normal vectors in the loss. normals == 0 does not')
    parser.add_argument('--unsupervised', type=int, default=0, 
                help='--unsupervised 0: In the loss, we compare distance between vertices in each RGB with the distance between vertices of its corresponding mesh. --unsupervised 1: In the loss, we compare distance between vertices in each RGB with the distance between vertices of a template mesh. This applies only if --neighb-dist-weight!=0 (default: 0). The same happens with what is used as label for the angle between adjacent edges when --loss-w-horizConsecEdges!=0 or --loss-w-verConsecEdges!=0.')
    parser.add_argument('--n-epochs-pred-GT', type=int, default=0, 
                help='Number of epochs in which we try to get the predicted vertices close to the towel pixels, before starting doing the opposite for the rest of the epochs (default: 0)')
    
    # New loss weights, used to weight the different losses
    parser.add_argument('--normWeights1', type=int, default=1, 
                        help="If --normWeights1==1, the loss weights are normalized so that they sum 1. Otherwise, they are not. (default: 1).")
    parser.add_argument('--loss-w-uv', type=int, default=0, 
                        help="--loss-w-uv is the weight of the uv coordinates loss (see normalize_loss_weights()). Set --loss-w-uv=0 to train an unsupervised model (default: 0)")
    parser.add_argument('--loss-w-xyz', type=int, default=0, 
                        help="--loss-w-xyz is the weight of the xyz coordinates loss (see normalize_loss_weights()). Set --loss-w-xyz=0 to train an unsupervised model (default: 0)")
    parser.add_argument('--loss-w-D', type=int, default=0, 
                        help="--loss-w-D is the weight of the depth loss (see normalize_loss_weights()).(default: 0)")
    parser.add_argument('--loss-neighb-dist-weight', type=int, default=0,
                        help="--loss-neighb-dist-weight is the weight of the loss computing the squared Euclidean distance of every pair of adjacent vertices in a submesh (normalized so that the sum of the weights of all the parts of the loss sums up to 1). If --predict-uv-or-xyz=='uv', the distance uses the uv pixel coordinates of the vertices. If --predict-uv-or-xyz=='xyz', the distance uses the xyz coordinates of the vertices. If --predict-uv-or-xyz=='uvy', the distance uses the xyz previously obtained from the uv and y, using the camera parameters. (default: '0') I keep --loss-neighb-dist-weight just for compatibility purposes. We should use --loss-w-geo instead. See notes in --loss-w-geo")
    parser.add_argument('--loss-w-geo', type=int, default=0,
                        help="If --loss-w-geo is the same as --loss-neighb-dist-weight. The maximum of the two will be used, since the default for both of them is 0. I keep --loss-neighb-dist-weight just for compatibility purposes.")
    parser.add_argument('--loss-w-horizConsecEdges', type=int, default=0,
                        help="If --loss-w-horizConsecEdges is the weight of the loss that tries to minimize the angle between horizontally adjacent edges of the submesh (dafault: 0)")
    parser.add_argument('--loss-w-verConsecEdges', type=int, default=0,
                        help="If --loss-w-verConsecEdges is the weight of the loss that tries to minimize the angle between vertically adjacent edges of the submesh (dafault: 0)")
    parser.add_argument('--loss-w-chamfer-GT-pred', type=int, default=0,
                        help="--loss-w-chamfer-GT-pred is the weight within the loss for directional Chamfer distance that sums the distance of each uv pixel occupied by the towel in the RGB image, to the set of predicted uv vertices. (default: 0)")
    parser.add_argument('--loss-w-chamfer-pred-GT', type=int, default=0,
                        help="--loss-w-chamfer-pred-GT is the weight within the loss for directional Chamfer distance that sums the distance of predicted uv vertex to the set of uv pixels occupied by the towel in the RGB image. (default: 0)")
    parser.add_argument('--loss-w-chamfer-GTcontour-pred', type=int, default=0,
                        help="--loss-w-chamfer-GTcontour-pred is the weight within the loss for directional Chamfer distance that sums the distance of each uv pixel of the contour of the region occupied by the towel in the RGB image, to the set of predicted uv vertices. (default: 0)")
    parser.add_argument('--loss-w-chamfer-pred-GTcontour', type=int, default=0,
                        help="--loss-w-chamfer-pred-GTcontour is the weight within the loss for directional Chamfer distance that sums the distance of predicted uv vertex to the contour of the set of uv pixels occupied by the towel in the RGB image. (default: 0)")
    parser.add_argument('--n-outputs', type=int, default=2,
                        help="--n-outputs == 1 sums the two Directional Hausdorff distances and losses, without having control over the weights of each direction. --n-outputs == 2 allows to control independently the two Directional Hausdorff distances and losses. (default: 2)")
    parser.add_argument('--loss-w-normals', type=int, default=0,
                        help="--loss-w-normals is the weight within the loss for the normal vectors in the 3D reconstruction. (default: 0)")

    # Old loss weights, used to weight the tensors from which the losses are computed
    # Deprecated. Use New loss weights above instead
    parser.add_argument('--w-coord', type=int, default=1, 
                        help="--w-coord is the weight of the GT coordinates in the loss to be compared to the other weights in the loss (see normalize_weights()). Set --w-coord=0 to train an unsupervised model (default: 1). If args.predict_uv_or_xyz=='uv' or 'xyz', this is where you must set the weight for the coordinates. If --unsupervised==1, --w-coord will be automatically set to 0.")
    parser.add_argument('--w-uv', type=int, default=0, 
                        help="You must only use this if args.predict_uv_or_xyz=='uvy'. Otherwise, use --w-coord. --w-uv is the weight of the GT uv coordinates in the loss to be compared to the other weights in the loss (see normalize_weights()). Set --w-coord=0 to train an unsupervised model (default: 0)")
    parser.add_argument('--w-xyz', type=int, default=0, 
                        help="You must only use this if args.predict_uv_or_xyz=='uvy'. Otherwise, use --w-coord. --w-xyz is the weight of the GT xyz coordinates in the loss to be compared to the other weights in the loss (see normalize_weights()). Set --w-coord=0 to train an unsupervised model (default: 0)")
    parser.add_argument('--neighb-dist-weight', type=int, default=0,
                        help="If --neighb-dist-weight!=0, we add to the labels and predicted outputs used in the loss the squared Euclidean distance of every pair of adjacent vertices in a submesh multiplied by --neighb-dist-weight (normalized so that the sum of the weights of all the parts of the loss sums up to 1). If --predict-uv-or-xyz=='uv', the distance uses the uv pixel coordinates of the vertices. If --predict-uv-or-xyz=='xyz', the distance uses the xyz coordinates of the vertices. If --predict-uv-or-xyz=='uvy', the distance uses the xyz previously obtained from the uv and y, using the camera parameters. (default: '0')")
    parser.add_argument('--w-geo', type=int, default=0,
                        help="If --w-geo is the same as --neighb-dist-weight. The maximum of the two will be used, since the default for both of them is 0. I keep --neighb-dist-weight just for compatibility purposes.")
    parser.add_argument('--w-chamfer-GT-pred', type=int, default=0,
                        help="--w-chamfer-GT-pred is the weight within the loss for directional Chamfer distance that sums the distance of each uv pixel occupied by the towel in the RGB image, to the set of predicted uv vertices. (default: 0)")
    parser.add_argument('--w-chamfer-pred-GT', type=int, default=0,
                        help="--w-chamfer-pred-GT is the weight within the loss for directional Chamfer distance that sums the distance of predicted uv vertex to the set of uv pixels occupied by the towel in the RGB image. (default: 0)")
    parser.add_argument('--w-chamfer-GTcontour-pred', type=int, default=0,
                        help="--w-chamfer-GTcontour-pred is the weight within the loss for directional Chamfer distance that sums the distance of each uv pixel of the contour of the region occupied by the towel in the RGB image, to the set of predicted uv vertices. (default: 0)")
    parser.add_argument('--w-chamfer-pred-GTcontour', type=int, default=0,
                        help="--w-chamfer-pred-GTcontour is the weight within the loss for directional Chamfer distance that sums the distance of predicted uv vertex to the contour of the set of uv pixels occupied by the towel in the RGB image. (default: 0)")
    parser.add_argument('--w-normals', type=int, default=0,
                        help="--w-normals is the weight within the loss for the normal vectors in the 3D reconstruction. (default: 0)")

    # Visualizing predictions
    parser.add_argument('--batch-size-to-show', type=int, default=12, 
                        help='Number of elements for which to visualize prediction (default: == 12). In visualize_predictions.py, dataloaders work with args.batch_size_to_show, and args.batch_size is only used for finding the directory where the model is stored.')
    parser.add_argument('--dataset4predictions', type=int, default=0, 
                        help="The dataset's number used for predictions. If 0 (default), it will coincide with --dataset-number, which is the dataset used for training the model. CAVEAT: Regardless of the choice of dataset, it will still be split into train/val/test and evaluated only on the chosen of the 3 subsets.")
    parser.add_argument('--sequence4predictions', type=str, default='', 
                        help="The dataset's sequence name used for predictions. If '' (default), it will coincide with --sequence-name, which is the dataset used for training the model. CAVEAT: Regardless of the choice of dataset, it will still be split into train/val/test and evaluated only on the chosen of the 3 subsets.")
    parser.add_argument('--degrees-of-each-rotation', type=int, default=20,
                        help='Degrees of rotation of each step in the GIF creation. (default: 20')
    parser.add_argument('--directory-prepend', type=str, default='01_Bad_normalization_Barycenter_radius1_squareBdingBox',
                        help='What to prepend to the name of the directory which contain the model. (default: "01_Bad_normalization_Barycenter_radius1_squareBdingBox"')
    parser.add_argument('--swap-axes', type=int, default=1, 
                    help='swap_axes==1 to perform an axis swap so that the camera coordinates plot look like the world coordinates. (default: 1)')
    parser.add_argument('--train-or-val', type=int, default=1, 
                    help='train_or_val==1 visualizes the predictions on a validation batch. (default: 1) train_or_val==0 visualizes the predictions on a training batch. train_or_val==2 visualizes the predictions on a test batch.')
    # Used for instance in order_vertices_from_Blender.py
    parser.add_argument('--plot-or-GIF', type=int, default=1, 
                        help='0: Save plot. 1: Save GIF. 2: save a text file with the reordered vertex dataset. 3: save a text file with the reordered face dataset (default: 1)')
    parser.add_argument('--save-losses', type=int, default=0, 
                        help='0: do not save any losses. 1: save all the types of losses evaluated on training, validation and test sets and plot visualizations. 2: save all the types of losses evaluated on training, validation and test sets but do not plot visualizations. (default: 0)')
    parser.add_argument('--weights2visualize', type=int, default=-1, 
                        help="If -1, we will only visualize the model chosen as best (which at the moment, is chosen as the one with the best xyz or uv validation loss). If 0, we will visualize all models specified with the rest of the arguments in the parser. If 1, then we will only visualize the models with the weights appearing in the list ['0.00', '0.10', '0.40', '1.00'] If 2, then we will only visualize the models with the weights appearing in the list ['0.00', '0.80']. (default: 0)")
    parser.add_argument('--save-png-path', type=str, default=None, 
                        help="If not left default (None), this will be either the directory or the full name of the place to save some prediction plots.")
    parser.add_argument('--print-prediction', type=int, default=0, 
                        help="If 1, it prints the u (or x) of the first element in each batch during training (default: 0)")
    parser.add_argument('--transparency', type=float, default=0.05, 
                        help="Transparency for the faces. 0 means transparent, 1 means opaque. Between 0 and 1, translucent. (default: 0.05) For 2x3 meshes, 0.2 is a good value. for 8x12 meshes, 0.05 is a good value.")
    parser.add_argument('--annotate', type=int, default=0,
                        help='if --annotate==1, the uv predictio is taged with their vertex index, so check the order of the prediction. (default: 0)')
    parser.add_argument('--show-vertices', type=int, default=1, 
                        help="If 1, it plots GT and prediction on top of RGB images. If 0, it only plots the RGB images. (default: 1)")
    parser.add_argument('--plot-3D-pred', type=int, default=1, 
                        help="If 1, plot 3D GT. If 0, do not. (default: 1)")
    parser.add_argument('--plot-3D-GT', type=int, default=1, 
                        help="If 1, plot 3D GT. If 0, do not. (default: 1)")
    parser.add_argument('--ms-per-frame', type=int, default=25, 
                        help="milliseconds per frame to visualize videos. the value of 0 makes each frame to stay until a key is pressed (default: 25).")
    parser.add_argument('--test-uv-unnorm-batch', type=int, default=0, help="If set to 1, then visualize_predictions.py will unnormalize uv via unnormalize_uv_01_tensor_batch() rather than via unnormalize_uv_01(). This was only done to test the function unnormalize_uv_01_tensor_batch() and it does work.")
    parser.add_argument('--grid', type=int, default=1,
                       help="In visualize_predictions, grid==1 will plot predictions of a batch in a grid. grid==0 will plot them in separate figures.")
    parser.add_argument('--show-transformed-RGB', type=int, default=0,
                        help='1 shows prediction on transformed RGB. 0 (default) only shows it on original RGB.')
    parser.add_argument('--append-to-model-dir', type=str, default=None,
                        help="anything wanted to be appended to the directory name")
    parser.add_argument('--title-binary', type=int, default=1,
                        help="If set to 0, it will not print the title of the figures. If kept to 1 (default), it will.")
    parser.add_argument('--elt-within-batch', type=int, default=None,
                        help="Index of an element within a batch to be plotted when only one is being plotted.")
    parser.add_argument('--param-plots-4-paper', type=int, default=0,
                        help="If set to 1, some specific parameters for the plots to publish in the paper will be used for visualization.")
    parser.add_argument('--auto-save', type=int, default=0,
                        help="If set to 1, instead of showing visualizations, these will only be saved.")
    parser.add_argument('--triangular-faces', type=int, default=1,
                        help="If kept as 1, the plotted 3D faces will be triangular. If set to 0, they will be squared.")
    parser.add_argument('--batch-to-show', type=int, default=0,
                        help="The batch number batch-to-show will be visualized (with 0 indexing). (default: 0)")
    parser.add_argument('--line-width', type=float, default=None,
                        help="line width for plots (default: None, so the default for each plot will be used)")
    
        
    # Tensor Format
    parser.add_argument('--dtype', type=int, default=0, 
                        help="dtype=0 keeps all tensors and network weights as float; dtype=1 keeps them all as double (default: 0) Float format is strongly recommended. See Important_rmks_and_Conclusions/things_I_can_add_at_end_time_permitting.txt")
    
    args = parser.parse_args()
        
    if args.crop_centre_or_ROI==1:
        print("\n"*10 + "Before running, double check that there are no trivial crops in your dataset. E.g., RendersTowelWall11/Group.011/64_ROI.png" + "\n"*10)
    
    # Operations on the data
    if args.crop_centre_or_ROI==0: # centre crop
        args.transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                       ])    
    elif args.crop_centre_or_ROI in [1, 2, 3]: # Squared/rectangular box containing the towel or no crop
        args.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                       ]) 
    
    # Submesh
    if args.reordered_dataset==1:
        args.num_selected_vertices = args.submesh_num_vertices_vertical * args.submesh_num_vertices_horizontal
    elif args.reordered_dataset==0:
        args.submesh_num_vertices_vertical=None
        args.submesh_num_vertices_horizontal=None
            
    # Dataset parameters
    args.dataset_number = str(args.dataset_number)
    if args.sequence_name in ['kinect_tshirt', 'kinect_paper'] and args.dataset_number=='7':
        args.dataset_number = ''
    args.dataset4predictions = args.dataset_number if args.dataset4predictions==0 else str(args.dataset4predictions)
    if args.sequence4predictions=='': args.sequence4predictions = args.sequence_name
    if args.sequence_name == 'TowelWall':
        args.num_groups, args.num_animationFramesPerGroup = dataset_params(args.dataset_number)
        args.num_groups4predictions, args.num_animationFramesPerGroup4predictions = dataset_params(args.dataset4predictions)
        
    # Normalization
    # If all the normalization parameters to create the dataset classes are provided by the user,
    # then avoid computing those parameters
    if (args.normalize_datasetClass_D_min is not None) and (args.normalize_datasetClass_x_min is not None or args.predict_uv_or_xyz=='uvD'):
        args.all_normalizing_provided = 1
    else: 
        args.all_normalizing_provided = 0
    
    # 2D or 3D        
    # args.str_coord: To chose which coordinate-only validation loss to save 
    # at the end of training to compare hyperparameter performance
    # args.num_coord_per_vertex: To creating the last layer of the network, among other things:
    # >>> model_conv.fc = nn.Linear(num_ftrs, args.num_selected_vertices*args.num_coord_per_vertex) 
    if args.predict_uv_or_xyz in ['xyz', 'uvy', 'uvD'] and args.uv_loss_only==0:
        args.num_coord_per_vertex=3
        args.str_coord = 'xyz'
    else:
        args.num_coord_per_vertex=2
        args.str_coord = 'uv'    
    
    
    # Visualization
    args.swap_axes = args.camera_coordinates # Swap axes only in camera coordinates
#     if args.batch_size_to_show>args.batch_size:
#         args.batch_size_to_show = args.batch_size
    if args.lengths_proportion_train_4visuals is None:
        args.lengths_proportion_train_4visuals = args.lengths_proportion_train

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    ###
    ### Torch device (CPU vs GPU)
    ###
    args.device = torch.device("cuda:" + str(args.gpu_device) if torch.cuda.is_available() else "cpu")
    print('\nCuda device =', args.device)
    
    if args.sequence_name == 'TowelWall':
        # Camera parameters
        variables = load_camera_params(args.sequence_name, args.dataset_number)
        args.RT_matrix = variables['RT_matrix']
        args.RT_extended = variables['RT_extended']
        args.Camera_proj_matrix = variables['Camera_proj_matrix']
        args.Camera_proj_matrix_tensor = torch.from_numpy(args.Camera_proj_matrix).to(args.device)
        if args.dtype==0: args.Camera_proj_matrix_tensor=args.Camera_proj_matrix_tensor.float()

        # Inverse of the first 3 columns of Camera_proj_matrix (3x3 matrix)
        args.Camera_proj_matrix_inv = inv(args.Camera_proj_matrix[:,:-1]) 
        args.Camera_proj_matrix_inv_tensor = torch.from_numpy(args.Camera_proj_matrix_inv).to(args.device)
        if args.dtype==0: args.Camera_proj_matrix_inv_tensor=args.Camera_proj_matrix_inv_tensor.float()
        if args.dtype==0: 
            args.RT_matrix, args.RT_extended, args.Camera_proj_matrix = args.RT_matrix.astype(np.float32), args.RT_extended.astype(np.float32), args.Camera_proj_matrix.astype(np.float32)

    # Training
    args.invert_distance_direction=0 # Make sure every towel pixel is close to a uv prediction, rather than the other way around
    if (args.predict_uv_or_xyz!='uvy' or args.neighb_dist_weight!=0) and args.uv_loss_only==1:
        print("\n"*10 + "args.uv_loss_only==1 only makes sense if args.predict_uv_or_xyz=='uvy' and args.neighb_dist_weight==0" + "\n"*10)
    if args.uv_loss_only==1:
        args.neighb_dist_weight = 0
    args.loss_w_geo = max(args.loss_w_geo, args.loss_neighb_dist_weight)
    args.loss_neighb_dist_weight = args.loss_w_geo
    # Loss criteria
    args.loss_mse = nn.MSELoss()
    args.criterion_Hauss = diff_Hausdorff_loss.AveragedHausdorffLoss()
    
    # Old loss Weights
    if args.unsupervised==1: args.w_coord=0
    if args.predict_uv_or_xyz=='uv': args.w_uv=args.w_coord
    elif args.predict_uv_or_xyz=='xyz': args.w_xyz=args.w_coord
    elif args.predict_uv_or_xyz=='uvy': args.w_coord=0
    
    args.w_geo = max(args.w_geo, args.neighb_dist_weight)
    args.neighb_dist_weight = args.w_geo
    
    # Permutation for the output layer after performing a first unsupervised uv prediction round
    if args.permutation_dir is not None:
        if args.directory_prepend!='01_Bad_normalization_Barycenter_radius1_squareBdingBox': 
            args.permutation_dir = os.path.join(args.directory_prepend, args.permutation_dir) # used in visualize_predictions.py only
        args.permutation_4_output_layer = load_permutation(args.permutation_dir)

    return args
    # For more on this kind of parser, see 
    # https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/mnist/main.py

def load_permutation(model_directory):
    with open (os.path.join(model_directory, 'permutation_output_layer'), 'rb') as fp:
        permutation_4_output_layer = pickle.load(fp)
    return permutation_4_output_layer

def dataset_params(dataset_number):  
    """ Get the number of groups and animation frames per group,
    given a datset of the form 
    'RendersTowelWall' + dataset_number. """
    if dataset_number in ['2','3','7','11']:
        num_groups = 40
        num_animationFramesPerGroup = 99
    elif dataset_number=='4':
        num_groups = 4
        num_animationFramesPerGroup = 12
    elif dataset_number in ['5', '6', '8']:
        num_groups = 5
        num_animationFramesPerGroup = 4
    elif dataset_number in ['10', '12']:
        num_groups = 40
        num_animationFramesPerGroup = 10
    elif dataset_number in ['13', '14']:
        num_groups = 1
        num_animationFramesPerGroup = 1
    elif dataset_number in ['15']:
        num_groups = 2
        num_animationFramesPerGroup = 1
    elif dataset_number in ['16']:
        num_groups = 2
        num_animationFramesPerGroup = 5
    return num_groups, num_animationFramesPerGroup
    
def find_min_max_xyz_training(dataloaders):
#     print('i_batch, x_min, y_min, z_min, x_max, y_max, z_max')
    for i_batch, sample_batched in enumerate(dataloaders['train']):
        labels = sample_batched['xyz'] # Shape batch_size x num_selected_vertices x 3
        if i_batch == 0:
            x_min = torch.min(labels[:,:,0]).item()
            y_min = torch.min(labels[:,:,1]).item()
            z_min = torch.min(labels[:,:,2]).item()
            x_max = torch.max(labels[:,:,0]).item()
            y_max = torch.max(labels[:,:,1]).item()
            z_max = torch.max(labels[:,:,2]).item()
        else: 
            x_min = min(x_min, torch.min(labels[:,:,0]).item())
            y_min = min(y_min, torch.min(labels[:,:,1]).item())
            z_min = min(z_min, torch.min(labels[:,:,2]).item())
            x_max = max(x_max, torch.max(labels[:,:,0]).item())
            y_max = max(y_max, torch.max(labels[:,:,1]).item())
            z_max = max(z_max, torch.max(labels[:,:,2]).item())
#         print("%d     %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" % (i_batch, x_min, y_min, z_min, x_max, y_max, z_max))
    return x_min, y_min, z_min, x_max, y_max, z_max

def reject_outliers(data, m = 3.5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s =  0.6745*d/mdev if mdev else 0
    return data[s<m]

def find_min_max_xyz_training_wo_outliers(dataloaders, boxplot_on = 0, sequence_name = 'TowelWall', f_u=600.172, f_v=600.172, args=None):
    """ This assumes that nothing uv is normalized if and only if args.uv_normalization==1.
    D and xyz are not normalized. """
    x_min, y_min, z_min, x_max, y_max, z_max = [], [], [], [], [], []
    for i_batch, sample_batched in enumerate(dataloaders['train']):
        if sequence_name=='TowelWall':
            labels = sample_batched['xyz'] # Shape batch_size x num_selected_vertices x 3
        else:
#             labels = normalized_uvD_to_unnormalized_xyz_batch(sample_batched['uv'], sample_batched['D'], args, normalize_D_min=normalize_D_min, normalize_D_max=normalize_D_max, f_u=f_u, f_v=f_v)
            if args.uv_normalization==1:
                uv_unnormalized = unnormalize_uv_01_tensor_batch(
                    sample_batched['uv'], sequence_name=args.sequence_name)
            else: 
                uv_unnormalized = sample_batched['uv']
            labels = uvD_to_xyz_batch(uv_unnormalized, sample_batched['D'], f_u=f_u, f_v=f_v)
        batch_size=labels.shape[0]
        for i in range(batch_size):
            x_min.append(torch.min(labels[i,:,0]).item()) # append the minimum X in one observation
            y_min.append(torch.min(labels[i,:,1]).item()) 
            z_min.append(torch.min(labels[i,:,2]).item()) 
            x_max.append(torch.max(labels[i,:,0]).item()) # append the maximum X in one observation
            y_max.append(torch.max(labels[i,:,1]).item()) 
            z_max.append(torch.max(labels[i,:,2]).item()) 
    if boxplot_on==1:
        import matplotlib.pyplot as plt
        plt.boxplot([x_min, y_min, z_min, x_max, y_max, z_max])
        plt.xticks(list(range(1,7)), ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max'])
        plt.show()
    x_min, y_min, z_min =np.array(x_min), np.array(y_min), np.array(z_min)
    x_max, y_max, z_max = np.array(x_max), np.array(y_max), np.array(z_max)
    x_min_wo_outliers, y_min_wo_outliers, z_min_wo_outliers = reject_outliers(x_min), reject_outliers(y_min), reject_outliers(z_min)
    x_max_wo_outliers, y_max_wo_outliers, z_max_wo_outliers = reject_outliers(x_max), reject_outliers(y_max), reject_outliers(z_max)
    x_min_wo_outliers = np.min(x_min_wo_outliers)
    y_min_wo_outliers = np.min(y_min_wo_outliers)
    z_min_wo_outliers = np.min(z_min_wo_outliers)
    x_max_wo_outliers = np.max(x_max_wo_outliers)
    y_max_wo_outliers = np.max(y_max_wo_outliers)
    z_max_wo_outliers = np.max(z_max_wo_outliers)

    return x_min_wo_outliers, y_min_wo_outliers, z_min_wo_outliers, x_max_wo_outliers, y_max_wo_outliers, z_max_wo_outliers
    
def find_min_max_D_training(dataloaders, boxplot_on=0):
    for i_batch, sample_batched in enumerate(dataloaders['train']):
        labels = sample_batched['D'] # Shape batch_size x num_selected_vertices x 1
        if i_batch == 0:
            D_min = torch.min(labels[:,:,0]).item()
            D_max = torch.max(labels[:,:,0]).item()
        else: 
            D_min = min(D_min, torch.min(labels[:,:,0]).item())
            D_max = max(D_max, torch.max(labels[:,:,0]).item())
    return D_min, D_max

def find_min_max_D_training_wo_outliers(dataloaders, boxplot_on = 0):
    D_min, D_max = [], []
    for i_batch, sample_batched in enumerate(dataloaders['train']):
        labels = sample_batched['D'] # Shape batch_size x num_selected_vertices x 1
        batch_size=labels.shape[0]
        for i in range(batch_size): 
            D_min.append(torch.min(labels[i,:,0]).item()) # append the minimum D in one observation
            D_max.append(torch.max(labels[i,:,0]).item()) # append the maximum D in one observation
    if boxplot_on==1:
        import matplotlib.pyplot as plt
        plt.boxplot([D_min, D_max])
        plt.xticks(list(range(1,3)), ['D_min', 'D_max'])
        plt.show()
    D_min, D_max = np.array(D_min), np.array(D_max)
    D_min_wo_outliers, D_max_wo_outliers = reject_outliers(D_min), reject_outliers(D_max)
    D_min_wo_outliers, D_max_wo_outliers = np.min(D_min_wo_outliers), np.max(D_max_wo_outliers)
    return D_min_wo_outliers, D_max_wo_outliers

###
### Convert: World coordinates to Camera coordinates
###
# RT_matrix = np.genfromtxt('RendersTowelWall2/camera_params.txt', delimiter=' ', skip_header=11)
# # print('RT matrix:\n', RT_matrix)
# # print()

# # RT matrix for homogeneous coordinates
# zeros_and_1 = np.zeros(4)
# zeros_and_1[-1] = 1
# zeros_and_1 = np.reshape(zeros_and_1, (1,4))
# RT_extended = np.concatenate((RT_matrix, zeros_and_1), axis=0)
# # print("RT extended with zeros and 1 below (for homogeneous coordinates):\n", RT_extended)
# # print()

def world_to_camera_coordinates(X_world, Y_world, Z_world,
                                RT_extended, dtype=1):
    '''
    Input: 
    - 1 or multiple points given by their X_world, Y_world, Z_world world coordinates
    - Extended RT matrix (I.e., RT matrix with zeros below the rotation and a 1 below the translation).
        This is used to perform operations on homogeneous coordinates.
    Output:
    - Camera coordinates of the point/points.
    '''
    ones_row = np.ones(X_world.size)
    ones_row = np.reshape(ones_row, (1,X_world.size))
    if dtype==0: ones_row = ones_row.astype(np.float32)
    X_world_row = np.reshape(X_world, (1, X_world.size))
    Y_world_row = np.reshape(Y_world, (1, Y_world.size))
    Z_world_row = np.reshape(Z_world, (1, Z_world.size))
    homogeneous_world = np.vstack((X_world_row, Y_world_row, Z_world_row, ones_row))
    homogeneous_camera = np.matmul(RT_extended, homogeneous_world)
    # Notice that since our RT_extended last row is 0 0 0 1, homogeneous_camera[3, :] == 1
    X_camera = np.true_divide(homogeneous_camera[0, :],homogeneous_camera[3, :])
    Y_camera = np.true_divide(homogeneous_camera[1, :],homogeneous_camera[3, :])
    Z_camera = np.true_divide(homogeneous_camera[2, :],homogeneous_camera[3, :])   
    return (X_camera, Y_camera, Z_camera)
# See visualize_mesh.ipynb for a test of this function and some plots

def world_to_camera_coordinates_normals(nX_world, nY_world, nZ_world,
                                RT_matrix):
    '''
    Input: 
    - 1 or multiple normal vectors given by their nX_world, nY_world, nZ_world world coordinates
    - RT matrix (from which we will only need the rotation).
    Output:
    - Camera coordinates of the normal(s). This consists of rotating the normal(s).
    '''
    nX_world_row = np.reshape(nX_world, (1, nX_world.size))
    nY_world_row = np.reshape(nY_world, (1, nY_world.size))
    nZ_world_row = np.reshape(nZ_world, (1, nZ_world.size))
    normals_world = np.vstack((nX_world_row, nY_world_row, nZ_world_row))
    rotation_matrix = RT_matrix[:, :-1]
    normals_camera = np.matmul(rotation_matrix, normals_world)
    nX_camera = normals_camera[0, :]
    nY_camera = normals_camera[1, :]
    nZ_camera = normals_camera[2, :]
    return (nX_camera, nY_camera, nZ_camera)
# See visualize_mesh.ipynb for a test of this function and some plots

def normals_in_camera_coordiantes(normal_coordinates, RT_matrix):
    """ To be used in the dataset class data_loading.vertices_Dataset() as follows:
    >>> if self.args.camera_coordinates==1:
    >>>    normal_coordinates = normals_in_camera_coordiantes(normal_coordinates, self.args.RT_matrix)"""
    (nX_camera, nY_camera, nZ_camera) = world_to_camera_coordinates_normals(
        normal_coordinates[:,0], normal_coordinates[:,1], normal_coordinates[:,2], RT_matrix)
    nX_camera_col = np.reshape(nX_camera, (nX_camera.size, 1))
    nY_camera_col = np.reshape(nY_camera, (nY_camera.size, 1))
    nZ_camera_col = np.reshape(nZ_camera, (nZ_camera.size, 1))
    return np.hstack((nX_camera_col, nY_camera_col, nZ_camera_col))

def world_to_camera(Vertex_coordinates, RT_extended, dtype=1):
    """ To be used in the dataset class data_loading.vertices_Dataset() as follows:
    >>> if self.args.camera_coordinates==1:
    >>>    Vertex_coordinates = world_to_camera(Vertex_coordinates, self.args.RT_extended)"""
    # Convert vertex and normal vector coordinates from world coordinates to camera coordinates
    (X_camera, Y_camera, Z_camera) = world_to_camera_coordinates(
        Vertex_coordinates[:,0], Vertex_coordinates[:,1], Vertex_coordinates[:,2], RT_extended, dtype)
    X_camera_col = np.reshape(X_camera, (X_camera.size, 1))
    Y_camera_col = np.reshape(Y_camera, (Y_camera.size, 1))
    Z_camera_col = np.reshape(Z_camera, (Z_camera.size, 1))
    return np.hstack((X_camera_col, Y_camera_col, Z_camera_col))

def world_to_pixel_coordinates(X_world, Y_world, Z_world, Camera_proj_matrix, 
                               u_original=None, v_original=None, verbose=1, rounding=0):
    """Convert the world coordinates X_world, Y_world, Z_world of a point into its u, v pixel coordinates.
    Optional args: u_original and v_original are the ground truth pixel coordinates, 
    which are there just for comparison.
    
    If rounding==1, the uv projection is rounded to the nearest integer."""
    homogeneous_world = [X_world, Y_world, Z_world, 1]
    homogeneous_world = np.reshape(homogeneous_world, (4,1)).astype(float)

    # At the beginning, I thought I had to do
    # Intrinsic_matrix @ Camera_proj_matrix @ RT_extended @ homogeneous_world,
    # because I expected the Camera projection matrix of camera_parameters.txt 
    # to be the one that sends camera coordinates to film coordinates, 
    # but I checked in visualize_coordinate_conversion.ipynb
    # that the Camera projection matrix of camera_parameters.txt is the one that sends 
    # world coordinates to pixel coordinates.what.
    # Therefore, I simply need to use the next line:
    homogeneous_pixel = np.matmul(Camera_proj_matrix, homogeneous_world)
    homogeneous_factor = homogeneous_pixel.item(2)
    u_pixel_projected = homogeneous_pixel.item(0)/homogeneous_factor
    v_pixel_projected = homogeneous_pixel.item(1)/homogeneous_factor
        
    if verbose==1:
        print("Convert: World coordinates --> pixel coordinates")
        print("------------------------------------------------")
        print("Original X, Y, Z world coordinates from Blender:\n", X_world, Y_world, Z_world)
        print("Projected u, v pixel coordinates from world coordinates using the Camera Projection Matrix:\n",
              u_pixel_projected, v_pixel_projected)
        # Compare to "ground truth" pixel coordinates given by Blender
        if u_original is not None:
            print("Grond Truth u, v pixel coordinates (from Blender):\n", u_original, v_original, "\n")
    
    if rounding==0: return u_pixel_projected, v_pixel_projected, homogeneous_factor
    else: return round(u_pixel_projected), round(v_pixel_projected), homogeneous_factor

def world_to_pixel_coordinates_whole_set(X_world, Y_world, Z_world, Camera_proj_matrix, rounding=1):
    """Convert the world coordinates X_world, Y_world, Z_world of a set of points into their u, v pixel coordinates.
    
    If rounding==1, the uv projection is rounded to the nearest integer.
    
    Input and output coordinates are numpy arrays."""    
    dtype = int if rounding==1 else float
    u_proj = np.ones((len(X_world),), dtype=dtype)
    v_proj = np.ones((len(X_world),), dtype=dtype)
    for vertex_number in range(len(X_world)):
        u_proj[vertex_number], v_proj[vertex_number], homogeneous_factor = world_to_pixel_coordinates(
            X_world[vertex_number], Y_world[vertex_number], Z_world[vertex_number],
            Camera_proj_matrix = Camera_proj_matrix, verbose=0, rounding=rounding)
    return u_proj, v_proj

def world_to_pixel_coordinates_whole_set_tensor(X_world, Y_world, Z_world, Camera_proj_matrix, rounding=1, args=None):
    """Convert the world coordinates X_world, Y_world, Z_world of a set of points into their u, v pixel coordinates.
    
    If rounding==1, the uv projection is rounded to the nearest integer.
    
    Input and output coordinates are tensors."""    
    u_proj = torch.ones([len(X_world),]).to(args.device)
    v_proj = torch.ones([len(X_world),]).to(args.device)
    if args.dtype==1:
        u_proj=u_proj.double()
        v_proj=v_proj.double()
    for vertex_number in range(len(X_world)):
        u_proj[vertex_number], v_proj[vertex_number], homogeneous_factor = world_to_pixel_coordinates(
            X_world[vertex_number], Y_world[vertex_number], Z_world[vertex_number],
            Camera_proj_matrix = Camera_proj_matrix, verbose=0, rounding=rounding)
    return u_proj, v_proj

def pixel_to_world_coordinates_knowing_Y(u, v, C,
                               Y_world_GT, X_world_GT=None, Z_world_GT=None,
                               verbose=1, C_inv=None):
    """ 
    Change of coordinates of a point P.
    We know its pixel coordinates u, v and its Y in world coordinates Y_world.
    We obtain the world coordinates of P: X_world, Y_world, Z_world.
    
    Arguments: 
    - u, v: pixel coordinates of P
    - C = Camera_proj_matrix (3x4 matrix)
    - Y_world_GT: known Y_world of P
    Optional arguments:
    - X_world_GT, Z_world_GT are the ground truth X, Z world coordinates, 
    which are there just for comparison.
    - C_inv = inv(C[:,:-1]), i.e., inverse of first 3 columns of C (3x3 matrix)

    Note on accuracy:
    In order to obtain an X, Y, Z recovery as accurate as possible, the input u, v should be the 
    ones obtained by projecting X, Y, Z to pixel coordinates.
    If instead, we use the u, v from Blender, the u, v are the projected ones, but rounded to the nearest integer.      

    General idea: 
    We know that, with NON-homogeneous coordiantes
    l * (u, v, 1) = C @ (X, Y, Z, 1),      (*)
    where l =  C[-1, :] @ (X, Y, Z, 1).
    In our case, l = C[2, 1] * Y + C[2, 3].
    Hence, we can rewrite equation (*) as follows:
    ( l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3])=C[:,:-1] @ (X, Y, Z).
    Hence, we can compute the world coordinates by inverting the camera projection matrix (without the last row):
    (X, Y, Z) = inverse_of_C[:,:-1] @ ( l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3]).
    Note that the third coordinate of the first term is l - C[2, 3] = Y.
    
    Input, output and operations within the function are numpy-based.
    """
    l = C[2, 1] * Y_world_GT + C[2, 3]
    if C_inv is None:
        world_coordinates = inv(C[:,:-1]) @ np.reshape(np.array([[ l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3]]], dtype=float), (3))
    else:
        world_coordinates = C_inv @ np.reshape(np.array([[ l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3]]], dtype=float), (3))
    X_world_projected = world_coordinates[0]
    Y_world_projected = world_coordinates[1]
    Z_world_projected = world_coordinates[2]

    if verbose==1:
        print("Convert: Pixel coordinates --> World coordinates")
        print("------------------------------------------------")
        print("It is better if the input u, v coordinates do not come directly from Blender (int) but rather, if they are projected from the x,y,z world coordinates to obtain a float.")
        print("Since Blender rounds them to the nearest integer, the exact X, Y, Z recovery is impossible in general with Blender's input.")
        print("Input u, v pixel coordinates:\n", u, v)
        print("Projected X, Y, Z world coordinates from pixels using the Camera projection matrix and Y")
        print(X_world_projected, Y_world_projected, Z_world_projected)    
        # Compare to world coordinates given by Blender
        if X_world_GT is not None:
            print("Ground Truth X, Y, Z world coordinates (from Blender):\n", X_world_GT, Y_world_GT, Z_world_GT)
        print()
        
    return X_world_projected, Y_world_projected, Z_world_projected

def pixel_to_world_of_cloud_knowing_Y_tensor(uv_coord, C, C_inv_tensor, Y_world_GT, xyz_world_GT=None,
                               verbose=0):
    """ 
    Change of coordinates of a mesh of points. Call P any such point.
    We know the pixel coordinates u, v and the Y world coordinate Y_world of each P.
    We obtain the world coordinates of all those points: X_world, Y_world, Z_world.
    
    Arguments: 
    - uv_coord: tensor of shape Nx2 of pixel coordinates of each P
    - C = Camera_proj_matrix (3x4 matrix)
    - Y_world_GT: tensor of shape Nx1 of known Y_world of each P
    Optional arguments:
    - xyz_world_GT: tensor of shape Nx3 of the ground truth X, Y, Z world coordinates, 
    which are there just for comparison.
    - C_inv_tensor = torch.from_numpy(inv(C[:,:-1])).to(args.device).double(), 
        i.e., C_inv_tensor = inverse of first 3 columns of C (3x3 matrix) as a tensor

    Note on accuracy:
    In order to obtain an X, Y, Z recovery as accurate as possible, the input u, v should be the 
    ones obtained by projecting X, Y, Z to pixel coordinates.
    If instead, we use the u, v from Blender, the u, v are the projected ones, but rounded to the nearest integer.      

    General idea on 1 point P: 
    We know that, with NON-homogeneous coordiantes
    l * (u, v, 1) = C @ (X, Y, Z, 1),      (*)
    where l =  C[-1, :] @ (X, Y, Z, 1).
    In our case, l = C[2, 1] * Y + C[2, 3].
    Hence, we can rewrite equation (*) as follows:
    ( l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3])=C[:,:-1] @ (X, Y, Z).
    Hence, we can compute the world coordinates by inverting the camera projection matrix (without the last row):
    (X, Y, Z) = inverse_of_C[:,:-1] @ ( l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3]).
    Note that the third coordinate of the first term is l - C[2, 3] = Y.
    
    General idea for the whole set:
    tensor of shape 3XN where each column represents the XYZ of 1 point =
    inverse_of_C[:,:-1] @ 
    tensor of shape 3XN where each column represents ( l*u - C[0, 3], l*v - C[1, 3], l - C[2, 3]) of 1 point, 
    where recall that l = C[2, 1] * Y + C[2, 3] and therefore depends on the point P.
    
    Input, output and operations within the function are tensor-based, 
    so that variables which require Grad will not be detached.
    
    Output: tensor of XYZ world coordinates of shape (N, 3)
    """
    n_points = uv_coord.shape[0]
    u = uv_coord[:,0].view(1, n_points)
    v = uv_coord[:,1].view(1, n_points)
    l = (C[2, 1] * Y_world_GT + C[2, 3]).view(1, n_points)
    u = u * l - C[0, 3]
    v = v * l - C[1, 3]
    last_row = l - C[2, 3]
    xyz = torch.t(torch.matmul(C_inv_tensor, torch.cat((u, v, last_row), 0)))
    
    if verbose==1:
        print("Convert: Pixel coordinates --> World coordinates")
        print("------------------------------------------------")
        print("It is better if the input u, v coordinates do not come directly from Blender (int) but rather, if they are projected from the x,y,z world coordinates to obtain a float.")
        print("Since Blender rounds them to the nearest integer, the exact X, Y, Z recovery is impossible in general with Blender's input.")
        print("Input u, v pixel coordinates:\n", uv_coord)
        print("Projected X, Y, Z world coordinates from pixels using the Camera projection matrix and Y\n", xyz)
        # Compare to world coordinates given by Blender
        if xyz_world_GT is not None:
            print("Ground Truth X, Y, Z world coordinates (from Blender):\n", xyz_world_GT)
        print()
        
    return xyz

def pixel_to_world_of_mesh_of_clouds_knowing_Y(uv_coord, C, C_inv_tensor, Y_world_GT, xyz_world_GT=None,
                               verbose=0):
    """ 
    Apply pixel_to_world_of_cloud_knowing_Y_tensor() to a mesh o N points.
    
    Change of coordinates of a mesh of points. Call P any such point.
    We know the pixel coordinates u, v and the Y world coordinate Y_world of each P.
    We obtain the world coordinates of all those points: X_world, Y_world, Z_world.
    
    Arguments: 
    - uv_coord: tensor of shape (N,2) of pixel coordinates of each P
    - C = Camera_proj_matrix (3x4 matrix)
    - Y_world_GT: tensor of shape (N,1) of known Y_world of each P
    Optional arguments:
    - xyz_world_GT: tensor of shape (N,3) of the ground truth X, Y, Z world coordinates, 
    which are there just for comparison.
    - C_inv_tensor = torch.from_numpy(inv(C[:,:-1])).to(args.device).double(), 
        i.e., C_inv_tensor = inverse of first 3 columns of C (3x3 matrix) as a tensor

    Note on accuracy:
    In order to obtain an X, Y, Z recovery as accurate as possible, the input u, v should be the 
    ones obtained by projecting X, Y, Z to pixel coordinates.
    If instead, we use the u, v from Blender, the u, v are the projected ones, but rounded to the nearest integer.      
    
    Output: tensor of XYZ world coordinates of shape (N,3)
    """
    n_points = uv_coord.shape[0]
    u = uv_coord[:,0].view(1, n_points)
    v = uv_coord[:,1].view(1, n_points)
    l = (C[2, 1] * Y_world_GT + C[2, 3]).view(1, n_points)
    u = u * l - C[0, 3]
    v = v * l - C[1, 3]
    last_row = l - C[2, 3]
    xyz = torch.t(torch.matmul(C_inv_tensor, torch.cat((u, v, last_row), 0)))
    
    if verbose==1:
        print("Convert: Pixel coordinates --> World coordinates")
        print("------------------------------------------------")
        print("It is better if the input u, v coordinates do not come directly from Blender (int) but rather, if they are projected from the x,y,z world coordinates to obtain a float.")
        print("Since Blender rounds them to the nearest integer, the exact X, Y, Z recovery is impossible in general with Blender's input.")
        print("Input u, v pixel coordinates:\n", uv_coord)
        print("Projected X, Y, Z world coordinates from pixels using the Camera projection matrix and Y\n", xyz)
        # Compare to world coordinates given by Blender
        if xyz_world_GT is not None:
            print("Ground Truth X, Y, Z world coordinates (from Blender):\n", xyz_world_GT)
        print()
        
    return xyz

def pixel_to_world_of_batch_of_clouds_knowing_Y(uv_coord, C, C_inv_tensor, Y_world_GT, args, xyz_world_GT=None,
                               verbose=0):
    """ 
    Apply pixel_to_world_of_mesh_of_clouds_knowing_Y() to a batch.
    
    The idea is to get the points of all batches together as if they were just the points of one mesh 
    and then reshape the result back to batch format.
    
    Change of coordinates of a batch of points. Call P any such point.
    We know the pixel coordinates u, v and the Y world coordinate Y_world of each P.
    We obtain the world coordinates of all those points: X_world, Y_world, Z_world.
    
    Arguments: 
    - uv_coord: tensor of shape (batch_size,N,2) of pixel coordinates of each P
    - C = Camera_proj_matrix (3x4 matrix)
    - Y_world_GT: tensor of shape (batch_size,N,1) of known Y_world of each P
    Optional arguments:
    - xyz_world_GT: tensor of shape (batch_size,N,3) of the ground truth X, Y, Z world coordinates, 
    which are there just for comparison.
    - C_inv_tensor = torch.from_numpy(inv(C[:,:-1])).to(args.device).double(), 
        i.e., C_inv_tensor = inverse of first 3 columns of C (3x3 matrix) as a tensor

    Note on accuracy:
    In order to obtain an X, Y, Z recovery as accurate as possible, the input u, v should be the 
    ones obtained by projecting X, Y, Z to pixel coordinates.
    If instead, we use the u, v from Blender, the u, v are the projected ones, but rounded to the nearest integer.      
    
    Output: tensor of XYZ world coordinates of shape (batch_size,N,3)
    """
    batch_size = uv_coord.shape[0]
    num_selected_vertices = uv_coord.shape[1] # n_points_per_batch
    uv_coord = uv_coord.view(-1, 2) # shape (batch_size * num_selected_vertices, 2)
    Y_world_GT = Y_world_GT.view(-1, 2) # shape (batch_size * num_selected_vertices, 1)
    xyz = pixel_to_world_of_mesh_of_clouds_knowing_Y(uv_coord, C, C_inv_tensor, Y_world_GT, xyz_world_GT, verbose)
    xyz_coord = torch.zeros([batch_size, num_selected_vertices, 3], dtype=uv_coord.dtype,
                                         requires_grad=False).to(args.device)
    if args.dtype==1: xyz_coord=xyz_coord.double()
    for i in range(batch_size):
        xyz_coord[i, :, :] = xyz[num_selected_vertices*i:num_selected_vertices*(i+1), :]
    return xyz_coord

if __name__=='__main__':
    a = torch.tensor([[[1,2], [3,4], [5,6]], [[7,8], [9, 10], [11, 12]], [[1,2], [3,4], [5,6]], [[7,8], [9, 10], [11, 12]]])
    print(a)
    print(a.shape)
    print(a.view(-1, 2))
    print(a.view(-1, 2).shape, '\n')

def pixel_to_world_coordinates(u, v, homogeneous_factor, Camera_proj_matrix,
                               Z_world_GT, X_world_GT=None, Y_world_GT=None,
                               verbose=1):
    """ to convert from pixel to world coordinates, use pixel_to_world_coordinates_knowing_Y() instead of this function;
        it is much better in all senses.
        
        pixel_to_world_coordinates():
        
        Convert the pixel coordinates u, v of a point into world coordinates X_world, Y_world, Z_world.
        Optional args: X_world_GT, Y_world_GT are the ground truth X, Y world coordinates, 
        which are there just for comparison.
        
        In order to obtain an X, Y, Z recovery as accurate as possible, the input u, v should be the 
        ones obtained by projecting X, Y, Z to pixel coordinates.
        If instead, we use the u, v from Blender, the u, v are the projected ones, but rounded to the nearest integer.      
        
        We assume we know Z_world.
        
        General idea:
        Extending Camera_proj_matrix_ext to 5x4 to force homogeneous_world.item(2) = Z, homogeneous_world.item(3) = 1
        Therefore, the solver for (X, Y, Z, 1) only needs to solve for X and Y. """

    # Start with homogeneous_pixel coordinates (u, v, 1)*homogeneous_factor
    homogeneous_pixel_ext = [u*homogeneous_factor, v*homogeneous_factor, 1*homogeneous_factor]
    # Extend homogeneous_pixel with an extra 1 and the known Z_world_GT at the end
    homogeneous_pixel_ext.append(1)
    homogeneous_pixel_ext.append(Z_world_GT)
    homogeneous_pixel_ext = np.reshape(homogeneous_pixel_ext, (5,1))

    # To impose homogeneous_world.item(2) = Z, homogeneous_world.item(3) = 1,
    # I extend Camera_proj_matrix with two rows at the bottom: 0 0 0 1 and 0 0 1 0
    Camera_proj_matrix_ext = np.vstack((Camera_proj_matrix, np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])))

    # Solve the system of linear equations Camera_proj_matrix_ext * homogeneous_world = homogeneous_pixel_ext
    homogeneous_world = np.linalg.lstsq(Camera_proj_matrix_ext, homogeneous_pixel_ext)[0]
    X_world_projected = homogeneous_world.item(0)/homogeneous_world.item(3)
    Y_world_projected = homogeneous_world.item(1)/homogeneous_world.item(3)
    Z_world_projected = homogeneous_world.item(2)/homogeneous_world.item(3)

    if verbose==1:
        print("Convert: Pixel coordinates --> World coordinates")
        print("------------------------------------------------")
        print("It is better if the input u, v coordinates do not come directly from Blender (int) but rather, if they are projected from the x,y,z world coordinates to obtain a float.")
        print("Since Blender rounds them to the nearest integer, the exact X, Y, Z recovery is impossible in general with Blender's input.")
        print("Input u, v pixel coordinates:\n", u, v)
        print("Projected X, Y, Z world coordinates from pixels using the Camera projection matrix and Z")
        print(X_world_projected, Y_world_projected, Z_world_projected)    
        # Compare to world coordinates given by Blender
        if X_world_GT is not None:
            print("Ground Truth X, Y, Z world coordinates (from Blender):\n", X_world_GT, Y_world_GT, Z_world_GT)
        print()
        
    return X_world_projected, Y_world_projected, Z_world_projected

def uvD_to_xyz(uvD, resW=223., resH=223., f_u=600.172, f_v=600.172):
    """ Input: uv pixel coordinates and depth D
    Output: xyz camera coordinates.
    Format: uv, D and xyz will be numpy arrays."""
    u_centre = resW / 2.0
    v_centre = resH / 2.0
    if isinstance(uvD, np.ndarray):
        xyz = np.zeros(uvD.shape, dtype=uvD.dtype)
        xyz[:, 0] = np.multiply(1/f_u * uvD[:,2], uvD[:,0]- u_centre)
        xyz[:, 1] = np.multiply(1/f_v * uvD[:,2], uvD[:,1]- v_centre)
        xyz[:, 2] = uvD[:,2]
    return xyz

def uvD_to_xyz_tensor(uv, D, resW=223., resH=223., f_u=600.172, f_v=600.172):
    """ Input: uv pixel coordinates and depth D
    Output: xyz camera coordinates
    Format: uv, D and xyz will be tensors.
    Caveat: D must hav shape (batch_size, 1), rather than (batch_size)."""
    u_centre = resW / 2.0
    v_centre = resH / 2.0
    batch_size = uv.shape[0]
    xyz = torch.zeros((batch_size, 3), dtype=uv.dtype, requires_grad=False, device=uv.device)
    xyz[:, 0] = torch.mul(1/f_u * D[:,0], uv[:,0]- u_centre)
    xyz[:, 1] = torch.mul(1/f_v * D[:,0], uv[:,1]- v_centre)
    xyz[:, 2] = D[:,0]
    return xyz

def uvD_to_xyz_batch(uv_batch, D_batch, resW=223., resH=223., f_u=600.172, f_v=600.172):
    """ uv_batch: tensor of shape (batch_size, num_selected_vertices, 2)
    D_batch: tensor of shape (batch_size, num_selected_vertices, 1)"""
    batch_size = uv_batch.shape[0]
    num_selected_vertices = uv_batch.shape[1]
    uv_batch_linear = uv_batch.view(-1, 2) # Vertically stack the data of all elements in the batch
    D_batch_linear = D_batch.view(-1, 1)
    xyz_batch_linear = uvD_to_xyz_tensor(uv_batch_linear, D_batch_linear, resW, resH, f_u, f_v)
    xyz_batch = xyz_batch_linear.view(batch_size, num_selected_vertices, 3)
    return xyz_batch

if __name__=='__main__':
    uv_batch=torch.zeros(12, 6, 2)
    uv_batch[0, :, 0] = torch.tensor([1, 2, 3, 4, 5, 6])
    D_batch=torch.ones(12, 6, 1)
    xyz = uvD_to_xyz_batch(uv_batch, D_batch, resW=223., resH=223., f_u=600.172, f_v=600.172)
    print(xyz[0,:,:])
    print(xyz[1,:,:])
    # A better example showing this function works can be found in visualize_mesh_DeepCloth2.py
    
    
# Load the vertices files disregarding the string '# ' at the beginning of the file
def get_variables_from_vertex_full_Dataframe(sequence_name, dataset_number, group_number,
                                             animation_frame, RT_extended, reordered=0, submesh_idx=None, verbose=1):
    filename = 'Renders' + sequence_name + dataset_number + '/Group.' + group_number 
    filename += '/vertices_' + animation_frame
    if reordered == 1:
        filename+= '_reordered'
    filename+='.txt'
    f = open(filename, 'r')
    line1 = f.readline()
    df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())
    if verbose == 1:
        print(df_vertices_all_data.head())
        print()

    #
    # Select the submesh only
    #
    if submesh_idx is not None:
        df_vertices_all_data = df_vertices_all_data.ix[submesh_idx]

    # Occlusion mask
#     print('Recall:\n\'occluded = 1\' means the vertex is occluded\n\'occluded = 0\' means the vertex is visible\n')
    df_occlusion_mask = df_vertices_all_data['occluded']
    occlusion_mask_values = df_occlusion_mask.values

    # Pixel coordinates u, v
    df_u = df_vertices_all_data['u']
    df_v = df_vertices_all_data['v']
    u = df_u.values
    v = df_v.values

    # Pixel coordinates u, v of each visible vertex only
    df_u_visible = df_u[df_vertices_all_data['occluded'] == 0] # consider only the visible vertices
    df_v_visible = df_v[df_vertices_all_data['occluded'] == 0]
    u_visible = df_u_visible.values
    v_visible = df_v_visible.values

    # Pixel coordinates u, v of each occluded vertex only
    df_u_occluded = df_u[df_vertices_all_data['occluded'] == 1] # consider only the occluded vertices
    df_v_occluded = df_v[df_vertices_all_data['occluded'] == 1]
    u_occluded = df_u_occluded.values
    v_occluded = df_v_occluded.values

    # World coordinates
    X = df_vertices_all_data['x'].values
    Y = df_vertices_all_data['y'].values
    Z = df_vertices_all_data['z'].values
    if verbose==1:
        print('Number of vertices = ' + str(X.size))
    X_world = X
    Y_world = Y
    Z_world = Z
    (X_camera, Y_camera, Z_camera) = world_to_camera_coordinates(X_world, Y_world, Z_world, RT_extended)

    # Normal vectors to the surface at the vertices
    nX = df_vertices_all_data['nx'].values
    nY = df_vertices_all_data['ny'].values
    nZ = df_vertices_all_data['nz'].values
#     print('X_world_coordinates of the normal vector of the first 3 vertices:', nX[0:3])
    nX_world=nX
    nY_world=nY
    nZ_world=nZ
    
    variables = {'occlusion_mask_values':occlusion_mask_values, 'u':u, 'v':v, 'u_visible':u_visible, 'v_visible':v_visible,
                 'u_occluded':u_occluded, 'v_occluded':v_occluded, 'X_world':X_world, 'Y_world':Y_world, 'Z_world': Z_world, 'X_camera':X_camera, 'Y_camera':Y_camera, 'Z_camera':Z_camera, 'nX_world':nX_world, 'nY_world':nY_world, 'nZ_world':nZ_world}
    return variables

    # # Example of use
    # variables = get_variables_from_vertex_full_Dataframe(sequence_name, dataset_number, group_number,
    #                                              animation_frame, reordered)

    # occlusion_mask_values = variables['occlusion_mask_values']
    # u = variables['u']
    # v = variables['v']
    # u_visible = variables['u_visible']
    # v_visible = variables['v_visible']
    # X_world = variables['X_world']
    # Y_world = variables['Y_world']
    # Z_world = variables['Z_world']
    # X_camera = variables['X_camera']
    # Y_camera = variables['Y_camera']
    # Z_camera = variables['Z_camera']
    # nX_world = variables['nX_world']
    # nY_world = variables['nY_world']
    # nZ_world = variables['nZ_world']

def load_camera_params(sequence_name = 'TowelWall', dataset_number = '2', verbose=0):
    dataset_directory = 'Renders' + sequence_name + dataset_number
    camera_filename = os.path.join(dataset_directory,'camera_params.txt')
    
    Camera_proj_matrix = np.genfromtxt(camera_filename, delimiter=' ', skip_header=1, skip_footer=8)
    if verbose==1:
        print('Camera projection matrix:\n', Camera_proj_matrix)
        print()

    Intrinsic_matrix = np.genfromtxt(camera_filename, delimiter=' ', skip_header=6, skip_footer=4)
    if verbose==1:
        print('Instrinsic matrix:\n', Intrinsic_matrix)
        print()

    RT_matrix = np.genfromtxt(camera_filename, delimiter=' ', skip_header=11)
    if verbose==1:
        print('RT matrix:\n', RT_matrix)
        print()

    camera_rotationMatrix = RT_matrix[:, 0:-1]
    if verbose==1:
        print('Rotation matrix:\n', camera_rotationMatrix)
        print()
    camera_translation = RT_matrix[:, -1]
    if verbose==1:
        print('Translation from origin:\n', camera_translation)
        print()

    # RT matrix extended with zeros and 1 below (for dealing with homogeneous coordinates)
    zeros_and_1 = np.zeros(4)
    zeros_and_1[-1] = 1
    zeros_and_1 = np.reshape(zeros_and_1, (1,4))
    RT_extended = np.concatenate((RT_matrix, zeros_and_1), axis=0)
    if verbose==1:
        print("RT extended with zeros and 1 below (for homogeneous coordinates):\n", RT_extended)
        print()

    # Coordinates in the world of the camera center: -R^{-1}*t
    camera_worldCoord_x, camera_worldCoord_y, camera_worldCoord_z = -np.dot(np.linalg.inv(camera_rotationMatrix),
                                                                            camera_translation)
    if verbose==1:
        print('World coordinates of the camera position:', camera_worldCoord_x, camera_worldCoord_y, camera_worldCoord_z)
    
    variables = {'RT_matrix': RT_matrix, 'RT_extended': RT_extended, 'camera_worldCoord_x': camera_worldCoord_x,
                'camera_worldCoord_y': camera_worldCoord_y, 'camera_worldCoord_z': camera_worldCoord_z,
                'Intrinsic_matrix': Intrinsic_matrix, 'Camera_proj_matrix': Camera_proj_matrix}
    return variables

def load_faces(sequence_name = 'TowelWall', dataset_number = '2', verbose=1, reordered = 0):
    """Load the faces file.
    Each face is represented by 4 numbers in a row.
    Each of these numbers represents a vertex.
    Vertices are ordered by row in any file of the form 'RendersTowelWall/vertices_*****.txt'
    There is no heading, so I will import it directly as a numpy array, rather than as a panda DataFrame.
    
    Note: In the faces file, the vertices are indexed starting from 0.
    """
    dataset_directory = 'Renders' + sequence_name + dataset_number
    if reordered==0:
        face_filename = os.path.join(dataset_directory,'faces_mesh.txt')
    elif reordered==1:
        face_filename = os.path.join(dataset_directory,'faces_mesh_reordered.txt')

    faces = genfromtxt(face_filename, delimiter=' ')
    faces = faces.astype(int)
    if verbose==1:    
        print('Number of faces = ' + str(faces.size))

    return faces

def get_picture_size(verbose=0, image_path=None, sequence_name='TowelWall'):
    if sequence_name in ['DeepCloth', 'kinect_tshirt', 'kinect_paper']:
        width, height = 224, 224
    else:
        if image_path is None:
            sequence_name = 'TowelWall'
            dataset_number = '3'
            group_number = '001'
            animation_frame = '00022'
            image_path = os.path.join('Renders' + sequence_name + dataset_number, 'Group.' + group_number, str(int(animation_frame)) + '.png') 
        image = io.imread(image_path)
        if verbose==1:
            print('height, width, channels:', image.shape, "\n")
        width = image.shape[1] 
        height = image.shape[0]
    return width, height 

if __name__=='__main__':
    RGB_tshirt = os.path.join('kinect_tshirt', 'processed', 'color', '100.png')
    for img_path in ['RendersTowelWall11/Group.038/41.png', 'DeepCloth2/train/train_non-text/imgs/000000.png', RGB_tshirt]:
        print("Size of " + img_path + ":", end=' ')
        get_picture_size(1, img_path)

def normalize_uv_01(u=None,v=None, uv=None, verbose=0, width=None, height=None, sequence_name='TowelWall'):
    """ Get u and v be between 0 and 1 by dividing by the (cropped or uncropped) picture size. """
    if width is None:
        width, height = get_picture_size(verbose, sequence_name=sequence_name) # Make these float?
        
    if uv is None:
        # Transform u, v to be within 0<=u,v<=1
        u=u/(width-1) # divide u by width of picture
        v=v/(height-1) # divide v by height of picture
        # Do not need to set as float, as the type of u and v is set in the dataset class already, at least in DeepCloth2
#         u=u.astype(float)/(width-1) # divide u by width of picture
#         v=v.astype(float)/(height-1) # divide v by height of picture
        if width==1 or height==1:
            print('\n'*10 + 'Error: width or height is 1 (we divide by 0 whe normalizing) ')
            print('u:', u, '\n'*10)
        return u, v
    else:
        u=uv[:,0]
        v=uv[:,1]
        u, v = normalize_uv_01(u, v, verbose=verbose, width=width, height=height, sequence_name=sequence_name)
        u = np.reshape(u, (u.shape[0], 1))
        v = np.reshape(v, (v.shape[0], 1))
        uv = np.hstack((u, v))
        return uv

def unnormalize_uv_01(u=None,v=None, uv=None, verbose=0, width=None, height=None, sequence_name='TowelWall'):
    """ Recover original unnormalized values of u and v from their normalized versions between 0 and 1 by multiplying by the (cropped or uncropped) picture size.
    
    In this case, I am roundign to integer values, but I can change that."""
    if width is None:
        width, height = get_picture_size(verbose, sequence_name=sequence_name) # Make these float?
        
    if uv is None:
        # Transform u, v from 0<=u,v<=1 back to the true pixel coordinates
#         u=(np.round(u*(width-1))).astype(int) # multiply u by width of picture
#         v=(np.round(v*(height-1))).astype(int) # multiply v by height of picture
        u=u*(width-1) 
        v=v*(height-1)
        return u, v
    else:
        u=uv[:,0]
        v=uv[:,1]
        u, v = unnormalize_uv_01(u, v, verbose=verbose, width=width, height=height, sequence_name=sequence_name)
        u = np.reshape(u, (u.shape[0], 1))
        v = np.reshape(v, (v.shape[0], 1))
        uv = np.hstack((u, v))
        return uv
    
def unnormalize_uv_01_tensor(u=None,v=None, uv=None, verbose=0, width=None, height=None, sequence_name='TowelWall'):
    """ Recover original unnormalized values of u and v from their normalized versions between 0 and 1 by multiplying by the (cropped or uncropped) picture size. 
    Do this for tensors with may require grad. """
    if width is None:
        width, height = get_picture_size(verbose, sequence_name=sequence_name) # Make these float?
        width, height = float(width), float(height)
        
    if uv is None:
        # Transform u, v from 0<=u,v<=1 back to the true pixel coordinates
        u=u*(width-1) 
        v=v*(height-1)
        return u, v
#     else:
#         u=uv[:,0]
#         v=uv[:,1]
#         u, v = unnormalize_uv_01(u, v, verbose=verbose, width=width, height=height)
#         u = np.reshape(u, (u.shape[0], 1))
#         v = np.reshape(v, (v.shape[0], 1))
#         uv = np.hstack((u, v))
#         return uv

def unnormalize_uv_01_tensor_batch(uv_batch, verbose=0, width=None, height=None, sequence_name='TowelWall'):
    """ Recover original unnormalized values of u and v from their normalized versions between 0 and 1 by multiplying by the (cropped or uncropped) picture size. 
    Do this for tensors of batches with may require grad. """
    if width is None:
        width, height = get_picture_size(verbose, sequence_name=sequence_name) # Make these float?
        width, height = float(width), float(height)
    
    # Transform u, v from 0<=u,v<=1 back to the true pixel coordinates
    uv_batch_unnormalized = torch.zeros(uv_batch.shape, dtype=uv_batch.dtype,
                                         requires_grad=False, device=uv_batch.device)
    uv_batch_unnormalized[:,:,0] = uv_batch[:,:,0]*(width-1) 
    uv_batch_unnormalized[:,:,1] = uv_batch[:,:,1]*(height-1)
    return uv_batch_unnormalized
    
###
### Reshape functions
###
def reshape_labels(labels, batch_size, num_selected_vertices, num_coord_per_vertex=3):
    """
    Reshape set of 2D or 3D coordinates from a matrix to a vector.
    (num_coord_per_vertex is the D in 2D or 3D)
    
    The input 'labels' has shape: (batch_size, num_selected_vertices, num_coord_per_vertex).
    
    The output 'labels' has shape: (batch_size, num_selected_vertices * num_coord_per_vertex).
    E.g., if num_coord_per_vertex==3, labels[0] is a list of the form [x_0, y_0, z_0, ..., x_i, y_i, z_i], 
    where i = num_selected_vertices - 1.
    E.g., if num_coord_per_vertex==2, labels[0] is a list of the form [u_0, v_0, ..., u_i, v_i], 
    where i = num_selected_vertices - 1.
    
    """
    return labels.view(batch_size, num_selected_vertices * num_coord_per_vertex)

def reshape_labels_back(labels, batch_size, num_selected_vertices, num_coord_per_vertex=3):
    """
    Reshape set of 2D or 3D coordinates from a vector to a matrix so that applying
    reshape_labels and then reshape_labels_back will leave the matrix unchanged.
    (num_coord_per_vertex is the D in 2D or 3D)
    
    The input 'labels' has shape: (batch_size, num_selected_vertices * num_coord_per_vertex).
    E.g., if num_coord_per_vertex==3, labels[0] is a list of the form [x_0, y_0, z_0, ..., x_i, y_i, z_i], 
    where i = num_selected_vertices - 1.
    E.g., if num_coord_per_vertex==2, labels[0] is a list of the form [u_0, v_0, ..., u_i, v_i], 
    where i = num_selected_vertices - 1.
    
    The output 'labels' has shape: (batch_size, num_selected_vertices, num_coord_per_vertex).
    """
    return labels.view(batch_size, num_selected_vertices, num_coord_per_vertex)

def reshape_numpy_to_1D(u):
    """
    Args:
    - u is numpy array of shape (N, D), 
    where each row corresponds to the D coordinates of a point.
    E.g., if D=2, u looks like this:
    [[ x_0  y_0]
     [ ...  ...]
     [ x_{N-1}  y_{N-1}]]

    Output:
    - reshape of u with shape (N * D, )
    E.g., D=2, u will be reshaped to [x_0 y_0 ... x_{N-1} y_{N-1}]
    """
    return np.reshape(u, (u.shape[0]*u.shape[1], ))

def reshape_numpy_from_1D(u, D):
    """
    Args:
    - u is numpy array of shape (N * D, )
    E.g., if D=2, u looks like this: [x_0 y_0 ... x_{N-1} y_{N-1}]
    
    Output:
    - reshape of u with (N, D), 
    where each row corresponds to the D coordinates of a point.
    E.g., D=2, u will be reshaped to
    [[ x_0  y_0]
     [ ...  ...]
     [ x_{N-1}  y_{N-1}]]
    """
    
    return np.reshape(u, (u.shape[0]//D, D))

# Example using reshape_numpy_to_1D() and reshape_numpy_from_1D():
# u = np.array([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]) # Shape Nx2
# print(u)
# print(u.shape)
# u_1D = functions_data_processing.reshape_numpy_to_1D(u)
# print(u_1D)
# print(u_1D.shape)
# u_back_from_1D = functions_data_processing.reshape_numpy_from_1D(u_1D, 2)
# print(u_back_from_1D)
# print(u_back_from_1D.shape)

def reshape_numpy_to_1D_xxxyyy(u):
    """
    Args:
    - u is numpy array of shape (N, D), 
    where each row corresponds to the D coordinates of a point.
    E.g., if D=2, u looks like this:
    [[ x_0  y_0]
     [ ...  ...]
     [ x_{N-1}  y_{N-1}]]

    Output:
    - reshape of u with shape (N * D, )
    E.g., D=2, u will be reshaped to [x_0 ... x_{N-1} y_0 ... y_{N-1}]
    """
    return reshape_numpy_to_1D(np.transpose(u))

def reshape_numpy_from_1D_xxxyyy(u, D):
    """
    Args:
    - u is numpy array of shape (N * D, )
    E.g., D=2, u looks like [x_0 ... x_{N-1} y_0 ... y_{N-1}]
    
    Output:
    - reshape of u with (N, D), 
    where each row corresponds to the D coordinates of a point.
    E.g., D=2, u will be reshaped to
    [[ x_0  y_0]
     [ ...  ...]
     [ x_{N-1}  y_{N-1}]]
    """
    return np.transpose(np.reshape(u, (D, u.shape[0]//D)))

def check_all_numpy_reshape_functions():
    """Checking the 4 reshape functions:
    reshape_numpy_to_1D, reshape_numpy_from_1D, reshape_numpy_to_1D_xxxyyy, reshape_numpy_from_1D_xxxyyy"""
    pt_cld_a = np.array([(2.1, 0.0), (0.0, 1.9), (-2.0, 0.0), (0.2, -3.8)]) # Shape (M, 2)
    print('pt_cld_a in the form')
    print('[[ x_0  y_0]\n [ x_1  y_1]\n [ x_2  y_2]\n [ x_3  y_3]]:')
    print(pt_cld_a, '\n') 
    print('pt_cld_a reshaped to [x_0 y_0 x_1 y_1 x_2 y_2 x_3 y_3]:') 
    print(reshape_numpy_to_1D(pt_cld_a), '\n')
    print('pt_cld_a recovered from [x_0 y_0 x_1 y_1 x_2 y_2 x_3 y_3]:') 
    b = reshape_numpy_to_1D(pt_cld_a)
    print(reshape_numpy_from_1D(b, 2), '\n')
    print('pt_cld_a reshaped to [x_0 x_1 x_2 x_3 y_0 y_1 y_2 y_3] :')
    print(reshape_numpy_to_1D_xxxyyy(pt_cld_a), '\n')
    print('pt_cld_a recovered from [x_0 x_1 x_2 x_3 y_0 y_1 y_2 y_3] :')
    b=reshape_numpy_to_1D_xxxyyy(pt_cld_a)
    print(reshape_numpy_from_1D_xxxyyy(b, 2), '\n')
if __name__=='__main__':
    check_all_numpy_reshape_functions()
    
def x_y_from_xy(xy):
    """ Split numpy array of the form 
    xy=[x_0 x_1 x_2 x_3 y_0 y_1 y_2 y_3]
    into two numpy arrays of the form
    x=[x_0 x_1 x_2 x_3]
    y=[y_0 y_1 y_2 y_3] """
    l = len(xy)
    x = xy[:l//2]
    y = xy[l//2:]
    return x, y

def xy_from_x_y(x, y):
    """
    Args:
    - x and y are numpy arrays of shape (N, ) or (N, 1) or (1, N) or ( ,N)
    E.g., x may look like [x_0 ... x_{N-1}]
    E.g., y may look like [y_0 ... y_{N-1}]
    
    Output:
    - numpy array xy of shape (N, 2), 
    where each row corresponds to the 2 coordinates of a point.
    E.g., xy will look like
    [[ x_0  y_0]
     [ ...  ...]
     [ x_{N-1}  y_{N-1}]]
    """
    return np.concatenate((np.reshape(x, (x.shape[0], 1)), np.reshape(y, (y.shape[0], 1))), axis=1)

def xy_from_x_y_tensor(x, y):
    """
    Args:
    - x and y are tensors of shape (N, ) or (N, 1) or (1, N) or ( ,N)
    E.g., x may look like [x_0 ... x_{N-1}]
    E.g., y may look like [y_0 ... y_{N-1}]
    
    Output:
    - tensor xy of shape (N, 2), 
    where each row corresponds to the 2 coordinates of a point.
    E.g., xy will look like
    [[ x_0  y_0]
     [ ...  ...]
     [ x_{N-1}  y_{N-1}]]
    """
    return torch.cat((x.view(x.shape[0], 1), y.view(y.shape[0], 1)), 1)
if __name__=='__main__':
    x = torch.zeros([5, 1])
    y = torch.ones([5, ])
    print(xy_from_x_y_tensor(x, y))

# Set up number of vertices, animation frames, etc.
def num_vertices_in_the_mesh(args):
    if args.sequence_name == 'DeepCloth':
        root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number, args.texture_type)
        num_vertices = data_loading.load_coordinates_DeepCloth2(root_dir, idx=0, dtype=args.dtype, texture_type=args.texture_type)['uv'].shape[0]
    elif args.sequence_name in ['kinect_tshirt', 'kinect_paper']:
        root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number)
        num_vertices = data_loading.load_coordinates_realDatasets(root_dir, idx=100, dtype=args.dtype)['uv'].shape[0]
    else:
        animation_frame = '00001'

        # Load the vertices files disregarding the string '# ' at the beginning of the file
        filename = os.path.join('Renders' + args.sequence_name + args.dataset_number, 'Group.001', 
                                'vertices_' + animation_frame + '.txt')
        f = open(filename, 'r')
        line1 = f.readline()
        df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())
        df_X = df_vertices_all_data['x']
        X = df_X.values
        num_vertices = X.size
    return num_vertices

if __name__ == '__main__':
    # Parser for entering training setting arguments and setting default ones
    args = parser()

    num_vertices = num_vertices_in_the_mesh(args)
    print('num_vertices = ' + str(num_vertices))
    num_vertices_height = (2**2)*13
    print('num_vertices_height =', num_vertices_height)
    num_vertices_width = 103
    print('num_vertices_width =', num_vertices_width)

def uv_transformations(uv, img_name=None, root_dir=None, args=None, group_number=None, animation_frame=None):    
    """ Note: img_name is already cropped if the correct option is on"""
    # Shape of the loaded image without transforms
    width, height = get_picture_size(verbose=0, image_path=img_name, sequence_name=args.sequence_name) 
    # width is the width for the uv normalization
    # For RendersTowelWall datasets, width = 960 for uncropped images. Smaller for cropped images.
    # height is the height for the uv normalization
    # For RendersTowelWall datasets, height = 540 for uncropped images. Smaller for cropped images.
    
    if args.crop_centre_or_ROI==2: 
        # Region of interest. Rectangular box containing the towel
        # Make (u,v)=(0,0) on the upper-left corner of the cropped box
        if root_dir==None:
            root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number)
        directory_name = root_dir + group_number
        text_name = os.path.join(directory_name, 'uv_width_height_rectROI_' + animation_frame +'.txt')
        uvwh = np.genfromtxt(fname=text_name, dtype='int', delimiter=' ', skip_header=1) 
        u_crop_corner = uvwh[0]
        v_crop_corner = uvwh[1]
        width = uvwh[2] # I already load this above as width
        height = uvwh[3] # I already load this above as height
#             u_visible_crop = u_visible - u_crop_corner
#             v_visible_crop = v_visible - v_crop_corner
#             u_occluded_crop = u_occluded - u_crop_corner
#             v_occluded_crop = v_occluded - v_crop_corner
        uv[:,0] -= u_crop_corner
        uv[:,1] -= v_crop_corner

#             # Adapt uv to the rescaling, to make 0<=u,v<=223 rather than 0<=u,v<=1
#             width_rescaled_image = image.shape[2]
#             height_rescaled_image = image.shape[1]
#             if args.uv_normalization==0:# Normalize to 0<=u,v<=223
#                 if width_rescaled_image==224 and height_rescaled_image==224:
#                     uv = functions_data_processing.normalize_uv_01(uv=uv, width=width_crop/width_rescaled_image, height=height_crop/height_rescaled_image) # No!!!! we should first apply normaliation with width_crop and thenw ith width_rescaled_image, since there is a -1 factor being applied in the normalization, so doing it in 1 step means normalizing with (width_crop/width_rescaled_image)-1, whereas doing it in 2 steps means doing it with (width_crop-1)/(width_rescaled_image-1)
    if args.uv_normalization==2: 
        uv = unnormalize_uv_01(uv=uv, width=224, height=224, sequence_name=args.sequence_name) 

    if args.uv_normalization in [1,2]: 
        uv = normalize_uv_01(uv=uv, width=width, height=height, sequence_name=args.sequence_name) 
        
    if np.isinf(uv[0,0]) or np.isnan(uv[0,0]):
        print("NaN uv values in " + img_name)
    return uv

def uv_transformations_back(uv, img_name=None, root_dir=None, args=None, group_number=None, animation_frame=None):    
    """ Do the opposite of uv_transformations()"""
    # Shape of the loaded image without transforms
    width, height = get_picture_size(verbose=0, image_path=img_name, sequence_name=args.sequence_name) 
    # width is the width for the uv normalization
    # For RendersTowelWall datasets, width = 960 for uncropped images. Smaller for cropped images.
    # height is the height for the uv normalization
    # For RendersTowelWall datasets, height = 540 for uncropped images. Smaller for cropped images.
    
    if args.uv_normalization in [1,2]: 
        uv = unnormalize_uv_01(uv=uv, width=width, height=height, sequence_name=args.sequence_name) 

    if args.uv_normalization==2: 
        uv = normalize_uv_01(uv=uv, width=224, height=224, sequence_name=args.sequence_name) 

    if args.crop_centre_or_ROI==2: 
        if root_dir==None:
            root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number)
#         # To be done
#         # Region of interest. Rectangular box containing the towel
#         # Make (u,v)=(0,0) on the upper-left corner of the cropped box
#         directory_name = root_dir + group_number
#         text_name = os.path.join(directory_name, 'uv_width_height_rectROI_' + animation_frame +'.txt')
#         uvwh = np.genfromtxt(fname=text_name, dtype='int', delimiter=' ', skip_header=1) 
#         u_crop_corner = uvwh[0]
#         v_crop_corner = uvwh[1]
#         width = uvwh[2] # I already load this above as width
#         height = uvwh[3] # I already load this above as height
#         uv[:,0] -= u_crop_corner
#         uv[:,1] -= v_crop_corner

    if np.isinf(uv[0,0]) or np.isnan(uv[0,0]):
        print("NaN uv values in " + img_name)
    return uv
            
def process_uv(sample, img_name, root_dir, args, group_number, animation_frame):
    for key in ['uv', 'towel_pixel_subsample', 'towel_pixel_contour_subsample']:
        if key in sample:
            sample[key] = uv_transformations(
                uv=sample[key], img_name=img_name, root_dir=root_dir, args=args, group_number=group_number,
                animation_frame=animation_frame)
    return sample

def normalize_min_max_to_0_1(observation, minimum, maximum):
    """ Normalize an observation with the min and max of the training set, 
    so that the min and max after this has been applied to all training set becomes 0 and 1."""
    return (observation-minimum)/(maximum-minimum)

def unnormalize_min_max_from_0_1(observation, minimum, maximum):
    """ Recover an observation which has been resized via normalize_min_max_to_0_1().
    This can also be applied to a batch."""
    return observation*(maximum-minimum) + minimum
    
def process_xyz(sample, normalize_xyz_min, normalize_xyz_max):
    """Normalize vertex coordinates 0<=x,y,z<=1 with min and max from training."""
    if 'xyz' in sample and normalize_xyz_min is not None:
        for i in range(3): 
            sample['xyz'][:,i] = normalize_min_max_to_0_1(sample['xyz'][:,i], normalize_xyz_min[i], normalize_xyz_max[i])
    return sample

def process_D(sample, normalize_D_min, normalize_D_max):
    """Normalize vertex depth 0<=D<=1 with min and max from training."""
    if 'D' in sample and normalize_D_min is not None:
        for i in range(1): 
            sample['D'][:,i] = normalize_min_max_to_0_1(sample['D'][:,i], normalize_D_min, normalize_D_max)
    return sample

def unprocess_uv(sample, img_name, root_dir, args, group_number, animation_frame):
    """Recover a sample which has been processed via process_uv()."""
    for key in ['uv', 'towel_pixel_subsample', 'towel_pixel_contour_subsample']:
        if key in sample:
            sample[key] = uv_transformations_back(
                uv=sample[key], img_name=img_name, root_dir=root_dir, args=args, group_number=group_number,
                animation_frame=animation_frame)
    return sample

def unprocess_D(sample, normalize_D_min, normalize_D_max):
    """Recover a sample which has been processed via process_D()."""
    if 'D' in sample and normalize_D_min is not None:
        for i in range(1): 
            sample['D'][:,i] = unnormalize_min_max_from_0_1(sample['D'][:,i], normalize_D_min, normalize_D_max)
    return sample

def unprocess_D_batch(D_batch, normalize_D_min=0, normalize_D_max=1):
    """Recover a batch which has been processed via process_D().
    Do this for tensors of batches with may require grad. """
    D_batch_unnormalized = unnormalize_min_max_from_0_1(D_batch, normalize_D_min, normalize_D_max)
    return D_batch_unnormalized

def unprocess_xyz_batch(xyz_batch, normalize_xyz_min=[0,0,0], normalize_xyz_max=[1,1,1]):
    """Recover a batch which has been processed via process_xyz().
    Do this for tensors of batches with may require grad. """
    batch_size = xyz_batch.shape[0]
    num_selected_vertices = xyz_batch.shape[1]
    xyz_batch_linear = xyz_batch.view(-1, 3) # Vertically stack the data of all elements in the batch
    xyz_batch_linear_unnormalized = torch.zeros(
        xyz_batch_linear.shape, device=xyz_batch_linear.device, dtype=xyz_batch_linear.dtype, requires_grad=False)
    if normalize_xyz_min!=[0,0,0]:
        for i in range(3): 
            xyz_batch_linear_unnormalized[:,i] = unnormalize_min_max_from_0_1(
                xyz_batch_linear[:,i], normalize_xyz_min[i], normalize_xyz_max[i])
        return xyz_batch_linear_unnormalized.view(batch_size, num_selected_vertices, 3)
    else:
        return xyz_batch_linear.view(batch_size, num_selected_vertices, 3)

def load_min_max_xyz_from_training(args, model_directory=None):
    """ Load from the file created during the training of the model. """
    if args.normalization == 3:
        # Load min and max of x, y, z coordinates of all vertices within the training set.
        if model_directory is None:
            model_directory=functions_train.trained_model_directory(args)
        file_name = os.path.join(model_directory, 'normalization_params_from_training.txt')
        loaded_vars = np.genfromtxt(file_name, delimiter=' ', skip_header=1).astype(np.float32)
        normalize_xyz_min, normalize_xyz_max = loaded_vars.tolist()[:3], loaded_vars.tolist()[3:]
        text_to_print = "x_min, y_min, z_min, x_max, y_max, z_max of training set"
        print("Loaded " + text_to_print + " from " + file_name + ":")
        print("%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" % (normalize_xyz_min[0], normalize_xyz_min[1], normalize_xyz_min[2], 
                                                       normalize_xyz_max[0], normalize_xyz_max[1], normalize_xyz_max[2]))
    else: 
        normalize_xyz_min, normalize_xyz_max = [0,0,0], [1,1,1]
    return normalize_xyz_min, normalize_xyz_max
    
def load_min_max_D_from_training(args, model_directory=None):
    """ Load from the file created during the training of the model. """
    if args.D_normalization == 3:
        # Load min and max of depth of all vertices within the training set.        
        if model_directory is None:
            model_directory=functions_train.trained_model_directory(args)
        file_name = os.path.join(model_directory, 'depth_params_from_training.txt')
        loaded_vars = np.genfromtxt(file_name, delimiter=' ', skip_header=1).astype(np.float32)
        normalize_D_min, normalize_D_max = loaded_vars.tolist()[0], loaded_vars.tolist()[1]
        text_to_print = "D_min, D_max from training set"
        print("Loaded " + text_to_print + " from " + file_name + ":")
        print("%8.2f %8.2f" % (normalize_D_min, normalize_D_max))
    else: 
        normalize_D_min, normalize_D_max = 0, 1 # normalizing with these values does nothing
    return normalize_D_min, normalize_D_max

def normalized_uvD_to_unnormalized_xyz_batch(uv_normalized, D_normalized, args, normalize_D_min=0, normalize_D_max=1,
                                            f_u=600.172, f_v=600.172):            
    # Unnormalize uv and D if necessary
    uv_unnormalized = unnormalize_uv_01_tensor_batch(
        uv_normalized, sequence_name=args.sequence_name)
    D_unnormalized = unprocess_D_batch(D_normalized, normalize_D_min, normalize_D_max)

    # convert uv+D to xyz and unnormalize it if necessary (xyz will be used like this in the loss)
    xyz_unnormalized = uvD_to_xyz_batch(uv_unnormalized, D_unnormalized, resW=223., resH=223., f_u=f_u, f_v=f_v)
    return xyz_unnormalized

def normalized_uvD_to_normalized_xyz_batch(uv_normalized, D_normalized, args, normalize_D_min=0, normalize_D_max=1,
                                          normalize_xyz_min=[0,0,0], normalize_xyz_max=[1,1,1], f_u=600.172, f_v=600.172):   
    xyz_unnormalized = normalized_uvD_to_unnormalized_xyz_batch(
        uv_normalized, D_normalized, args, normalize_D_min, normalize_D_max, f_u=f_u, f_v=f_v)
    if args.normalization==3:
        xyz_normalized = unprocess_xyz_batch(xyz_unnormalized, normalize_xyz_min, normalize_xyz_max)
        return xyz_normalized
    else: 
        return xyz_unnormalized

def normalize_weights(args):
    """ Normalize old loss weights so that they sum up to 1."""
    if args.loss_weights==0:
        total_sum = (args.w_uv + args.w_xyz + args.neighb_dist_weight + args.w_chamfer_GT_pred + args.w_chamfer_pred_GT + 
                         args.w_chamfer_GTcontour_pred + args.w_chamfer_pred_GTcontour + args.w_normals)
        if total_sum!=0:
            args.w_uv/=total_sum
            args.w_xyz/=total_sum
            args.neighb_dist_weight/=total_sum
            args.w_chamfer_GT_pred/=total_sum
            args.w_chamfer_pred_GT/=total_sum
            args.w_chamfer_GTcontour_pred/=total_sum
            args.w_chamfer_pred_GTcontour/=total_sum
            args.w_normals/=total_sum
        print('Old loss weights: w_uv,\t w_xyz,\t neighb_dist_weight, w_chamfer_GT_pred, w_chamfer_pred_GT, w_chamfer_GTcontour_pred, w_chamfer_pred_GTcontour, w_normals:')
        print(args.w_uv, args.w_xyz, args.neighb_dist_weight, args.w_chamfer_GT_pred, args.w_chamfer_pred_GT, args.w_chamfer_GTcontour_pred, args.w_chamfer_pred_GTcontour, args.w_normals)
    return args

def normalize_loss_weights(args):
    """ Normalize new loss weights so that they sum up to 1."""
    if args.loss_weights==1:
        total_sum = (args.loss_w_uv + args.loss_w_xyz + args.loss_w_D + args.loss_w_geo + args.loss_w_horizConsecEdges + args.loss_w_verConsecEdges + args.loss_w_chamfer_GT_pred + args.loss_w_chamfer_pred_GT + 
                         args.loss_w_chamfer_GTcontour_pred + args.loss_w_chamfer_pred_GTcontour + args.loss_w_normals)
        if total_sum!=0:
            args.loss_w_uv/=total_sum
            args.loss_w_xyz/=total_sum
            args.loss_w_D/=total_sum
            args.loss_w_geo/=total_sum
            args.loss_w_horizConsecEdges/=total_sum
            args.loss_w_verConsecEdges/=total_sum
            args.loss_w_chamfer_GT_pred/=total_sum
            args.loss_w_chamfer_pred_GT/=total_sum
            args.loss_w_chamfer_GTcontour_pred/=total_sum
            args.loss_w_chamfer_pred_GTcontour/=total_sum
            args.loss_w_normals/=total_sum
        print('New loss weights:\nw_uv,\tw_xyz,\tw_D,\tw_geo,\tw_horizConsecEdges,\tw_verConsecEdges:')
        print(str(args.loss_w_uv) + "\t" + str(args.loss_w_xyz) + "\t" + str(args.loss_w_D) + "\t" + str(args.loss_w_geo) + "\t" + str(args.loss_w_horizConsecEdges) + "\t\t\t" + str(args.loss_w_verConsecEdges))
        print('w_chamfer_GT_pred,\tw_chamfer_pred_GT,\tw_chamfer_GTcontour_pred,\tw_chamfer_pred_GTcontour,\tw_normals:')
        print(str(args.loss_w_chamfer_GT_pred) + "\t\t\t" + str(args.loss_w_chamfer_pred_GT) + "\t\t\t" + str(args.loss_w_chamfer_GTcontour_pred) + "\t\t\t\t" + str(args.loss_w_chamfer_pred_GTcontour) + "\t\t\t\t" + str(args.loss_w_normals))
    return args

def weight_processing(args):
    # Normalize loss weights to sum up to 1
    if args.normWeights1==1:
        args = normalize_weights(args) # Deprecated
        args = normalize_loss_weights(args)

    # Sum the weights of both Hausdorff directions if wanting to use them combined
    if args.n_outputs==1:
        args.loss_w_chamfer = args.loss_w_chamfer_GT_pred + args.loss_w_chamfer_pred_GT
        args.loss_w_chamfer_contour = args.loss_w_chamfer_GTcontour_pred + args.loss_w_chamfer_pred_GTcontour
    else: 
        args.loss_w_chamfer = 0
        args.loss_w_chamfer_contour = 0
    return args

def forget_depth(uvD):
    """ Input: uv pixel coordinates and depth D
    Output: uv pixel coordinates"""
    return uvD[:, :2]

def create_kwargs_normalize(normalize_xyz_min, normalize_xyz_max, normalize_D_min, normalize_D_max, args):
    # normalization for creating the dataset class
    kwargs_normalize={'normalize_xyz_min':normalize_xyz_min, 'normalize_xyz_max':normalize_xyz_max,
                          'normalize_D_min':normalize_D_min, 'normalize_D_max':normalize_D_max}
    if args.normalize_datasetClass_D_min is not None:
        kwargs_normalize['normalize_D_min'] = args.normalize_datasetClass_D_min
        kwargs_normalize['normalize_D_max'] = args.normalize_datasetClass_D_max
    if args.normalize_datasetClass_x_min is not None:
        kwargs_normalize['normalize_xyz_min'] = [args.normalize_datasetClass_x_min, args.normalize_datasetClass_y_min, args.normalize_datasetClass_z_min]
        kwargs_normalize['normalize_xyz_max'] = [args.normalize_datasetClass_x_max, args.normalize_datasetClass_y_max, args.normalize_datasetClass_z_max]
    # normalization for creating labels and predictions
    kwargs_normalize_labels_pred={'normalize_xyz_min':normalize_xyz_min, 'normalize_xyz_max':normalize_xyz_max,
                      'normalize_D_min':normalize_D_min, 'normalize_D_max':normalize_D_max}
    if args.normalize_labelsPred_D_min is not None:
        kwargs_normalize_labels_pred['normalize_D_min'] = args.normalize_labelsPred_D_min
        kwargs_normalize_labels_pred['normalize_D_max'] = args.normalize_labelsPred_D_max
    if args.normalize_labelsPred_x_min is not None:
        kwargs_normalize_labels_pred['normalize_xyz_min'] = [args.normalize_labelsPred_x_min, args.normalize_labelsPred_y_min, args.normalize_labelsPred_z_min]
        kwargs_normalize_labels_pred['normalize_xyz_max'] = [args.normalize_labelsPred_x_max, args.normalize_labelsPred_y_max, args.normalize_labelsPred_z_max]
    return kwargs_normalize, kwargs_normalize_labels_pred