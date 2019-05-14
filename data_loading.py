
# coding: utf-8

# # CAVEAT: All changes to this file prior to 26/09/18 must be looked for at the git history of data_loading_ipynb_old.ipynb
# To visualize these functions, see data_loading-visualization.py

# **Author**: `Francisco Belchí <frbegu@gmail.com>, <https://github.com/KikoBelchi/2d_to_3d>`_

###
### Imports
###
from __future__ import print_function, division
import itertools
import os
import torch
import pandas as pd
from skimage import io, transform # package 'scikit-image'
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import binary_image
import submesh
import functions_data_processing
import functions_train

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# All functions involving plotting have can be found in data_loading-visualization.py
# Imports for plotting
# import matplotlib.pyplot as plt # Do not use when running on the server
# from mpl_toolkits.mplot3d import axes3d # Do not use when running on the server

# Allow the interactive rotation of 3D scatter plots in jupyter notebook
import sys    
import os    
file_name =  os.path.basename(sys.argv[0])
# file_name == 'ipykernel_launcher.py' # This basicaly asks whether this file is a jupyter notebook

if __name__ == "__main__":
    if file_name == 'ipykernel_launcher.py': # Run only in .ipynb, not in exported .py scripts
        get_ipython().run_line_magic('matplotlib', 'notebook') # Equivalent to ''%matplotlib notebook', but it is also understood by .py scripts



###        
### RGBA to RBG functions
###
# The original Blender pictures in our dataset are RGBA with a constant alpha = 255, 
# i.e., where all pixels are 100% opace. 
# Therefore, they are actually RGB images which contain a 4th constant channel

def RGBA_to_RGB(image_RGBA):
    image_RGB = image_RGBA[:,:,0:3]
    return image_RGB

# For tensors, which have the channels reordered as CxHxW:
def RGBA_to_RGB_tensors(tensor_RGBA):
    tensor_RGB = tensor_RGBA[0:3,:,:]
    return tensor_RGB





# # Dataset class
# 
# Sample of our dataset will be a dict
# ``{'image': image, 'xyz': xyz, 'img_name': img_name}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. 

# ### Index to group and frame
# In the 'vertices_Dataset' class, we will use '__getitem__' to get a data element given a single key index.
# Hence, we will need to convert such index into a 'group_number' and an 'animation_frame',
# so that it is easy to load the data.
# To that end, we define the following function

# num_groups = 40 # There are 40 groups of animations in TowelWall2
# # print('num_groups =', num_groups)
# num_animationFramesPerGroup = 99 # Number of animation frames per group
# # print('num_animationFramesPerGroup =', num_animationFramesPerGroup)

def group_and_frame_from_idx(idx, num_groups = 40, num_animationFramesPerGroup = 99):
    group_number = int(int(idx) // num_animationFramesPerGroup) + 1
    animation_frame = (idx + 1) % num_animationFramesPerGroup
    if animation_frame == 0: animation_frame = num_animationFramesPerGroup
    return group_number, animation_frame

def post_process_grp_and_frame(group_number, animation_frame):
    """ Set group and animation frame in string format ready for data loading """
    group_number = str(group_number)
    group_number = group_number.zfill(3) # pad with zeros on the left until the string has length 3
    animation_frame = str(int(animation_frame)) 
    return group_number, animation_frame




###
### Group and frame to index
###
# In the 'random_split_notMixingSequences' function, we will set the indices of the elements of each of the training and test sets by using the group and animation frame numbers.
# Hence, we will need to convert such 'group_number' and 'animation_frame' into 'idx'
# To that end, we define the following functions

def idx_from_group_and_frame(group, frame, num_groups = 40, 
                             num_animationFramesPerGroup = 99):
    idx = num_animationFramesPerGroup*(group-1) + frame - 1
    return idx

def range_of_idxs_from_group(group, num_groups = 40,
                             num_animationFramesPerGroup = 99):
    idx_first = idx_from_group_and_frame(group=group, frame=1, num_groups=num_groups, num_animationFramesPerGroup=num_animationFramesPerGroup)
    idx_last = idx_from_group_and_frame(group=group, frame=num_animationFramesPerGroup, num_groups=num_groups, num_animationFramesPerGroup=num_animationFramesPerGroup)
    return range(idx_first, idx_last+1)

# range_of_idxs = range_of_idxs_from_group(1)
# print(range_of_idxs)
# range_of_idxs = range_of_idxs_from_group(2)
# print(range_of_idxs)










###
### Normalize vertex 3D coordinates
###
def move_barycentre_to_origin(xyz, num_normalizing_landmarks=None):
    """Each row of 'xyz' represents the 3D coordinates of a vertex. 
    'xyz = move_barycentre_to_origin(xyz, num_normalizing_landmarks)' 
    moves all the vertices in 'xyz' using the translation which
    sends the barycenter of the first 'num_normalizing_landmarks' vertices in 'xyz'
    to the origin of the coordinate system.
    
    If num_normalizing_landmarks==None, then all points in xyz are used for the normalization.
    """
    #     print(xyz)
    if num_normalizing_landmarks == None:
        num_normalizing_landmarks = xyz.shape[0]
    barycentre = np.mean(xyz[0:num_normalizing_landmarks, :], axis=0)
#     print(barycentre)
    xyz -= barycentre
#     print(xyz)
#     barycentre = np.mean(xyz, axis=0)
#     print(barycentre) # It works, since this is now [0, 0, 0] with a tolerance of about e-15.
    return xyz
            
# def move_barycentre_to_origin(xyz):
#     """Each row of 'xyz' represents the 3D coordinates of a vertex. 
#     'xyz = move_barycentre_to_origin(xyz)' 
#     moves all the vertices in 'xyz' using the translation which
#     sends the their barycenter
#     to the origin of the coordinate system.
#     """
# #     print(xyz)
#     barycentre = np.mean(xyz, axis=0)
# #     print(barycentre)
#     xyz -= barycentre
# #     print(xyz)
# #     barycentre = np.mean(xyz, axis=0)
# #     print(barycentre) # It works, since this is now [0, 0, 0] with a tolerance of about e-15.
#     return xyz

def l2_norm_of_each_row(a):
    """Most efficient way to compute the L2 norm of each row in a matrix 'a'.
    See https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix/19794741#19794741
    """
    return np.sqrt(np.einsum('ij,ij->i', a, a))
    
def fit_tight_inside_ball(xyz, num_normalizing_landmarks=None):       
    """Each row of 'xyz' represents the 3D coordinates of a vertex. 
      The barycentre of the vertices in 'xyz' is assumed to be at the origin of the coordinate system.
      If that is not the case, run:
      'xyz = move_barycentre_to_origin(xyz, num_normalizing_landmarks)'.
      'xyz = fit_tight_inside_ball(xyz, num_normalizing_landmarks)' scales the vertices 
      of 'xyz' so that:
          1. the first 'num_normalizing_landmarks' vertices sit inside the closed unit ball centred at the origin, and 
          2. out of the first 'num_normalizing_landmarks' vertices, 
             the one which is furthest away from the centre lies on the boundary of the ball.
             
       If num_normalizing_landmarks==None, then all points in xyz are used for the normalization.
    """
    if num_normalizing_landmarks == None:
        num_normalizing_landmarks = xyz.shape[0]
#     print(l2_norm_of_each_row(xyz))
    norm_of_furthest_vertex = np.amax(l2_norm_of_each_row(xyz[0:num_normalizing_landmarks, :]))
#     print(norm_of_furthest_vertex)
    # Divide each vertex by the norm of the vertex which is the furthest away from the centre.
    xyz /= norm_of_furthest_vertex
#     print(xyz)
#     print(l2_norm_of_each_row(xyz))
#     norm_of_furthest_vertex = np.amax(l2_norm_of_each_row(xyz))
#     print(norm_of_furthest_vertex) # It works, since the norm of the furthest vertex is now 1.
    return xyz

# def fit_tight_inside_ball(xyz):
#     """Each row of 'xyz' represents the 3D coordinates of a vertex. 
#       The barycentre of the vertices in 'xyz' is assumed to be at the origin of the coordinate system.
#       If that is not the case, run:
#       'xyz = move_barycentre_to_origin(xyz)'.
#       'xyz = fit_tight_inside_ball(xyz)' scales the vertices 
#       so that:
#           1. all of them sit inside the closed unit ball centred at the origin, and 
#           2. the vertex which is furthest away from the centre lies on the boundary of the ball.
#     """
# #     print(l2_norm_of_each_row(xyz))
#     norm_of_furthest_vertex = np.amax(l2_norm_of_each_row(xyz))
# #     print(norm_of_furthest_vertex)
#     # Divide each vertex by the norm of the vertex which is the furthest away from the centre.
#     xyz /= norm_of_furthest_vertex
# #     print(xyz)
# #     print(l2_norm_of_each_row(xyz))
# #     norm_of_furthest_vertex = np.amax(l2_norm_of_each_row(xyz))
# #     print(norm_of_furthest_vertex) # It works, since the norm of the furthest vertex is now 1.
#     return xyz

def root_dir_from_dataset(sequence_name, dataset_number, texture_type='train_non-text'):
    if sequence_name == 'TowelWall':
        root_dir = os.path.join('Renders' + sequence_name + dataset_number, 'Group.')
    elif sequence_name == 'DeepCloth':
        root_dir = os.path.join(sequence_name + dataset_number, 'train', texture_type)
    elif sequence_name in ['kinect_tshirt', 'kinect_paper']:
        root_dir = sequence_name
    return root_dir

def Deep_Cloth2_length_function():
    """
    DeepCloth2/train/train_low-text    contains files 000000.png to 034559.png
    DeepCloth2/train/train_non-text    contains files 000000.png to 047999.png
    DeepCloth2/train/train_struct-text contains files 000000.png to 008319.png
    DeepCloth2/train/train_text        contains files 000000.png to 062719.png
    """
    DeepCloth2_lengths = {'train_low-text': 34560, 'train_non-text': 48000, 'train_struct-text': 8320, 'train_text': 62720}
    DeepCloth2_lengths['']=sum(DeepCloth2_lengths.values())
    return DeepCloth2_lengths
DeepCloth2_lengths = Deep_Cloth2_length_function()

def dataset_length_from_name(args):
    if args.sequence_name == 'TowelWall':
        return args.num_groups * args.num_animationFramesPerGroup
    elif args.sequence_name == 'DeepCloth':
        return DeepCloth2_lengths[args.texture_type]
    elif args.sequence_name == 'kinect_tshirt':
        # There are files 2.csv to 313.csv in kinect_tshirt/processed/vertexs: a total of 312 files
        # There are files 3.png to 313.csv in kinect_tshirt/processed/color: a total of 311 files
        return 311 # For the moment, use files 3 to 313.
    elif args.sequence_name == 'kinect_paper':
        # There are files 2.csv to 193.csv in kinect_paper/processed/vertexs: a total of 192 files
        # There are files 2.png to 193.csv in kinect_paper/processed/color: a total of 192 files
        return 192 # For the moment, use files 2 to 193.
    
def imgNameEnd_from_cropOption(crop_centre_or_ROI):
    if crop_centre_or_ROI in [0,3]: # centre crop or no crop
        img_name_end = '.png'
    elif crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
        img_name_end = '_ROI.png'
    elif crop_centre_or_ROI==2: # Region of interest. Rectangular box containing the towel
        img_name_end = '_rectROI.png'
    crop_option = img_name_end.split(".")[0]
    return img_name_end, crop_option

def root_dir_and_idx_from_idx(idx, root_dir, DeepCloth2_lengths=DeepCloth2_lengths):
    """ When using all 4 texture styles of DeepCloth2, 
    the observation of index idx within the dataset class corresponds to 
    an observation of smaller or same index within one of the 4 texture directories.
    This functinos returs the root_dir of the corresponding texture and the idx within that directory. """
    low, non = DeepCloth2_lengths['train_low-text'], DeepCloth2_lengths['train_non-text']
    struct, involved = DeepCloth2_lengths['train_struct-text'], DeepCloth2_lengths['train_text']
    if idx < low:
        root_dir = os.path.join(root_dir, 'train_low-text')
    elif idx < (low+non):
        idx-=low
        root_dir = os.path.join(root_dir, 'train_non-text')
    elif idx < (low+non+struct):
        idx-=(low+non)
        root_dir = os.path.join(root_dir, 'train_struct-text')
    elif idx < (low+non+struct+involved):
        idx-=(low+non+struct)
        root_dir = os.path.join(root_dir, 'train_text')
    return idx, root_dir

def imgName_from_idx(root_dir, args, img_name_end='.png', idx=None):
    idx = int(idx) # so that, in the dataloders, idx is not used as a tensor of a single element, but an element
    if args.sequence_name=='TowelWall':
        group_number, animation_frame = group_and_frame_from_idx(
            idx, num_groups=args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup)
        group_number, animation_frame = post_process_grp_and_frame(group_number, animation_frame)
        img_name = os.path.join(root_dir + group_number, animation_frame + img_name_end)
    elif args.sequence_name=='DeepCloth':
        if args.texture_type=='': idx, root_dir = root_dir_and_idx_from_idx(idx, root_dir)                
        img_name = os.path.join(root_dir, 'imgs', str(idx).zfill(6) + img_name_end)
    elif args.sequence_name in ['kinect_tshirt', 'kinect_paper']:
        # Processed image: resized to 224x224, background turned to black
        img_name = os.path.join(root_dir, 'processed', 'color', str(idx) + img_name_end)
    return img_name
        
def im_from_imgName(img_name, transform):
    if transform:          
        im = Image.open(img_name) # type: <class 'PIL.PngImagePlugin.PngImageFile'>
        transformed_image = transform(im)  

        # If the tensor represents an RGBA image, turn it to RGB
        if transformed_image.shape[0] == 4: transformed_image = RGBA_to_RGB_tensors(transformed_image)

        # To use 'torchvision.utils.make_grid', use:
        image = transformed_image
        # To use any other way of plotting, use:
#             image = tensor_to_plot(transformed_image, self.mean_for_normalization, self.std_for_normalization)
    else:
        image = io.imread(img_name)

        # If the image is RGBA, turn it to RGB
        if image.shape[2] == 4: image = RGBA_to_RGB(image)
    return image
            
def load_coordinates_RendersTowelWall(self, group_number, animation_frame):
    """ Load vertex and normal vector coordinates of datasets of the form RendersTowelWallxx """
    if self.args.reordered_dataset == 0:
        filename = os.path.join(self.root_dir + group_number, 'vertices_' + animation_frame.zfill(5) + '.txt')
        f = open(filename, 'r')
        line1 = f.readline()
        df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())
        if self.args.predict_uv_or_xyz=='xyz' or self.args.predict_uv_or_xyz=='uvy':
            # x, y, z world coordinates of the first 'num_vertices' vertices and their normal vectors as numpy arrays
            xyz = df_vertices_all_data[['x', 'y', 'z']].values[range(self.args.num_selected_vertices)] 
            if self.args.dtype==0: xyz=xyz.astype(np.float32)
            if self.args.w_normals!=0:
                normal_coordinates = df_vertices_all_data[['nx', 'ny', 'nz']].values[range(self.args.num_selected_vertices)] 
        if self.args.predict_uv_or_xyz=='uv' or self.args.predict_uv_or_xyz=='uvy':
            # u, v coordinates of the first 'num_vertices' vertices as numpy arrays
            uv = df_vertices_all_data[['u','v']].values[range(self.args.num_selected_vertices)] 
    elif self.args.reordered_dataset == 1:
        filename = os.path.join(self.root_dir + group_number, 'vertices_' + animation_frame.zfill(5) + '_reordered')
        if self.args.GTxyz_from_uvy==0:
            filename += '.txt'
            f = open(filename, 'r')
            line1 = f.readline()
            df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.split())
            df_vertices_all_data = df_vertices_all_data.ix[self.submesh_idx] # Select the submesh only
            if self.args.predict_uv_or_xyz=='xyz' or self.args.predict_uv_or_xyz=='uvy':
                # x, y, z world coordinates of vertices and normals of the submesh
                xyz = df_vertices_all_data[['x', 'y', 'z']].values 
                if self.args.dtype==0: xyz=xyz.astype(np.float32)
                if self.args.w_normals!=0:
                    normal_coordinates = df_vertices_all_data[['nx', 'ny', 'nz']].values 
            if self.args.predict_uv_or_xyz=='uv' or self.args.predict_uv_or_xyz=='uvy':
                # u, v coordinates of the submesh
                uv = df_vertices_all_data[['u', 'v']].values
                if self.args.verbose==1:
                    if np.amin(uv)<0 or np.amax(uv)>960:
                        print('\n'*10 + 'Error: uv ROI box has width or height = 1 (we divide by 0 when normalizing) ')
                        print('np.amin(uv), np.amax(uv):', np.amin(uv), np.amax(uv))
                        print(img_name, '\n'*10)
        else:
            filename += '_uvxyz_world_coord_from_uvyGT.txt'
            loaded_vars = np.genfromtxt(filename, delimiter=',', skip_header=1).astype(np.float32)
            loaded_vars = loaded_vars[self.submesh_idx] # Select the submesh only
            uv = loaded_vars[:, :2]
            xyz = loaded_vars[:, 2:]

    if self.args.predict_uv_or_xyz=='xyz': coord={'xyz': xyz}
    elif self.args.predict_uv_or_xyz=='uv': coord={'uv': uv}
    elif self.args.predict_uv_or_xyz=='uvy': coord={'xyz': xyz, 'uv': uv}
    if self.args.w_normals!=0: coord['normal_coordinates']=normal_coordinates
    for key in coord:
            coord[key]=coord[key].astype(np.float32) if self.args.dtype==0 else coord[key].astype(np.float64)
    return coord

def load_coordinates_DeepCloth2(root_dir, idx, dtype, texture_type):
    """ Load uv+D vertex coordinates the datasets DeepCloth2. 
    The output consists of 2 numpy arrays, uv and D:
    u_pixel, v_pixel, depth = uv[:, 0], uv[:, 1], D"""
    if texture_type=='': idx, root_dir = root_dir_and_idx_from_idx(idx, root_dir)                
    vertex_filename = os.path.join(root_dir, 'poses', str(int(idx)).zfill(6) + '.csv')
    # int(idx) is needed so that, in the dataloders, idx is not used as a tensor of a single element, but an element
    uvD = np.genfromtxt(vertex_filename, delimiter=',') #.astype(np.float32)
    coord = {'uv': uvD[:,0:2], 'D': np.reshape(uvD[:, 2], (-1, 1))}
    for key in coord:
        coord[key]=coord[key].astype(np.float32) if dtype==0 else coord[key].astype(np.float64)
    return coord

def load_coordinates_realDatasets(root_dir='kinect_tshirt', idx=3, dtype=0):
    """ Load uv+D vertex coordinates of real datasets such as 'kinect_tshirt' and 'kinect_paper'. 
    The output consists of 2 numpy arrays, uv and D:
    u_pixel, v_pixel, depth = uv[:, 0], uv[:, 1], D"""
    # Loading uvD from processed images
    vertex_filename = os.path.join(root_dir, 'processed', 'vertexs', str(int(idx)) + '.csv')
    uvD = np.genfromtxt(vertex_filename, delimiter=',') # numpy array of shape (81, 3)
    coord = {'uv': uvD[:,0:2], 'D': np.reshape(uvD[:, 2], (-1, 1))}
    for key in coord:
        coord[key]=coord[key].astype(np.float32) if dtype==0 else coord[key].astype(np.float64)
    return coord

def towel_pixel_subsample_filename(args, root_dir, crop_option, idx=None, group_number=None, animation_frame=None):
    if args.sequence_name == 'TowelWall':
        filename = os.path.join(root_dir + str(group_number).zfill(3), 'vertices_' + str(animation_frame).zfill(5)) + crop_option
    else:
        directory_name = os.path.join(root_dir, 'Subsamples')
        filename = os.path.join(directory_name, str(idx).zfill(6) + crop_option)
        
    if args.contour==0: filename += '_towel_pixel_subsample' + str(args.subsample_size)
    elif args.contour==1: filename += '_towel_pixel_contour_subsample' + str(args.subsample_size_contour)
        
    if args.GTtowelPixel==1: filename += '_fromGT'
        
    if args.save_tensor==1: filename += '.pt'
    elif args.save_tensor==0: filename += '.txt'
    return filename

def towelPixelSubsample(self, img_name, sample, group_number=None, animation_frame=None, idx=None):
    """ Output format: sample['towel_pixel_subsample'] and sample['towel_pixel_contour_subsample'] are
    numpy arrays of shape (M, 2) and (N, 2), for some M and N."""
    if self.args.towelPixelSample==1:
        self.args.contour=0
        sample['towel_pixel_subsample'] = functions_train.sample_of_towel_pixels(img_name, self.args) 
        self.args.contour=1
        sample['towel_pixel_contour_subsample'] = functions_train.sample_of_towel_pixels(img_name, self.args) 
    elif self.args.towelPixelSample==2:
        # Load as numpy array
        # The dataloader takes numpy arrays and stacks the ones corresponding to a whole batch together as a tensor.
        # If you try to load a tensor directly, it gives problems
        if self.args.loss_w_chamfer_pred_GT!= 0 or self.args.loss_w_chamfer_GT_pred!=0:
            self.args.contour=0
            if self.args.GTtowelPixel==1:
                sample['towel_pixel_subsample']=sample['uv']
            else:
                filename = towel_pixel_subsample_filename(
                    self.args, self.root_dir, self.crop_option, int(idx), group_number, animation_frame)
                    # int(idx) is needed so that, in the dataloders, idx is not used as a tensor of a single element, but an element
                sample['towel_pixel_subsample'] = np.genfromtxt(filename) #.astype(np.float32)
            
        if self.args.loss_w_chamfer_GTcontour_pred!=0 or self.args.loss_w_chamfer_pred_GTcontour!=0:
            self.args.contour=1
            if self.args.GTtowelPixel==1:
                sample['towel_pixel_contour_subsample'] = sample['uv'][submesh.contour_of_mesh(self.args)]
            else:
                filename = towel_pixel_subsample_filename(
                    self.args, self.root_dir, self.crop_option, int(idx), group_number, animation_frame)
                # int(idx) is needed so that, in the dataloders, idx is not used as a tensor of a single element, but an element
                sample['towel_pixel_contour_subsample'] = np.genfromtxt(filename) #.astype(np.float32)
            
#         if self.args.unsupervised==1:
#             sample['uv_towel'] = binary_image.find_uv_from_img_name(
#                 input_png_name=img_name, verbose=self.args.verbose, numpy_or_tensor=1).to(self.args.device).double()
#             print(sample['uv_towel'].shape) 
#             # Since different RGBs give different number of towel pixels,
#             # a batch of the dataloader would try to vertically concatenate 
#             # tensors of different shapes.
#             # Hence, there are 2 options:
#             # 1. Do not compute it within the dataset class. 
#             #    This even allows you to use a random different subsample ratio at every epoch.
#             #    This can be a good option to avoid the vertices going too close to some specific pixels.
#             # 2. Fix a subsample size (rather than a subsample ratio) before creating the dataset and 
#             #    save that subsample in sample['uv_towel'].
    return sample

###
### Dataset classes
###
class vertices_Dataset(Dataset):
    """If we want to predict xyz coordinates, i.e., if predict_uv_or_xyz=='xyz':
        For each idx, vertices_Dataset[idx] produces a sample, which is a dictionary
        {'image': image, 'xyz': xyz, 
                      'img_name': img_name, 'normal_coordinates': normal_coordinates},
        where
        - 'image' is a numpy array (*) with shape HxWxC representing the input (2D RGB) image. 
        - 'img_name' contains the image name (path from current directory).
        - 'xyz' is a numpy array of shape num_vertices x 3 consisting of the 
        3D (world or camera) coordinates of the first 'num_vertices' vertices of the 3D reconstruction
        - 'normal_coordinates' is a numpy array of shape num_vertices x 3 consisting of the 
        3D (world or camera) coordinates of the normal vectors at the first 'num_vertices' vertices of the 3D reconstruction.

        (*) If a transform is given when instanciating the class, then 'image' is a torch.Tensor
        representing the input (2D RGB) image, where the channels are CxHxW.
    
    If we want to predict uv coordinates, i.e., if predict_uv_or_xyz=='uv':
        then the dictionary is
        sample = {'image': image, 'uv': uv, 'img_name': img_name},
        where  
        - 'uv' is a numpy array of shape num_vertices x 2 consisting of the 
        2D pixel location coordinates of the first 'num_vertices' vertices of the input 2D RGB image              
        
    0<=x,y,z<=1 normalization coming from the metadata from the training data normalize_xyz_min, normalize_xyz_max 
    is performed if these optional values are not None.
    
    0<=u,v<=1 normalization is performed only if uv_normalization==1
    """

    def __init__(self, args, mean_for_normalization = [0.485, 0.456, 0.406],
                 std_for_normalization = [0.229, 0.224, 0.225], num_normalizing_landmarks=None, 
                 normalize_xyz_min=None, normalize_xyz_max=None, normalize_D_min=None, normalize_D_max=None):           
        """
        Args:
            - args.sequence_name (string): At the moment, it can only take the value 'TowelWall'
            - args.dataset_number (string): It is the number which follows 'TowelWall':
                for instance, dataset 4 consists of a tiny subsample of dataset 3 I use to train the network to check that it overfits the training set for diagnosing what may be wrong with the net.
            - args.transform (callable, optional): Optional transform to be applied on a sample.
            - mean_for_normalization is the mean for normalization neccessary for using ResNet.
            - std_for_normalization is the mean for normalization neccessary for using ResNet.
            - args.camera_coordinates == 1 if we want to use camera coordinates. 
              args.camera_coordinates == 0 if we want to use world coordinates. 
            - centre/ROI (cropping option). 
              args.crop_centre_or_ROI==0: centre crop; 
              args.crop_centre_or_ROI==1: Region of interest. Bounding box containing the towel.
              args.crop_centre_or_ROI==2: Region of interest. Rectangular box containing the towel.
              args.crop_centre_or_ROI==3: no crop.
            - If args.reordered_dataset == 0, to create the dataset, 
              only the first args.num_selected_vertices vertices (in the order provided by Blender) will be used
              from the 3D reconstruction.
              If args.reordered_dataset == 1, to create the dataset, 
              only args.submesh_num_vertices_vertical * args.submesh_num_vertices_horizontal vertices will be used
              from the 3D reconstruction.
            - args.num_selected_vertices (it can be used if args.reordered_dataset == 0):
              number of the first vertices (in the order provided by Blender) to use 
              from the 3D reconstruction to create the dataset.
            - num_normalizing_landmarks (it can be used if reordered_dataset == 0).
              If num_normalizing_landmarks == None, the normalization of the first args.num_selected_vertices vertices 
              in xyz is computed using the first args.num_selected_vertices vertices in xyz.
              Otherwise, the first num_normalizing_landmarks vertices of xyz are used to normalize 
              the first args.num_selected_vertices vertices in xyz. 
              This is done so that we can plot 6 points and the whole mesh in the same figure and 
              having the 6 points correspond to the exact point they belong to within the mesh. 
              To do that, we need to normalize the whole mesh using only the 6 points.
            - args.submesh_num_vertices_vertical (it can be used if args.reordered_dataset == 1): 
              number of vertices in the vertical direction which want to be selected.
              The vertices will be chosen so that the distance between any consecutive vertices is the same.
            - args.submesh_num_vertices_horizontal (it can be used if args.reordered_dataset == 1): 
              number of vertices in the horizontal direction which want to be selected.
              The vertices will be chosen so that the distance between any consecutive vertices is the same.
            - normalize_xyz_min = [x_min, y_min, z_min], 
                where x_min corresponds to the minimum of the X coordinate 
                of all the vertices within the training set. Same with Y and Z.
            - normalize_xyz_max = [x_max, y_max, z_max],
                same as before, with max instead of min.
            - normalize_D_min corresponds to the minimum Depth of all the vertices within the training set.
            - normalize_D_max corresponds to the maximum Depth of all the vertices within the training set.
        """
        self.root_dir = root_dir_from_dataset(args.sequence_name, args.dataset_number, args.texture_type)
        self.args = args
        self.mean_for_normalization = mean_for_normalization
        self.std_for_normalization = std_for_normalization
        if args.reordered_dataset == 0:
            if num_normalizing_landmarks == None:
                self.num_normalizing_landmarks = args.num_selected_vertices
            else:
                self.num_normalizing_landmarks = num_normalizing_landmarks
        elif args.reordered_dataset == 1:
            self.submesh_idx = submesh.submesh_idx_from_num_vertices_in_each_direction(
                submesh_num_vertices_vertical = args.submesh_num_vertices_vertical,
                submesh_num_vertices_horizontal = args.submesh_num_vertices_horizontal)
            if num_normalizing_landmarks == None:
                self.num_normalizing_landmarks = args.submesh_num_vertices_vertical * args.submesh_num_vertices_horizontal
            else:
                self.num_normalizing_landmarks = num_normalizing_landmarks
                
        self.normalize_xyz_min, self.normalize_xyz_max = normalize_xyz_min, normalize_xyz_max
        self.normalize_D_min, self.normalize_D_max = normalize_D_min, normalize_D_max

    def __len__(self):
        return dataset_length_from_name(self.args)

    def __getitem__(self, idx):
        if self.args.sequence_name == 'kinect_tshirt':
            # Since the examples in 'kinect_tshirt' are 1-indexed and examples 1 and 2 are not even present, 
            # we want the example in the file with idx name 3 to be indexed in the dataset class as idx 0 and so on.
            idx+=3
        elif self.args.sequence_name == 'kinect_paper':
            # Since the examples in 'kinect_paper' are 1-indexed and example 1 is not even present, 
            # we want the example in the file with idx name 2 to be indexed in the dataset class as idx 0 and so on.
            idx+=2
        
        # Load image
        img_name_end, self.crop_option = imgNameEnd_from_cropOption(self.args.crop_centre_or_ROI)
        img_name = imgName_from_idx(self.root_dir, self.args, img_name_end, idx)
        image = im_from_imgName(img_name=img_name, transform=self.args.transform)
        
        if self.args.sequence_name == 'TowelWall':
            # Load vertex and normal vector coordinates
            group_number, animation_frame = group_and_frame_from_idx(
                idx, num_groups=self.args.num_groups, num_animationFramesPerGroup = self.args.num_animationFramesPerGroup)
            group_number, animation_frame = post_process_grp_and_frame(group_number, animation_frame)
            sample = load_coordinates_RendersTowelWall(self, group_number, animation_frame)
                    
            # Postprocessing
            if self.args.camera_coordinates==1 and (self.args.predict_uv_or_xyz=='xyz' or self.args.predict_uv_or_xyz=='uvy'): 
                sample['xyz'] = functions_data_processing.world_to_camera(sample['xyz'], self.args.RT_extended, self.args.dtype)
                if self.args.w_normals!=0:
                    sample['normal_coordinates'] = functions_data_processing.normals_in_camera_coordiantes(sample['normal_coordinates'], self.args.RT_matrix)                    
        elif self.args.sequence_name == 'DeepCloth':
            # Load vertex coordinates
            sample = load_coordinates_DeepCloth2(root_dir=self.root_dir, idx=idx, dtype=self.args.dtype, texture_type=self.args.texture_type)
            group_number, animation_frame = None, None
        elif self.args.sequence_name in ['kinect_tshirt', 'kinect_paper']:
            # Load vertex coordinates
            sample = load_coordinates_realDatasets(root_dir=self.root_dir, idx=idx, dtype=self.args.dtype)
            group_number, animation_frame = None, None
        
        # Subsample of towel pixels and contour
        sample = towelPixelSubsample(self, img_name, sample, group_number, animation_frame, idx)

        # Process uv, xyz, depth
        sample = functions_data_processing.process_uv(sample, img_name, self.root_dir, self.args, group_number, animation_frame)
            # This processes the GT uv, and the towel pixels and contours computed in towelPixelSubsample
        sample = functions_data_processing.process_xyz(sample, self.normalize_xyz_min, self.normalize_xyz_max)
        sample = functions_data_processing.process_D(sample, self.normalize_D_min, self.normalize_D_max)

        # Append image info to the sample
        sample['image'], sample['img_name'] = image, img_name

        return sample

class vertices_Dataset_barycenter(Dataset):
    """Dataset with the barycenter normalization which depends on each particular observation. This one is wrong, but I keep it to allow for prediction visualization of the previous approaches.
    
    For each idx, vertices_Dataset[idx] produces a sample, which is a dictionary
    {'image': image, 'xyz': xyz, 
                  'img_name': img_name, 'normal_coordinates': normal_coordinates},
    where
    - 'image' is a numpy array (*) with shape HxWxC representing the input (2D RGB) image. 
    - 'img_name' contains the image name (path from current directory).
    - 'xyz' is a numpy array of shape num_vertices * 3 consisting of the 
    3D (world or camera) coordinates of the first 'num_vertices' vertices of the 3D reconstruction,
    where the barycentre of the vertices has been moved to the origin,
    and the vertices coordinates have been rescaled so that:
        1. the vertices sit inside the closed unit ball centred at the origin, and 
        2. the vertex which is furthest away from the centre lies on the boundary of the ball.
    - 'normal_coordinates' is a numpy array of shape num_vertices * 3 consisting of the 
    3D (world or camera) coordinates of the normal vectors at the first 'num_vertices' vertices of the 3D reconstruction.
    
    (*) If a transform is given when instanciating the class, then 'image' is a torch.Tensor
    representing the input (2D RGB) image, where the channels are CxHxW."""

    def __init__(self, num_vertices, sequence_name = 'TowelWall', dataset_number = '2', 
                 transform=None, 
                 mean_for_normalization = [0.485, 0.456, 0.406],
                 std_for_normalization = [0.229, 0.224, 0.225], 
                 camera_coordinates = 1, crop_centre_or_ROI=0, 
                 reordered_dataset = 0, 
                 num_normalizing_landmarks=None, 
                 submesh_num_vertices_vertical = 2, 
                 submesh_num_vertices_horizontal = 3):           
        """
        Args:
            - sequence_name (string): At the moment, it can only take the value 'TowelWall'
            - dataset_number (string): It is the number which follows 'TowelWall':
                for instance, dataset 4 consists of a tiny subsample of dataset 3 I use to train the network to check that it overfits the training set for diagnosing what may be wrong with the net.
            - transform (callable, optional): Optional transform to be applied on a sample.
            - mean_for_normalization is the mean for normalization neccessary for using ResNet.
            - std_for_normalization is the mean for normalization neccessary for using ResNet.
            - camera_coordinates == 1 if we want to use camera coordinates. 
              camera_coordinates == 0 if we want to use world coordinates. 
            - centre/ROI (cropping option). 
              crop_centre_or_ROI==0: centre crop; 
              crop_centre_or_ROI==1: Region of interest. Bounding box containing the towel.
              crop_centre_or_ROI==2: Region of interest. Rectangular box containing the towel.
              crop_centre_or_ROI==3: no crop.
            - If reordered_dataset == 0, to create the dataset, 
              only the first 'num_vertices' vertices (in the order provided by Blender) will be used
              from the 3D reconstruction.
              If reordered_dataset == 1, to create the dataset, 
              only submesh_num_vertices_vertical x submesh_num_vertices_horizontal vertices will be used
              from the 3D reconstruction.
            - num_vertices (it can be used if reordered_dataset == 0):
              number of the first vertices (in the order provided by Blender) to use 
              from the 3D reconstruction to create the dataset.
            - num_normalizing_landmarks (it can be used if reordered_dataset == 0).
              If num_normalizing_landmarks == None, the normalization of the first 'num_vertices' vertices in xyz is computed using the first 'num_vertices' vertices in xyz.
              Otherwise, the first num_normalizing_landmarks vertices of xyz are used to normalize the first 'num_vertices' vertices in xyz. This is done so that we can plot 6 points and the whole mesh in the same figure and having the 6 points correspond to the exact point they belong to within the mesh. To do that, we need to normalize the whole mesh using only the 6 points.
            - submesh_num_vertices_vertical (it can be used if reordered_dataset == 1): 
              number of vertices in the vertical direction which want to be selected.
              The vertices will be chosen so that the distance between any consecutive vertices is the same.
            - submesh_num_vertices_horizontal (it can be used if reordered_dataset == 1): 
              number of vertices in the horizontal direction which want to be selected.
              The vertices will be chosen so that the distance between any consecutive vertices is the same.
        """
        self.root_dir = 'Renders' + sequence_name + dataset_number + '/Group.'
        if dataset_number in ['2','3','7']:
            self.num_groups = 40
            self.num_animationFramesPerGroup = 99
        elif dataset_number=='4':
            self.num_groups = 4
            self.num_animationFramesPerGroup = 12
        elif dataset_number in ['5', '6', '8']:
            self.num_groups = 5
            self.num_animationFramesPerGroup = 4
        elif dataset_number=='10':
            self.num_groups = 40
            self.num_animationFramesPerGroup = 10
        self.transform = transform
        self.mean_for_normalization = mean_for_normalization
        self.std_for_normalization = std_for_normalization
        self.camera_coordinates = camera_coordinates
        self.crop_centre_or_ROI = crop_centre_or_ROI
        self.reordered_dataset = reordered_dataset
        if reordered_dataset == 0:
            self.num_vertices = num_vertices
            if num_normalizing_landmarks == None:
                self.num_normalizing_landmarks = num_vertices
            else:
                self.num_normalizing_landmarks = num_normalizing_landmarks
        elif reordered_dataset == 1:
            self.submesh_idx = submesh.submesh_idx_from_num_vertices_in_each_direction(
                submesh_num_vertices_vertical = submesh_num_vertices_vertical,
                submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
            if num_normalizing_landmarks == None:
                self.num_normalizing_landmarks = submesh_num_vertices_vertical * submesh_num_vertices_horizontal
            else:
                self.num_normalizing_landmarks = num_normalizing_landmarks
        
        self.RT_matrix = np.genfromtxt('Renders' + sequence_name + dataset_number + '/camera_params.txt', 
                                  delimiter=' ', skip_header=11)
        # RT extended with zeros and 1 below (for homogeneous coordinates)
        zeros_and_1 = np.zeros(4)
        zeros_and_1[-1] = 1
        zeros_and_1 = np.reshape(zeros_and_1, (1,4))
        self.RT_extended = np.concatenate((self.RT_matrix, zeros_and_1), axis=0)

    def __len__(self):
        return self.num_groups * self.num_animationFramesPerGroup

    def __getitem__(self, idx):
        group_number, animation_frame = group_and_frame_from_idx(idx, num_groups=self.num_groups,
                                                                 num_animationFramesPerGroup = self.num_animationFramesPerGroup)
        group_number = str(group_number)
        group_number = group_number.zfill(3) # pad with zeros on the left until the string has length 3
        animation_frame = str(int(animation_frame)) 
        
        # Load image
        if self.crop_centre_or_ROI in [0, 3]: # centre crop or no crop
            img_name = self.root_dir + group_number + '/' + animation_frame + '.png'
        elif self.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
            img_name = self.root_dir + group_number + '/' + animation_frame + '_ROI.png'
        elif self.crop_centre_or_ROI==2: # Region of interest. Rectangular box containing the towel
            img_name = self.root_dir + group_number + '/' + animation_frame + '_rectROI.png'
            
        if self.transform:          
            im = Image.open(img_name)   
#             print('type of im:', type(im)) # type of im: <class 'PIL.PngImagePlugin.PngImageFile'>
            transformed_image = self.transform(im)  
            # If the tensor represents an RGBA image, turn it to RGB
#             print('type and shape of image:', type(transformed_image), transformed_image.shape)
    #         if self.crop_centre_or_ROI==0: # centre crop
    #             assert(transformed_image.shape[0] == 4)
    #         elif self.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
    #             assert(transformed_image.shape[0] == 3)
            if transformed_image.shape[0] == 4: transformed_image = RGBA_to_RGB_tensors(transformed_image)
            
            # To use 'torchvision.utils.make_grid', use:
            image = transformed_image
            # To use any other way of plotting, use:
#             image = tensor_to_plot(transformed_image, 
#                                                   self.mean_for_normalization, self.std_for_normalization)
        else:
            image = io.imread(img_name)
#             print('type and shape of image:', type(image), image.shape)
    #         print('Number of channels of image (3-->RGB, 4-->RGB+alpha transparency level)', image.shape[2])
    #         if self.crop_centre_or_ROI==0: # centre crop
    #             assert(image.shape[2] == 4)
    #         elif self.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
    #             assert(image.shape[2] == 3)

            # If the image is RGBA, turn it to RGB
            if image.shape[2] == 4: image = RGBA_to_RGB(image)
        
        # Load vertex and normal vector coordinates
        if self.reordered_dataset == 0:
            filename = self.root_dir + group_number + '/vertices_' + animation_frame.zfill(5) + '.txt'
            f = open(filename, 'r')
            line1 = f.readline()
            df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())
            xyz = df_vertices_all_data[['x', 'y', 'z']].values[range(self.num_vertices)] # Extract the x, y, z world coordinates of the first 'num_vertices' vertices as a numpy array
            normal_coordinates = df_vertices_all_data[['nx', 'ny', 'nz']].values[range(self.num_vertices)] # Extract the nx, ny, nz world coordinates of the normal vectors of the first 'num_vertices' vertices as a numpy array
        if self.reordered_dataset == 1:
            filename = self.root_dir + group_number + '/vertices_' + animation_frame.zfill(5) + '_reordered.txt'
            f = open(filename, 'r')
            line1 = f.readline()
            df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.split())
            df_vertices_all_data = df_vertices_all_data.ix[self.submesh_idx] # Select the submesh only
            xyz = df_vertices_all_data[['x', 'y', 'z']].values # Extract the x, y, z world coordinates of the submesh as a numpy array
            normal_coordinates = df_vertices_all_data[['nx', 'ny', 'nz']].values # Extract the nx, ny, nz world coordinates of the normal vectors of the submesh as a numpy array
        
        if self.camera_coordinates==1:
            # Convert vertex and normal vector coordinates from world coordinates to camera coordinates
            (X_camera, Y_camera, Z_camera) = functions_data_processing.world_to_camera_coordinates(xyz[:,0], 
                                                                         xyz[:,1],
                                                                         xyz[:,2],
                                                                         self.RT_extended)
            X_camera_col = np.reshape(X_camera, (X_camera.size, 1))
            Y_camera_col = np.reshape(Y_camera, (Y_camera.size, 1))
            Z_camera_col = np.reshape(Z_camera, (Z_camera.size, 1))
            xyz = np.hstack((X_camera_col, Y_camera_col, Z_camera_col))
            
            (nX_camera, nY_camera, nZ_camera) = functions_data_processing.world_to_camera_coordinates_normals(
                normal_coordinates[:,0],
                normal_coordinates[:,1],
                normal_coordinates[:,2],
                self.RT_matrix)
            nX_camera_col = np.reshape(nX_camera, (nX_camera.size, 1))
            nY_camera_col = np.reshape(nY_camera, (nY_camera.size, 1))
            nZ_camera_col = np.reshape(nZ_camera, (nZ_camera.size, 1))
            normal_coordinates = np.hstack((nX_camera_col, nY_camera_col, nZ_camera_col))
        
        # Normalize vertex coordinates
        xyz = move_barycentre_to_origin(xyz = xyz,
                                                      num_normalizing_landmarks=self.num_normalizing_landmarks)
        xyz = fit_tight_inside_ball(xyz = xyz,
                                                   num_normalizing_landmarks=self.num_normalizing_landmarks)
        
        sample = {'image': image, 'xyz': xyz, 
                  'img_name': img_name, 'normal_coordinates': normal_coordinates}

        return sample



###
### Transforms
###
# CAVEAT: Since our images have high resolution and many background pixels, some random crops don't even show the towel
# See data_loading-visualization.py for details





###
### CAVEAT: Permutation of axes by 'torchvision.transforms.ToTensor' 
###
# 'torchvision.transforms.ToTensor' <br>
# converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a 
# torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
# 
# To convert it back to a tensor of shape (H x W x C) in the range [0, 255], and to remove the extra alpha channel we are carrying with out RGBA pictures, we create the following functions:
def unnormalize_RGB_of_HWC_tensor(normalized_tensor, mean_for_normalization = [0.485, 0.456, 0.406], 
                 std_for_normalization = [0.229, 0.224, 0.225]):
    unnormalized_tensor = torch.tensor(data = normalized_tensor, dtype = normalized_tensor.dtype)
    for i in range(3):
        unnormalized_tensor[:,:,i] = normalized_tensor[:,:,i]*std_for_normalization[i] + mean_for_normalization[i]
    return unnormalized_tensor

def unnormalize_RGB_of_CHW_tensor(normalized_tensor, mean_for_normalization = [0.485, 0.456, 0.406], 
                 std_for_normalization = [0.229, 0.224, 0.225]):
    unnormalized_tensor = torch.tensor(data = normalized_tensor, dtype = normalized_tensor.dtype)
    for i in range(3):
        unnormalized_tensor[i,:,:] = normalized_tensor[i,:,:]*std_for_normalization[i] + mean_for_normalization[i]
    return unnormalized_tensor

def tensor_to_plot(tensor_image_CHW_RGBA, mean_for_normalization = [0.485, 0.456, 0.406], 
                 std_for_normalization = [0.229, 0.224, 0.225]):
    """
    Input: a torch.FloatTensor of shape (C x H x W), where C consists of RGBA.
    Output: a torch.FloatTensor of shape (H x W x C'), where C' consists of RGB, 
            where the values have been unnormalized using the given means and standard deviations.
    """
    # Remove the alpha channel from RGBA and do nothing if it is already RGB
    if tensor_image_CHW_RGBA.shape[0] == 4:
        tensor_image_HWC_RGB = RGBA_to_RGB_tensors(tensor_image_CHW_RGBA)
    
    # Permute the axes from (C' x H x W) to (H x W x C') 
    tensor_image_HWC_RGB = tensor_image_HWC_RGB.permute(1, 2, 0)
    
    # Unnormalize the values
    tensor_image_HWC_RGB = unnormalize_RGB_of_HWC_tensor(tensor_image_HWC_RGB)
    
    return tensor_image_HWC_RGB





###
### CAVEAT: Permutation of axes by 'torchvision.transforms.ToTensor'. Part II 
###
# Notice, though, that 'torchvision.utils.make_grid' and some functions for CNNs take as input torch.FloatTensor of shape (C x H x W).


    
    
    


    
    


    
###
### Random split of the dataset
###
# Randomly choosing some sequences for training and some for test, and randomly reordering each of the resulting two subdatasets
def random_split_notMixingSequences(dataset, args):
    """
    Randomly split a dataset into two non-overlapping new datasets of given lengths,
    so that the sequences in the training and test sets are disjoint.
    I.e., if a frame from sequence i is in the test set, 
    then no frame from sequence i can be in the training set.

    Arguments:
        dataset (Dataset): Dataset to be split
        args.lengths_proportion_train (float between 0 and 1): percentatge of the dataset to be set for training  
        args.lengths_proportion_test (float between 0 and 1 or None): 
            If it is None, the split is only made into train and validation as follows.
                args.lengths_proportion_train for training, 
                and the rest for validation.
            If it is not None, then the split is done with:
                args.lengths_proportion_train for training, 
                args.lengths_proportion_test for test 
                and the rest for validation.
        args.lengths_proportion_train_nonAnnotated (float between 0 and 1, default: None): 
            When there is a part of the training set for which annotations are used 
            and a part for which the annotations are not used, 
            lengths_proportion_train_nonAnnotated is the percentatge of the training set for which no annotations will be used.
    """
    # Set proportions
    if args.lengths_proportion_train_nonAnnotated is not None:
        lengths_proportion_train_nonAnnot = args.lengths_proportion_train * args.lengths_proportion_train_nonAnnotated
        lengths_proportion_train_default = args.lengths_proportion_train - lengths_proportion_train_nonAnnot
    else:
        lengths_proportion_train_nonAnnot = 0
        lengths_proportion_train_default = args.lengths_proportion_train
    
    # Set number of groups
    num_groups_train_nonAnnot = round(lengths_proportion_train_nonAnnot * args.num_groups)
    num_groups_train_default = round(lengths_proportion_train_default * args.num_groups)
    num_groups_train_total = num_groups_train_default + num_groups_train_nonAnnot
    num_groups_test = 0 if args.lengths_proportion_test is None else round(args.lengths_proportion_test * args.num_groups)
    num_groups_val = args.num_groups - num_groups_train_default - num_groups_train_nonAnnot - num_groups_test

    if args.verbose==1:
        if args.lengths_proportion_train_nonAnnotated is not None:
            print('Number of annotated training set sequences:', num_groups_train_default)
            print('Number of non-annotated training set sequences:', num_groups_train_nonAnnot)
        else: 
            print('Number of training set sequences:', num_groups_train_default)
        print('Number of validation set sequences:', num_groups_val)
        print('Number of test set sequences:', num_groups_test)
        print()

    # Shuffle video sequences
    torch.manual_seed(args.random_seed_to_choose_video_sequences) # seeded for reproducibility
    torch.cuda.manual_seed(args.random_seed_to_choose_video_sequences) # seeded for reproducibility
    random_sequences = torch.randperm(args.num_groups) # random permutation of integers from 0 to num_groups - 1
    # The names of the sequences are Group.001, Group.002, ..., Group.040.
    # Hence, we are going to have sequences indices starting from 1:
    random_sequences = random_sequences + 1 # random permutation of integers from 1 to num_groups
    
    # Assign to each subset a few video sequences
    sequences_train = random_sequences[range(num_groups_train_default)] # Id of training set sequences
    sequences_train_nonAnnot = random_sequences[num_groups_train_default: num_groups_train_default + num_groups_train_nonAnnot] 
    sequences_test = random_sequences[num_groups_train_total: num_groups_train_total + num_groups_test] 
    sequences_val = random_sequences[num_groups_train_total + num_groups_test:] 
    if args.verbose==1:
        if args.lengths_proportion_train_nonAnnotated is not None:
            print('Annotated training sequences:', sequences_train.data.tolist())
            print('Non-annotated training sequences:', sequences_train_nonAnnot.data.tolist())
        else:
            print('Training sequences:', sequences_train.data.tolist())
        print('Validation sequences:', sequences_val.data.tolist())
        print('Test sequences:', sequences_test.data.tolist())
        print()

    # Shuffle indices (animation frames) within each subset
    np.random.seed(args.random_seed_to_shuffle_training_frames) # seeded for reproducibility
    list_of_ranges_train = [
        range_of_idxs_from_group(
            group, num_groups = args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup) 
        for group in sequences_train]
    idx_train = list(set(itertools.chain(*list_of_ranges_train))) # This orders the indices increasingly
    idx_train_random_order = np.random.permutation(idx_train)
    
    list_of_ranges_train_nonAnnot = [
        range_of_idxs_from_group(
            group, num_groups = args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup) 
        for group in sequences_train_nonAnnot]
    idx_train_nonAnnot = list(set(itertools.chain(*list_of_ranges_train_nonAnnot))) 
    idx_train_nonAnnot_random_order = np.random.permutation(idx_train_nonAnnot)

    list_of_ranges_val = [
        range_of_idxs_from_group(
            group, num_groups = args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup) 
        for group in sequences_val]
    idx_val = list(set(itertools.chain(*list_of_ranges_val))) 
    idx_val_random_order = np.random.permutation(idx_val)
    
    list_of_ranges_test = [range_of_idxs_from_group(group, num_groups = args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup) for group in sequences_test]
    idx_test = list(set(itertools.chain(*list_of_ranges_test))) 
    idx_test_random_order = np.random.permutation(idx_test)
    
    # Check the index sets for the training, non-annotated training, validation and test sets are disjoint
    llista=[idx_train_random_order, idx_train_nonAnnot_random_order, idx_val_random_order, idx_test_random_order]
    for i in range(len(llista)):
        for j in range(i+1, len(llista)):
                assert(set(llista[i]).intersection(set(llista[j])) == set()) 

    # Check the union of the index sets for the training, validation and test sets recover the whole index set 
    idx_union = set().union(set(idx_train_random_order), set(idx_train_nonAnnot_random_order), set(idx_val_random_order), set(idx_test_random_order))
    assert(len(idx_union) == args.num_groups * args.num_animationFramesPerGroup)
    assert(min(idx_union) == 0)
    assert(max(idx_union) == args.num_groups * args.num_animationFramesPerGroup-1)

    transformed_dataset_parts= [torch.utils.data.dataset.Subset(dataset, indices) for indices in [idx_train_random_order, idx_val_random_order, idx_test_random_order, idx_train_nonAnnot_random_order]]
        
    dataset_sizes = {'train': len(transformed_dataset_parts[0]),
                     'val': len(transformed_dataset_parts[1]),
                    'test': len(transformed_dataset_parts[2])}       
    dataloaders = {'train': DataLoader(transformed_dataset_parts[0], batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, drop_last=True),
                   'val': DataLoader(transformed_dataset_parts[1], batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, drop_last=True),
                  'test': DataLoader(transformed_dataset_parts[2], batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, drop_last=True)}
    if args.lengths_proportion_train_nonAnnotated is not None:
        dataset_sizes['train_nonAnnot'] = len(transformed_dataset_parts[3])
        dataloaders['train_nonAnnot'] = DataLoader(transformed_dataset_parts[3], batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, drop_last=True)

    return transformed_dataset_parts, dataset_sizes, dataloaders
    
def random_split_DeepCloth(dataset, args):
    """
    Randomly split a dataset whose args.sequence_name is 'DeepCloth'.
    
    Arguments:
        dataset (Dataset): Dataset to be split
        args.lengths_proportion_train (float between 0 and 1): percentatge of the dataset to be set for training  
        args.lengths_proportion_test (float between 0 and 1 or None): 
            If it is None, the split is only made into train and validation as follows.
                args.lengths_proportion_train for training, 
                and the rest for validation.
            If it is not None, then the split is done with:
                args.lengths_proportion_train for training, 
                args.lengths_proportion_test for test 
                and the rest for validation.
        args.lengths_proportion_train_nonAnnotated (float between 0 and 1, default: None): 
            When there is a part of the training set for which annotations are used 
            and a part for which the annotations are not used, 
            lengths_proportion_train_nonAnnotated is the percentatge of the training set for which no annotations will be used.
    """    
    # Set proportions
    num_examples = len(dataset)
    if args.lengths_proportion_train_nonAnnotated is not None:
        lengths_proportion_train_nonAnnot = args.lengths_proportion_train * args.lengths_proportion_train_nonAnnotated
        lengths_proportion_train_default = args.lengths_proportion_train - lengths_proportion_train_nonAnnot
    else:
        lengths_proportion_train_nonAnnot = 0
        lengths_proportion_train_default = args.lengths_proportion_train
    
    # Set number of examples
    num_examples_train_nonAnnot = round(lengths_proportion_train_nonAnnot * num_examples)
    num_examples_train_default = round(lengths_proportion_train_default * num_examples)
    num_examples_train_total = num_examples_train_default + num_examples_train_nonAnnot
    num_examples_test = 0 if args.lengths_proportion_test is None else round(args.lengths_proportion_test * num_examples)
    num_examples_val = num_examples - num_examples_train_default - num_examples_train_nonAnnot - num_examples_test

    if args.verbose==1:
        if args.lengths_proportion_train_nonAnnotated is not None:
            print('Number of annotated training set examples:', num_examples_train_default)
            print('Number of non-annotated training set examples:', num_examples_train_nonAnnot)
        else: 
            print('Number of training set examples:', num_examples_train_default)
        print('Number of validation set examples:', num_examples_val)
        print('Number of test set examples:', num_examples_test)
        print()

    # Shuffle examples
    torch.manual_seed(args.random_seed_to_choose_video_sequences) # seeded for reproducibility
    torch.cuda.manual_seed(args.random_seed_to_choose_video_sequences) # seeded for reproducibility
    random_sequences = torch.randperm(num_examples) # random permutation of integers from 0 to num_examples - 1
    
    # Assign to each subset a few examples
    examples_train = random_sequences[range(num_examples_train_default)] # Id of training set sequences
    examples_train_nonAnnot = random_sequences[num_examples_train_default: num_examples_train_default + num_examples_train_nonAnnot] 
    examples_test = random_sequences[num_examples_train_total: num_examples_train_total + num_examples_test] 
    examples_val = random_sequences[num_examples_train_total + num_examples_test:] 
  
    # Check the index sets for the training, non-annotated training, validation and test sets are disjoint
    llista=[examples_train, examples_train_nonAnnot, examples_val, examples_test]
    for i in range(len(llista)):
        for j in range(i+1, len(llista)):
                assert(set(llista[i]).intersection(set(llista[j])) == set()) 

    # Check the union of the index sets for the training, validation and test sets recover the whole index set 
    examples_union = set().union(set(examples_train), set(examples_train_nonAnnot), set(examples_val), set(examples_test))
    assert(len(examples_union) == num_examples)
    assert(min(examples_union) == 0)
    assert(max(examples_union) == num_examples-1)

    transformed_dataset_parts= [torch.utils.data.dataset.Subset(dataset, indices) for indices in [examples_train, examples_val, examples_test, examples_train_nonAnnot]]
        
    dataset_sizes = {'train': len(transformed_dataset_parts[0]),
                     'val': len(transformed_dataset_parts[1]),
                    'test': len(transformed_dataset_parts[2])}       
    dataloaders = {'train': DataLoader(transformed_dataset_parts[0], batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, drop_last=True),
                   'val': DataLoader(transformed_dataset_parts[1], batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, drop_last=True),
                  'test': DataLoader(transformed_dataset_parts[2], batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, drop_last=True)}
    if args.lengths_proportion_train_nonAnnotated is not None:
        dataset_sizes['train_nonAnnot'] = len(transformed_dataset_parts[3])
        dataloaders['train_nonAnnot'] = DataLoader(transformed_dataset_parts[3], batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, drop_last=True)

    return transformed_dataset_parts, dataset_sizes, dataloaders
    

    
def instanciate_dataset(args):
    if args.normalization ==1:
        transformed_dataset = vertices_Dataset_barycenter(
            sequence_name = args.sequence_name,             dataset_number = args.dataset_number,
            transform=args.transform,  camera_coordinates=args.camera_coordinates,
            crop_centre_or_ROI=args.crop_centre_or_ROI, reordered_dataset = args.reordered_dataset,
            num_vertices=args.num_selected_vertices, submesh_num_vertices_vertical = args.submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = args.submesh_num_vertices_horizontal)
    elif args.normalization in [2,3]:
        transformed_dataset = vertices_Dataset(args)

    if args.verbose==1:
        print('\nLength of transformed_dataset (without including data augmentation):', len(transformed_dataset))
    return transformed_dataset

def get_1st_batch(args):
    # Create dataset
    transformed_dataset = vertices_Dataset(args)

    # Create dataloaders
    transformed_dataset_parts, dataset_sizes, dataloaders = random_split_notMixingSequences( 
        dataset=transformed_dataset, args=args)

    # Pick the first training, validation or test batch
    if args.train_or_val==0:
        iterable_dataloaders = iter(dataloaders['train'])
    elif args.train_or_val==1:
        iterable_dataloaders = iter(dataloaders['val'])
    elif args.train_or_val==2:
        iterable_dataloaders = iter(dataloaders['test'])
    sample_batched = next(iterable_dataloaders)
    if args.verbose==1:
        print("sample_batched['image'].shape[0], batch_size:", sample_batched['image'].shape[0], args.batch_size)
    return sample_batched