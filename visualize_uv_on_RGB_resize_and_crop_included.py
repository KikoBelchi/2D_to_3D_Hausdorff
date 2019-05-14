# coding: utf-8

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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import data_loading
import functions_data_processing
import functions_plot

# Imports for plotting
import matplotlib.pyplot as plt # Do not use when running on the server
from mpl_toolkits.mplot3d import axes3d # Do not use when running on the server

# Allow the interactive rotation of 3D scatter plots in jupyter notebook
import sys    
import os    
file_name =  os.path.basename(sys.argv[0])
#print(file_name == 'ipykernel_launcher.py') # This basicaly asks whether this file is a jupyter notebook?
if __name__ == "__main__":
    if file_name == 'ipykernel_launcher.py': # Run only in .ipynb, not in exported .py scripts
        get_ipython().run_line_magic('matplotlib', 'notebook') # Equivalent to ''%matplotlib notebook', but it is also understood by .py scripts

args = functions_data_processing.parser()

if __name__ == '__main__':
    for uv_normalization in [1, 0]:
        args.uv_normalization=uv_normalization
        print('uv_normalization:', args.uv_normalization)

        print('Original image size:')
        functions_data_processing.get_picture_size(verbose=1)
        args.transform = transforms.Compose([transforms.Resize((540,960)), # original size
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])
                                           ])
        # Choose sequence, animation frame and order of vertices
        group_number = '001'
#         animation_frame = '00022'
        animation_frame = '00001' if args.dataset_number=='13' else '00010'
        reordered=1
        observation_within_dataset = 0 if args.dataset_number=='13' else 22
        
        args.crop_centre_or_ROI = 3 # no crop
        args.reordered_dataset = 1
        args.num_selected_vertices = args.submesh_num_vertices_vertical*args.submesh_num_vertices_horizontal
        args.predict_uv_or_xyz='uv'
        
        # Instanciate dataset class
        transformed_dataset = data_loading.vertices_Dataset(args)

        
        
        ###
        ### Plot RGB and landmarks for one sample - loading RGB from file path
        ###
        functions_plot.plot_RGB_and_landmarks_from_dataset(
            dataset = transformed_dataset, observation_within_dataset = observation_within_dataset, 
            transformed_image_or_not=0, uv_normalization=args.uv_normalization)
        plt.title('Plot RGB (from file path) and landmarks')






        ###
        ### Plot RGB and landmarks for one sample - loading RGB from transformed dataset  
        ###
        functions_plot.plot_RGB_and_landmarks_from_dataset(dataset = transformed_dataset, observation_within_dataset = observation_within_dataset, transformed_image_or_not=1, uv_normalization=args.uv_normalization)
        plt.title('Plot RGB (from transformed dataset) and landmarks')






        ###
        ### Plot RGB and landmarks on a cropped sample - loading RGB from file path
        ###
        from submesh import submesh_idx_from_num_vertices_in_each_direction
        submesh_idx = submesh_idx_from_num_vertices_in_each_direction(submesh_num_vertices_vertical = args.submesh_num_vertices_vertical,
                                                                      submesh_num_vertices_horizontal = args.submesh_num_vertices_horizontal)
#         print(submesh_idx)

        # Load camera parameters
        variables = functions_data_processing.load_camera_params(sequence_name = args.sequence_name, dataset_number = args.dataset_number)
        RT_matrix = variables['RT_matrix']
        RT_extended = variables['RT_extended']
        camera_worldCoord_x = variables['camera_worldCoord_x']
        camera_worldCoord_y = variables['camera_worldCoord_y']
        camera_worldCoord_z = variables['camera_worldCoord_z']
        Intrinsic_matrix = variables['Intrinsic_matrix']
        Camera_proj_matrix = variables['Camera_proj_matrix']

        # Load vertex data
        variables = functions_data_processing.get_variables_from_vertex_full_Dataframe(sequence_name=args.sequence_name,
                                                                                       dataset_number=args.dataset_number, 
                                                                                       group_number=group_number, 
                                                                                       animation_frame=animation_frame,
                                                                                       RT_extended=RT_extended, 
                                                                                       reordered=reordered,
                                                                                       submesh_idx=submesh_idx, verbose=1)
        u_visible = variables['u_visible']
        v_visible = variables['v_visible']
        u_occluded = variables['u_occluded']
        v_occluded = variables['v_occluded']

        # Load face data
        faces = functions_data_processing.load_faces(sequence_name = args.sequence_name, dataset_number = args.dataset_number, verbose=1, reordered = reordered)

        # Plot RGB and landmarks on a cropped sample
        directory_name = os.path.join('RendersTowelWall' + str(args.dataset_number), 'Group.' + str(group_number).zfill(3))
        text_name = os.path.join(directory_name, 'uv_width_height_rectROI_' + str(int(animation_frame)) +'.txt')
        uvwh = np.genfromtxt(fname=text_name, dtype='int', delimiter=' ', skip_header=1) 
        u_crop_corner = uvwh[0]
        v_crop_corner = uvwh[1]
        width_crop = uvwh[2]
        height_crop = uvwh[3]

        u_visible_crop = u_visible - u_crop_corner
        v_visible_crop = v_visible - v_crop_corner
        u_occluded_crop = u_occluded - u_crop_corner
        v_occluded_crop = v_occluded - v_crop_corner

        ROI_image_path = os.path.join(directory_name, str(int(animation_frame)) + '_rectROI.png')

        functions_plot.plot_RGB_and_landmarks(u_visible=u_visible_crop, v_visible=v_visible_crop, 
                                              u_occluded=u_occluded_crop, v_occluded=v_occluded_crop,
                                              image_path = ROI_image_path)
        plt.title('Plot RGB (from file path) and landmarks\ncropped image - (u,v)=(0,0) on upper-left corner')






        ###
        ### Plot RGB and landmarks on a cropped sample - loading RGB from transformed dataset
        ###
    #     Cropping option: 
    #     crop_centre_or_ROI==0: centre crop. 
    #     crop_centre_or_ROI==1: Squared box containing the towel. 
    #     crop_centre_or_ROI==2: Rectangular box containing the towel. 
    #     crop_centre_or_ROI==3: no crop. 
        args.crop_centre_or_ROI=2
        args.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                       ])
        transformed_dataset = data_loading.vertices_Dataset(args)

        functions_plot.plot_RGB_and_landmarks_from_dataset(dataset = transformed_dataset, observation_within_dataset = observation_within_dataset, transformed_image_or_not=1, uv_normalization=args.uv_normalization)
        plt.title('Plot RGB (from transformed dataset) and landmarks\ncropped image - (u,v)=(0,0) on upper-left corner')





        ###
        ### Plot RGB and landmarks on a cropped and resized sample - loading RGB from transformed dataset
        ###
    #     Cropping option: 
    #     crop_centre_or_ROI==0: centre crop. 
    #     crop_centre_or_ROI==1: Squared box containing the towel. 
    #     crop_centre_or_ROI==2: Rectangular box containing the towel. 
    #     crop_centre_or_ROI==3: no crop. 
        args.crop_centre_or_ROI=2
        args.transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                       ])
        transformed_dataset = data_loading.vertices_Dataset(args)
        functions_plot.plot_RGB_and_landmarks_from_dataset(dataset = transformed_dataset, observation_within_dataset = observation_within_dataset, transformed_image_or_not=1, uv_normalization=args.uv_normalization)
        plt.title('Plot RGB (from transformed dataset) and landmarks\nCropped and rescaled image')
        plt.show()





        ###
        ### Plot RGB and landmarks on a cropped sample - loading RGB from transformed dataset
        ### After scaling back from 224 to original crop size
        ###
        # See visualize_predictions.py