"""Save to file the uv pixel coordinates plotted on top of the RGB image for a fraction of the dataset to check whether there is a mismatch as in Datasets 2 and 3."""

# Imports general modules
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
import pandas as pd
from skimage import io
        
# Import my modules
import functions_data_processing    
from functions_data_processing import load_camera_params, get_variables_from_vertex_full_Dataframe, load_faces
import functions_plot
from functions_plot import plot_camera_and_vertices

# Argument parser
parser = argparse.ArgumentParser(description='Parser for choosing dataset and fraction of dataset plotted')
parser.add_argument('--sequence-name', type=str, default='TowelWall',
                    help="The dataset used is the one in 'Renders' + sequence_name + str(dataset_number). (default: 'TowelWall').")
parser.add_argument('--dataset-number', type=int, default=3, metavar='N',
                    help="The dataset used is the one in 'Renders' + sequence_name + str(dataset_number). (default: 3) At the moment, sequence_name must be 'TowelWall' and dataset_number can be either 2, 3, 4, 5 or 6.")
parser.add_argument('--save-every-these-many-frames', type=int, default=10, metavar='N',
                    help='Within each sequence, every --save-every-these-many-frames will be saved. (daeault: 10)')
parser.add_argument('--starting-group', type=int, default=1, metavar='N',
                    help='Group from which to start computations, mainly useful if the computations have been done for a few groups and we want to continue with the computations for the rest of them. (default: 1)')
parser.add_argument('--last-group', type=int, default=0, metavar='N',
                    help='Last group to check. If the value is 0 (default), then the last group will be the last group of the dataset')
args = parser.parse_args()

sequence_name = args.sequence_name
dataset_number = str(args.dataset_number)
save_every_these_many_frames = args.save_every_these_many_frames

# Create directory if it does not exist
directory_name = os.path.join('uv_RGB_match_check', 'DS'+dataset_number)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print('Directory ' + directory_name + ' created.')
    
# Choose number of vertices in each direction
submesh_num_vertices_horizontal = 52 # max: 52
submesh_num_vertices_vertical = 102 # max: 102
from submesh import submesh_idx_from_num_vertices_in_each_direction
submesh_idx = submesh_idx_from_num_vertices_in_each_direction(submesh_num_vertices_vertical = submesh_num_vertices_vertical,
                                                              submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
reordered=1 # needed if I take a submesh

# Number of groups and animation frames per group
num_groups, num_animationFramesPerGroup = functions_data_processing.dataset_params(dataset_number)
if args.last_group==0:
    args.last_group = num_groups
        
for group_number_int in range(args.starting_group, args.last_group+1):
    for animation_frame_int in range(1, num_animationFramesPerGroup+1, save_every_these_many_frames):
        group_number = str(group_number_int).zfill(3) # pad with zeros on the left until the string has length 3
        animation_frame = str(animation_frame_int).zfill(5) 
            
        # Load camera parameters
        variables = load_camera_params(sequence_name = sequence_name, dataset_number = dataset_number)
        RT_matrix = variables['RT_matrix']
        RT_extended = variables['RT_extended']
        camera_worldCoord_x = variables['camera_worldCoord_x']
        camera_worldCoord_y = variables['camera_worldCoord_y']
        camera_worldCoord_z = variables['camera_worldCoord_z']
        Intrinsic_matrix = variables['Intrinsic_matrix']
        Camera_proj_matrix = variables['Camera_proj_matrix']

        # Load vertex data
        variables = get_variables_from_vertex_full_Dataframe(sequence_name=sequence_name, dataset_number=dataset_number, 
                                                             group_number=group_number, animation_frame=animation_frame,
                                                             RT_extended=RT_extended, reordered=reordered,
                                                             submesh_idx=submesh_idx, verbose=0)
        u_visible = variables['u_visible']
        v_visible = variables['v_visible']
        u_occluded = variables['u_occluded']
        v_occluded = variables['v_occluded']

        # Plot RGB and landmarks
        fig=functions_plot.plot_RGB_and_landmarks(u_visible=u_visible, v_visible=v_visible, 
                                              u_occluded=u_occluded, v_occluded=v_occluded,
                                              sequence_name=sequence_name, dataset_number=dataset_number, 
                                              group_number=group_number, animation_frame=animation_frame,
                                             marker_size=0.01)
        file_name = os.path.join(directory_name, 'Group' + group_number + '_frame' + animation_frame + '.png')
        fig.savefig(file_name)
