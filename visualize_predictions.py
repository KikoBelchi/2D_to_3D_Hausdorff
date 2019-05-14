###
### Imports
###
import argparse
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import io
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Allow the interactive rotation of 3D scatter plots in jupyter notebook
import sys    
import os    
file_name =  os.path.basename(sys.argv[0])
#print(file_name == 'ipykernel_launcher.py') # This basicaly asks whether this file is a jupyter notebook?
if __name__ == "__main__":
    if file_name == 'ipykernel_launcher.py': # Run only in .ipynb, not in exported .py scripts
        get_ipython().run_line_magic('matplotlib', 'notebook') # Equivalent to ''%matplotlib notebook', but it is also understood by .py scripts
        
from create_faces_for_submesh import create_face_file_from_num_vertices
import data_loading
import functions_data_processing
import functions_plot
from functions_plot import plot_faces, connectpoints, plot_vertices_and_edges
from functions_plot import connectpoints_specific_colour, plot_vertices_and_edges_6edgeColours, create_GIF
from functions_plot import show_image_batch, show_mesh_batch, show_mesh_faces_batch, show_mesh_triangular_faces_batch
from functions_plot import show_mesh_triangular_faces_tensor
import functions_train


# Global variables
face_colour_GT = 'b'
face_colour_prediction = 'y'
vertex_colour_GT = 'b'
vertex_colour_prediction = 'y'

# Parser for entering arguments and setting default ones
def preparing_args():
    args = functions_data_processing.parser()
#     args.towelPixelSample = 0
   
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.predict_uv_or_xyz=='uv' and args.directory_prepend == '01_Bad_normalization_Barycenter_radius1_squareBdingBox': 
        # if args.directory_prepend was not specified
        args.directory_prepend = '04_uv_pred'

    #     face_colour_GT = 'y'
    #     face_colour_prediction = 'b'
    #     vertex_colour_GT = 'y'
    #     vertex_colour_prediction = 'b'

    return args
    
def axis_lims(args):
    xmin, ymin, zmin, xmax, ymax, zmax = None, None, None, None, None, None
#                 if '01_Bad_normalization_Barycenter_radius1_squareBdingBox' in args.directory_prepend or '02_No_normalization_squareBdingBox' in args.directory_prepend or '05_Barycenter_radius1_rectBdingBox' in args.directory_prepend: 
#                     # 01 --> fit roughly within unit ball centered at origin; squared box in the 2D image
#                     # 02 --> shouldn't really need this, since the 3d reconstruction could fall outside of these axis limits.
#                     # 05 --> fit roughly within unit ball centered at origin; rectangular box in the 2D image
#                     xmin=-1
#                     xmax=1
#                     ymin=-1
#                     ymax=1
#                     zmin=-1
#                     zmax=1
#                 elif ('05_no_normalizations' in args.directory_prepend) or ('06_checking_normalizations' in args.directory_prepend):
#                     xmin, ymin, zmin, xmax, ymax, zmax = functions_data_processing.find_min_max_xyz_training_wo_outliers(
#                         dataloaders, boxplot_on = 0)
#                 else: # 0<= x, y, z <=1 normalization
#                     xmin=0
#                     xmax=1
#                     ymin=0
#                     ymax=1
#                     zmin=0
#                     zmax=1
    return xmin, ymin, zmin, xmax, ymax, zmax

def get_labels_on5356vertices(args, kwargs_normalize):
    """ For the plot of surface + 6 vertices, I need labels_5356vertices too,
    which are the vertex (and normal) coordinates of all vertices.
    """
    num_vertices_aux = args.num_selected_vertices 
    args.num_selected_vertices = 5356
    transformed_dataset_5356vertices = data_loading.vertices_Dataset(
        args, num_normalizing_landmarks=args.num_selected_vertices, **kwargs_normalize)
    args.num_selected_vertices = num_vertices_aux

    transformed_dataset_parts_5356vertices, dataset_sizes_5356vertices, dataloaders_5356vertices =  data_loading.random_split_notMixingSequences( 
        dataset=transformed_dataset_5356vertices, args=args)

    # Pick the first training, validation or test batch
    if args.train_or_val==0:
        iterable_dataloaders_5356vertices = iter(dataloaders_5356vertices['train'])
    elif args.train_or_val==1:
        iterable_dataloaders_5356vertices = iter(dataloaders_5356vertices['val'])
    elif args.train_or_val==2:
        iterable_dataloaders_5356vertices = iter(dataloaders_5356vertices['test'])
    sample_batched_5356vertices = next(iterable_dataloaders_5356vertices)
    if args.verbose==1:
        print("sample_batched_5356vertices['image'].shape[0], batch_size:", 
              sample_batched_5356vertices['image'].shape[0], args.batch_size)

    if args.predict_uv_or_xyz=='xyz':
        if args.normals == 1:
            # Shape args.batch_size x (args.num_selected_vertices*2) x 3
            labels_5356vertices = torch.cat((sample_batched_5356vertices['Vertex_coordinates'],
                                             sample_batched_5356vertices['normal_coordinates']), 1) 
        else:
            # Shape args.batch_size x args.num_selected_vertices x 3
            labels_5356vertices = sample_batched_5356vertices['Vertex_coordinates'] 
        # labels_5356vertices = labels_5356vertices.to(args.device)
    else:
        # Shape args.batch_size x args.num_selected_vertices x 2
        labels_5356vertices = sample_batched_5356vertices['uv'] 
    return labels_5356vertices

def not_reordered_3D_plots(args, outputs, labels):
    # For every observation in the already selected batch
    for sample_idx_within_batch in range(args.batch_size):
#     for sample_idx_within_batch in [0]:
        ###
        ### 3D plot of the faces of the meshes
        ### (The prediction on the 5356 vertices is awful)
        ###    
        if args.num_selected_vertices==5356:
            # Load face data (see visualize_mesh.ipynb for details)
            faces = genfromtxt('Renders' + args.sequence_name + args.dataset_number + '/faces_mesh.txt', delimiter=' ')
            faces = faces.astype(int)

            fig = plt.figure()
            ax = Axes3D(fig)
            figure_name='groundTruth_faces_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            plot_faces(labels, faces, colour=face_colour_GT, swap_axes=args.swap_axes,
                       sample_idx_within_batch=sample_idx_within_batch)
            fig.savefig('Prediction_plots/' + figure_name + '.png')

            fig = plt.figure()
            ax = Axes3D(fig)
            figure_name='prediction_faces_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            plot_faces(outputs, faces, colour=face_colour_prediction, swap_axes=args.swap_axes,
                      sample_idx_within_batch=sample_idx_within_batch)
            fig.savefig('Prediction_plots/' + figure_name + '.png')





        ###
        ### Plot vertex position and edges connecting them
        ### Do it for ground-truth and prediction in separate figures
        ###
        if args.num_selected_vertices==6: 
            # Create ground truth plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            figure_name='bdryGT_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            plot_vertices_and_edges_6edgeColours(tensor_of_coordinates=labels, colour=vertex_colour_GT, swap_axes=args.swap_axes,
                          sample_idx_within_batch=sample_idx_within_batch)
            fig.savefig('Prediction_plots/' + figure_name + '.png')

            # Create GIF by rotating the previous plot
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, crop_centre_or_ROI=args.crop_centre_or_ROI) 

            # Create prediction plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # plot_faces(labels, faces, colour=vertex_colour_GT, figure_num=1)
            plot_vertices_and_edges_6edgeColours(outputs, 
                                                 colour=vertex_colour_prediction, swap_axes=args.swap_axes,
                                                 sample_idx_within_batch=sample_idx_within_batch)
            figure_name='bdryPredicted_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            # If there is only ground truth being plotted,
            # then we do not need to append the kind of crop at the end of the file name.
            if args.crop_centre_or_ROI==0: # centre crop
                figure_name = figure_name + '_centreCrop'
            elif args.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
                figure_name = figure_name + '_ROI'
            fig.savefig('Prediction_plots/' + figure_name + '.png')

            # Create GIF by rotating the previous plot
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, crop_centre_or_ROI=args.crop_centre_or_ROI) 





        ###
        ### Plot vertex position and edges connecting them.
        ### Do it for ground-truth and prediction in the same figure,
        ### where any ground truth edge and its corresponding prediction share the same colour.
        ### (The prediction on the 6 vertices is a decent first approximation. At least, it doesn't look random)
        ###

        if args.num_selected_vertices==6:            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_vertices_and_edges_6edgeColours(labels, colour=vertex_colour_GT, swap_axes=args.swap_axes,
                                   sample_idx_within_batch=sample_idx_within_batch)
            plot_vertices_and_edges_6edgeColours(outputs, colour=vertex_colour_prediction, swap_axes=args.swap_axes,
                                   sample_idx_within_batch=sample_idx_within_batch)
            figure_name = 'bdryGT_bdryPredicted_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            if args.crop_centre_or_ROI==0: # centre crop
                figure_name = figure_name + '_centreCrop'
            elif args.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
                figure_name = figure_name + '_ROI'
            fig.savefig('Prediction_plots/' + figure_name + '.png')

            # Create GIF
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, crop_centre_or_ROI=args.crop_centre_or_ROI) 



        ###
        ### Plot ground truth whole surface, 6 vertices and edges connecting them, 
        ### and same without the surface for prediction
        ###

        if args.num_selected_vertices==6:
            # Load face data (see visualize_mesh.ipynb for details)
            faces = genfromtxt('Renders' + args.sequence_name + args.dataset_number + '/faces_mesh.txt', delimiter=' ')
            faces = faces.astype(int)

            fig = plt.figure()
            ax = Axes3D(fig)

            # Plot ground truth surface
            plot_faces(labels_5356vertices, faces, colour=face_colour_GT, swap_axes=args.swap_axes,
                      sample_idx_within_batch=sample_idx_within_batch)
            figure_name = 'surfaceGT_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            fig.savefig('Prediction_plots/' + figure_name + '.png')
            # Create GIF
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, 
                       crop_centre_or_ROI=args.crop_centre_or_ROI)   
        #     ax = fig.add_subplot(111, projection='3d')

            # Add to the plot ground-truth 6 vertices and edges connecting them
            plot_vertices_and_edges_6edgeColours(labels, colour=vertex_colour_GT, swap_axes=args.swap_axes,
                                   sample_idx_within_batch=sample_idx_within_batch)
            figure_name = 'surfaceGT_bdryGT_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            fig.savefig('Prediction_plots/' + figure_name + '.png')
            # Create GIF
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, crop_centre_or_ROI=args.crop_centre_or_ROI)   

            # Add to the plot predicted 6 vertices and edges connecting them
            plot_vertices_and_edges_6edgeColours(outputs, colour=vertex_colour_prediction, swap_axes=args.swap_axes,
                                   sample_idx_within_batch=sample_idx_within_batch)
            figure_name = 'surfaceGT_bdryGT_bdryPredicted_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            if args.crop_centre_or_ROI==0: # centre crop
                figure_name = figure_name + '_centreCrop'
            elif args.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
                figure_name = figure_name + '_ROI'
            fig.savefig('Prediction_plots/' + figure_name + '.png')
            # Create GIF
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, crop_centre_or_ROI=args.crop_centre_or_ROI) 





        ###
        ### Plot ground truth whole surface and predicted 6 vertices and edges connecting them
        ###

        if args.num_selected_vertices==6:
            # Load face data (see visualize_mesh.ipynb for details)
            faces = genfromtxt('Renders' + args.sequence_name + args.dataset_number + '/faces_mesh.txt', delimiter=' ')
            faces = faces.astype(int)

            fig = plt.figure()
            ax = Axes3D(fig)

            # Plot ground truth surface
            plot_faces(labels_5356vertices, faces, colour=face_colour_GT, swap_axes=args.swap_axes,
                      sample_idx_within_batch=sample_idx_within_batch)

            # Add to the plot predicted 6 vertices and edges connecting them
            plot_vertices_and_edges_6edgeColours(outputs, colour=vertex_colour_prediction, swap_axes=args.swap_axes,
                                   sample_idx_within_batch=sample_idx_within_batch)
            figure_name = 'surfaceGT_bdryPredicted_'+str(args.num_selected_vertices)+'v'
            if args.train_or_val==1:
                figure_name = 'valSample'+str(sample_idx_within_batch)+'_' + figure_name
            else:
                figure_name = 'trainSample'+str(sample_idx_within_batch)+'_' + figure_name
            if args.swap_axes==0:
                figure_name=figure_name+'_noSwapAxes'
            if args.crop_centre_or_ROI==0: # centre crop
                figure_name = figure_name + '_centreCrop'
            elif args.crop_centre_or_ROI==1: # Region of interest. Bounding box containing the towel
                figure_name = figure_name + '_ROI'
            fig.savefig('Prediction_plots/' + figure_name + '.png')
            # Create GIF
            GIF_name = figure_name + '_' + str(args.degrees_of_each_rotation) + 'deg'
            create_GIF(fig, GIF_name, args.degrees_of_each_rotation, crop_centre_or_ROI=args.crop_centre_or_ROI) 

def check_label_n_output_ranges(labels, outputs):
    print('\nGround truth labels:')
    print('Type:', type(labels))
    print('Shape:', labels.shape)
    print('Mininum absolute value of ground truth coordinates', labels.abs().min())
    print('Maximum absolute value of ground truth coordinates', labels.abs().max())
    print('Full data:')
    print(labels)
    print('\nPredicted labels:')
    print('Type:', type(outputs))
    print('Shape:', outputs.shape)
    print('Mininum absolute value of predicted coordinates', outputs.abs().min())
    print('Maximum absolute value of predicted coordinates', outputs.abs().max())
    print('Full data:')
    print(outputs)

def visualize_predictions_1_directory(model_directory, args, visualize=1, kwargs_normalize=None, kwargs_normalize_labels_pred=None):
    # If visualize=0, no visualizations are plotted, but computations are made.
    # This is useful in order_unsup_uv_predictions.py
    
    # Dataset and random split (see transfer_learning_TowelWall.py for more information)
    # Every random choice has a seed for reproducibility.
    if '01_Bad_normalization_Barycenter_radius1_squareBdingBox' in args.directory_prepend or '05_Barycenter_radius1_rectBdingBox' in args.directory_prepend:
        transformed_dataset = data_loading.vertices_Dataset_barycenter(sequence_name = args.sequence_name, 
                                                            dataset_number = args.dataset_number,
                                                            transform=args.transform,
                                                            camera_coordinates=args.camera_coordinates,
                                                            crop_centre_or_ROI=args.crop_centre_or_ROI,
                                                            reordered_dataset = args.reordered_dataset,
                                                            num_vertices=args.num_selected_vertices,
                                                            submesh_num_vertices_vertical = args.submesh_num_vertices_vertical,
                                                            submesh_num_vertices_horizontal = args.submesh_num_vertices_horizontal)
    elif (args.directory_prepend =='02_No_normalization_squareBdingBox') or (args.predict_uv_or_xyz=='uv') or ('05_no_normalizations' in args.directory_prepend) or ('06_checking_normalizations' in args.directory_prepend):
        transformed_dataset = data_loading.vertices_Dataset(args)
    elif '03_xyz_normalization_squareBdingBox' in args.directory_prepend or '04_xyz_normalization_noCrop' in args.directory_prepend: # If one of these is a substring of args.directory_prepend
        text_filename = os.path.join(model_directory, "normalization_params_from_training.txt")
        f = open(text_filename,"r")
        text = f.readlines()
        f.close()
        second_line = text[1].split(' ') # list of all values in the 2nd line in string format
        normalize_xyz_min = [float(x) for x in second_line[0:3]]
        normalize_xyz_max = [float(x) for x in second_line[3:6]]

        transformed_dataset = data_loading.vertices_Dataset(
            args, normalize_xyz_min=normalize_xyz_min, normalize_xyz_max=normalize_xyz_max)
    elif '07' in args.directory_prepend:
        transformed_dataset = data_loading.vertices_Dataset(args, **kwargs_normalize)

    # Create dataloaders
    transformed_dataset_parts, dataset_sizes, dataloaders = functions_train.create_dataloaders(transformed_dataset, args)
    
    # Get template
    # CAVEAT: to make the first tests, I will take template = transformed_dataset[0],
    # but the 0th element may not be in the training set, 
    # so if the unsupervised approach works, I will change this to 
    # make the template be an observation independent from the rest of the dataset.
    args.template = transformed_dataset[0]
        
    # Load model
    # For models trained after 11 Jan 2019, 14:30h simply use this:
#     model_filename = 'model.pt' if args.round==1 else 'model_rd2.pt'
    model_filename = 'model.pt'
    model_filename = os.path.join(model_directory, model_filename)
#     # For older models, use this:
#     if args.hyperpar_opt==1 or ((('01_Bad_normalization_Barycenter_radius1_squareBdingBox' in args.directory_prepend or '02_No_normalization_squareBdingBox' in args.directory_prepend) and args.frozen_resnet==1) and args.dataset_number not in ['7', '8']):
#         model_filename = model_directory + '/model.pt'
#     elif args.predict_uv_or_xyz=='uv' and args.dataset_number not in ['7', '8']:
#         model_filename = model_directory + '/model_best_wrt_val_set.pt'
#     else:
#         model_filename = model_directory + '/model_after_last_epoch.pt'
    loaded_model = torch.load(model_filename, map_location='cpu')
    loaded_model.train() if args.testing_eval==3 else loaded_model.eval()

    # Save losses of training, validation and test set
    if args.save_losses==2:
        functions_train.evaluate_model(model=loaded_model, saving_directory=model_directory, dataloaders=dataloaders, args=args, kwargs_normalize_labels_pred=kwargs_normalize_labels_pred)
        labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord = -1, -1, -1, -1, -1, -1
    else:
        # Save losses of training, validation and test set
        if args.save_losses==1:
            functions_train.evaluate_model(model=loaded_model, saving_directory=model_directory, dataloaders=dataloaders, args=args, kwargs_normalize_labels_pred=kwargs_normalize_labels_pred)

        # Pick the first (0th) training, validation or test batch (or another one if specified by args.batch_to_show)
        if args.train_or_val==0:
            iterable_dataloaders = iter(dataloaders['train'])
        elif args.train_or_val==1:
            iterable_dataloaders = iter(dataloaders['val'])
        elif args.train_or_val==2:
            iterable_dataloaders = iter(dataloaders['test'])
        # Pick the batch number args.batch_to_show
        for i in range(args.batch_to_show + 1):
            sample_batched = next(iterable_dataloaders)
            
        if args.elt_within_batch is None:
            print("sample_batched['img_name']:", sample_batched['img_name'])
        else:
            print("sample_batched['img_name'][" + str(args.elt_within_batch) + "]:", sample_batched['img_name'][args.elt_within_batch])
            
        if args.verbose==1:
            print("sample_batched['image'].shape[0], batch_size:", sample_batched['image'].shape[0], args.batch_size)

        if args.verbose==1:
            if args.train_or_val==0:
                print('Output and Loss on the first training batch.')
            elif args.train_or_val==1:
                print('Output and Loss on the first validation batch.')
            elif args.train_or_val==2:
                print('Output and Loss on the first test batch.')
            print('\nImage names of batch:', sample_batched['img_name'])
            print()

        # Predict the output and loss of the shown batch
        labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord, labels_D, outputs_D = functions_train.predict_and_loss(
            model = loaded_model, sample_batched = sample_batched, args=args, kwargs_normalize=kwargs_normalize_labels_pred)
#         check_label_n_output_ranges(labels, outputs)






        ###
        ### For the plot of surface + 6 vertices, I need labels_5356vertices too,
        ### which are the vertex (and normal) coordinates of all vertices
        ###
        if args.sequence_name=='TowelWall': labels_5356vertices = get_labels_on5356vertices(args, kwargs_normalize)

        ###
        ### Run the visualizations 
        ###
        if visualize==1:
            if args.predict_uv_or_xyz in ['xyz', 'uvy', 'uvD']: # xyz prediction
                labels=labels_xyz_coord
                outputs=outputs_xyz_coord
                if args.reordered_dataset==1:
                    xmin, ymin, zmin, xmax, ymax, zmax = axis_lims(args)                    
#                     plot_pred = 0 if args.show_vertices==0 else 1
                    functions_plot.visualize_xyz_prediction_grid(args, labels, outputs, sample_batched=None, save_png=0, transparency=args.transparency, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, plot_pred=args.plot_3D_pred, plot_GT=args.plot_3D_GT)
                elif args.reordered_dataset==0:
                    not_reordered_3D_plots(args, outputs, labels)
            if args.predict_uv_or_xyz=='uvD': # uvD prediction
                xmin, ymin, zmin, xmax, ymax, zmax = axis_lims(args)                    
#                 plot_pred = 0 if args.show_vertices==0 else 1
#                 xmin=0, xmax=224, ymin=0, ymax=224, zmin=5, zmax=10
                print(outputs_uv_coord.shape, outputs_D.shape)
                functions_plot.plot_3D_vertices(
                    X=outputs_uv_coord[0, :,0], Y=outputs_uv_coord[0, :,1], Z=outputs_D[0,:,:], sequence_name=args.sequence_name,
                    dataset_number=args.dataset_number, marker_size = 25/9, swap_axes=1,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, title='uv + depth',
        X_label='u', Y_label='v', Z_label='z')
            if 'uv' in args.predict_uv_or_xyz: # uv prediction
                labels=labels_uv_coord
                outputs=outputs_uv_coord
                if args.log_epochs!=0:
                    epochs_to_load = [epoch for epoch in range(args.num_epochs) if epoch%args.log_epochs==0]
                    for epoch in epochs_to_load:
                        for phase in ['train', 'val']:
                            labels, outputs, loss, sample_batched_img_name = functions_train.load_predictions(
                                phase, epoch+1, saving_directory=model_directory)
                            outputs.requires_grad = False
                            print('epoch, phase, batch_loss:', epoch+1, phase, loss.item())
#                             if args.batch_size_to_show == 12 or args.grid==1:
                            if True:
                                functions_plot.visualize_uv_prediction_grid(
                                args, labels, outputs, sample_batched_img_name=sample_batched_img_name,
                                    model_directory=model_directory)
#                             else:
#                                 functions_plot.visualize_uv_prediction(
#                                 args, labels, outputs, sample_batched_img_name=sample_batched_img_name,
#                                     model_directory=model_directory)
                else:   
#                     if args.batch_size_to_show == 12 or args.grid==1:
                    if True:
                        # Plot prediction on original RGB
                        functions_plot.visualize_uv_prediction_grid(
                        args, labels, outputs, sample_batched_img_name=sample_batched['img_name'], model_directory=model_directory, save_png=0)
                        if args.uv_normalization in [1,2] and args.show_transformed_RGB==1: 
                            # Plot prediction on the transformed RGB (after potential crop, resize and normalization of colours)
                            functions_plot.visualize_uv_prediction_grid(
                            args, labels, outputs, sample_batched_img_name=sample_batched['img_name'], sample_batched=sample_batched,
                                model_directory=model_directory, plot_on_input=1)
                            # Plot prediction on the transformed RGB (after potential crop, resize and normalization of colours)
                            # and the predicted and ground truth 0<=u,v<=1 are multiplied by 
                            # the width and height of their corresponding crop.
                            functions_plot.visualize_uv_prediction_grid(
                            args, labels, outputs, sample_batched_img_name=sample_batched['img_name'], sample_batched=sample_batched, model_directory=model_directory,
                            plot_on_input=2)
                            # Plot transformed RGB (after potential crop, resize of colours)
                            functions_plot.show_unnormalized_image_batch(sample_batched)
#                     else:
#                         functions_plot.visualize_uv_prediction(
#                         args, labels, outputs, sample_batched_img_name=sample_batched['img_name'], model_directory=model_directory)

                    if (args.save_png_path is None) and (args.save_losses!=2) and (args.auto_save == 0): plt.show()
    return labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord
                
def run_visualization(args, visualize=1):
    # If visualize=0, no visualizations are plotted, but computations are made.
    # This is useful in order_unsup_uv_predictions.py
    
    ### Model directory
    model_directory = functions_train.trained_model_directory(args)
    model_directory = os.path.join(args.directory_prepend, model_directory)
    
    # Normalization parameters
    if args.all_normalizing_provided==0:
        normalize_xyz_min, normalize_xyz_max = functions_data_processing.load_min_max_xyz_from_training(args, model_directory=model_directory)
        normalize_D_min, normalize_D_max = functions_data_processing.load_min_max_D_from_training(args, model_directory=model_directory)
    else:
        normalize_xyz_min, normalize_xyz_max, normalize_D_min, normalize_D_max = None, None, None, None
    kwargs_normalize, kwargs_normalize_labels_pred = functions_data_processing.create_kwargs_normalize(normalize_xyz_min, normalize_xyz_max, normalize_D_min, normalize_D_max, args)

    # Once we have fixed the directory containing the models trained on args.dataset_number,
    # the rest (loading of dataset to split and make prediction on) will be done with args.dataset4predictions
    args.dataset_number = args.dataset4predictions
    args.sequence_name = args.sequence4predictions
    if args.sequence_name=='TowelWall':
        args.num_groups, args.num_animationFramesPerGroup = args.num_groups4predictions, args.num_animationFramesPerGroup4predictions
    # Once we have fixed the directory containing the models trained on args.batch_size,
    # the rest (loading of dataset and creation of dataloaders) will be done with args.batch_size_to_show
    args.batch_size = args.batch_size_to_show
    args.lengths_proportion_train = args.lengths_proportion_train_4visuals
    args.lengths_proportion_test = args.lengths_proportion_test_4visuals
    args.GTuv = 0 # Use predicted uv to compute xyz
    if args.append_to_model_dir is not None:
        args.models_lengths_proportion_train_nonAnnotated = args.append_to_model_dir.split('_')[0] + "_both_finetune"
    else:
        args.models_lengths_proportion_train_nonAnnotated = str(args.lengths_proportion_train_nonAnnotated)
        if args.train_w_annot_only == 0:
                args.models_lengths_proportion_train_nonAnnotated += "_both"
    args.lengths_proportion_train_nonAnnotated = None

    generic_save_png_path=args.save_png_path

    if args.weights2visualize!=-1:
    #     if args.hyperpar_opt==1 and (args.neighb_dist_weight!=0 or args.normals!=0):
        subdirectories=next(os.walk(model_directory))[1]
        for directory in subdirectories:
            if (args.weights2visualize==0) or (args.weights2visualize==1 and directory.split('neighbDistW')[1] in ['0.00', '0.10', '0.40', '1.00']) or (args.weights2visualize==2 and directory.split('neighbDistW')[1] in ['0.00', '0.80']):
                model_subdirectory = os.path.join(model_directory, directory)
                print('\nModel from directory:', model_subdirectory)
                if generic_save_png_path is not None:
                    if args.train_or_val==0: split_string = 'train'
                    elif args.train_or_val==1: split_string = 'val'
                    elif args.train_or_val==2: split_string = 'test'
                    args.save_png_path = os.path.join(
                        generic_save_png_path, split_string + directory.split('neighbDistW')[1] + '.png') 
                    print("Saving ", args.save_png_path)
                labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord = visualize_predictions_1_directory(model_directory=model_subdirectory, args=args, visualize=visualize, kwargs_normalize=kwargs_normalize, kwargs_normalize_labels_pred=kwargs_normalize_labels_pred)
    else:
        print('Model from directory:', model_directory)
        labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord = visualize_predictions_1_directory(model_directory=model_directory, args=args, visualize=visualize, kwargs_normalize=kwargs_normalize, kwargs_normalize_labels_pred=kwargs_normalize_labels_pred)
    
    return labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord, model_directory 

if __name__=='__main__':
    args = preparing_args()
    labels, labels_uv_coord, labels_xyz_coord, outputs, outputs_uv_coord, outputs_xyz_coord, model_directory = run_visualization(args)