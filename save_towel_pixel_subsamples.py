""" 
1. Find the towel pixels and the contour of such area.
2. Save a subsample of both subsets (as torch tensor and as numpy array).
3. Do this for every picture in a given dataset, and given the subsampling sizes. 

To SAVE numpy arrays on RendersTowelWall datasets, run:
>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --dataset-number 15 --subsample-size 6 --subsample-size-contour 7 --towelPixelSample 1

>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --dataset-number 11 --subsample-size 6 --subsample-size-contour 7 --towelPixelSample 1

>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --dataset-number 11 --subsample-size 100 --subsample-size-contour 100 --towelPixelSample 1

To SAVE numpy arrays on DeepCloth datasets, run:
>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --sequence-name 'DeepCloth' --dataset-number 2 --subsample-size 250 --subsample-size-contour 150 --towelPixelSample 1 --save-tensor 0

--subsample-size-contour 80 is too large, as you get the following error
DeepCloth2/train/train_non-text/imgs/009791.png
towel_pixel_subsample.shape: torch.Size([250, 2])
Saving to: DeepCloth2/train/train_non-text/Subsamples/009791_towel_pixel_subsample250.txt
Traceback (most recent call last):
  File "save_towel_pixel_subsamples.py", line 97, in <module>
    subsample = functions_train.sample_of_towel_pixels(img_name, args) # tensor of shape (M, 2)
  File "/home/fbelchi/2d_to_3d/functions_train.py", line 502, in sample_of_towel_pixels
    subsample_idx = random.sample(range(uv_towel.shape[0]), n_pixels_in_subsample)
  File "/home/fbelchi/2d_to_3d/2d_to_3d_py_venv/lib/python3.5/random.py", line 315, in sample
    raise ValueError("Sample larger than population")
ValueError: Sample larger than population

    SAVE towel pixels from GT uv:
    >>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --sequence-name 'DeepCloth' --dataset-number 2 --subsample-size 81 --subsample-size-contour 32 --towelPixelSample 1 --save-tensor 0 --GTtowelPixel 1 --submesh-num-vertices-vertical 9 --submesh-num-vertices-horizontal 9

To PLOT subsamples, to look for suitable values of subsample size for RendersTowelWall datasets, run:
>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --dataset-number 11 --subsample-size 100 --subsample-size-contour 100 --towelPixelSample 1 --save-tensor 2

To PLOT subsamples, to look for suitable values of subsample size for DeepCloth datasets, run:
>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --sequence-name 'DeepCloth' --dataset-number 2 --subsample-size 1 --subsample-size-contour 1 --towelPixelSample 1 --save-tensor 2

>>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --sequence-name 'DeepCloth' --dataset-number 2 --subsample-size 250 --subsample-size-contour 50 --towelPixelSample 1 --save-tensor 2

    PLOT towel pixels from GT uv:
    >>> python save_towel_pixel_subsamples.py --crop-centre-or-ROI 3 --sequence-name 'DeepCloth' --dataset-number 2 --subsample-size 81 --subsample-size-contour 32 --towelPixelSample 1 --save-tensor 2 --GTtowelPixel 1 --submesh-num-vertices-vertical 9 --submesh-num-vertices-horizontal 9



CAVEAT: 
To load the files from the dataset you'll need --towelPixelSample 2. 
I.e., to run the training with the results from this script.
However, to save the files, which is, for running this script, you need to use --towelPixelSample 1.
Otherwise, the subsample is not applied, and the full sample is saved in a file with the name subsample6, for instance.
This is due to the fact that we compute the subsamples within the dataset with --towelPixelSample 1 and save the files.
"""

import argparse
import numpy as np
import os
import pandas as pd
import torch

import data_loading
import functions_data_processing
import functions_train
import submesh
# NOTE ON THE FORMAT:
# I will do all computations as double and at the end, round to float, so run it by using:
# python uvy2xyz_GT.py --dtype 1 --submesh-num-vertices-vertical 52 --submesh-num-vertices-horizontal 103 --dataset-number 11

plot_RGB_alone=1

args = functions_data_processing.parser()
root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number)
print(root_dir)
dataset_length = data_loading.dataset_length_from_name(args)

if args.save_tensor==1:
    print("Saving subsamples as tensors")
elif args.save_tensor==0:
    print("Saving subsamples as numpy arrays")
elif args.save_tensor==2:
    print("Plotting subsamples without saving them")
    import matplotlib.pyplot as plt
    import functions_plot
               
def save_or_plot_subsample(args, subsample, filename, img_name=None):
    if args.save_tensor==1: torch.save(subsample, filename) # save as tensor
    elif args.save_tensor==0:  # save as numpy array
        if args.sequence_name == 'DeepCloth':
            np.savetxt(filename, subsample)
        else:
            np.savetxt(filename, subsample.to(torch.device("cpu")).numpy())
    elif args.save_tensor==2: 
        functions_plot.plot_RGB_and_landmarks(subsample[:,0], subsample[:,1], image_path=img_name, axis_on='off', visible_colour='y')
        if args.contour==1: functions_plot.show_for_seconds(milliseconds=400000)
            
def towel_uv_from_GT(idx, args):
    """ Get the GT uv, round them to integer values and keep this as set of towel pixels.
    Next thing we could add is removing repetitions of pixels """
    # Load vertex coordinates
    root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number)
    sample = data_loading.load_coordinates_DeepCloth2(root_dir=root_dir, idx=idx, dtype=args.dtype)
#     uv_pixel = np.round(sample['uv']) # I should NOT round here, since uv can already be normalized to be in [0, 1]
    uv_pixel = sample['uv']
    if args.contour==1:
        uv_pixel = uv_pixel[submesh.contour_of_mesh(args)]
    return uv_pixel
        
random_seed=0
# for idx in range(dataset_length):
for idx in [19, 9791]:
    if args.sequence_name == 'DeepCloth':
        group_number, animation_frame = None, None
        # Create directory if it does not exist
        directory_name = os.path.join(root_dir, 'Subsamples')
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            print('Directory ' + directory_name + ' created.')
    else:
        group_number, animation_frame = data_loading.group_and_frame_from_idx(
            idx, num_groups=args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup)
        group_number, animation_frame = data_loading.post_process_grp_and_frame(group_number, animation_frame)
        print(group_number, animation_frame)
        
    # Image name from index
    img_name_end, crop_option = data_loading.imgNameEnd_from_cropOption(args.crop_centre_or_ROI)
    img_name = data_loading.imgName_from_idx(root_dir, args, img_name_end, idx)
    print("\n" + img_name)

#         xyz_world_projected_np = xyz_world_projected.float().numpy()
#         uv=uv.astype(np.float32)
    
    if plot_RGB_alone==1:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        functions_plot.plot_RGB(ax=ax, image_path=img_name, axis_on='off')
    for contour in [0,1]:
        args.contour=contour
        # Find towel pixels 
        if args.GTtowelPixel==0:
            uv_towel = functions_train.towel_pixels(img_name, args) # tensor of shape (M, 2)
        else:
            uv_towel = towel_uv_from_GT(idx, args)
        # Subsample towel pixels
        random_seed+=1
        subsample = functions_train.sample_of_towel_pixels(uv_towel, args, random_seed) # tensor of shape (M, 2)
        text_to_print = "towel_pixel_subsample.shape:" if contour == 0 else "towel_pixel_contour_subsample.shape:"
        print(text_to_print, subsample.shape)
        # Visualize and save
        filename = data_loading.towel_pixel_subsample_filename(args, root_dir, crop_option, idx, group_number, animation_frame)
        if args.save_tensor!=2: print("Saving to:", filename)
        save_or_plot_subsample(args, subsample, filename, img_name)

def how_to_load_tensor(filename):
        # Load the tensor onto the chosen device like this:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("\nLoading:", filename)
        dictio={}
        dictio['subsample'] = torch.load(filename, map_location=device)
        # To let torch.device() be compatible with torch.load(..., map_location=torch.device()), 
        # upgrading torch to version 0.4.1:
        # pip install --upgrade torch torchvision
        print("loaded towel_pixel_contour_subsample.shape:", dictio['subsample'].shape)
        print("loaded towel_pixel_contour_subsample.device:", dictio['subsample'].device)

if args.save_tensor==1:
    filename = os.path.join(root_dir + str(group_number).zfill(3), 'vertices_' + str(animation_frame).zfill(5)) + crop_option
    filename += '_towel_pixel_contour_subsample' + str(args.subsample_size_contour) + '.pt'
    how_to_load_tensor(filename)

def how_to_load_np_array(filename):
        print("\nLoading:", filename)
        dictio={}
        dictio['subsample'] = np.genfromtxt(filename).astype(np.float32)
        print("loaded towel_pixel_contour_subsample.shape:", dictio['subsample'].shape)
        print("loaded towel_pixel_contour_subsample[:5,:]:\n", dictio['subsample'][:5,:])

if args.save_tensor==0:
    filename = data_loading.towel_pixel_subsample_filename(args, root_dir, crop_option, idx, group_number, animation_frame)
    how_to_load_np_array(filename)