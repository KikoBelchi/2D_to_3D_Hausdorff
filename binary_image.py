""" To see a video of the original and black and white segmented versions of all the examples in the TowelWall dataset, run
>>> python binary_image.py 

To see a video of the original and black and white segmented versions of all the examples in the DeepCloth2 dataset, run
>>> python binary_image.py --sequence-name 'DeepCloth' --dataset-number 2 --ms-per-frame 1000
"""

import cv2
import numpy as np
import os
import torch

def binary_img_from_towel_igm_name(input_png_name, verbose=0, threshold_chosen_TowelOnly=40, sequence_name='TowelWall'):
    """
    Output:
    binary_TowelOnly is a numpy array of shape (540, 960) containing:
        0 in the background and shade pixels, and 
        255 in the towel pixels
    """
    # Read image
    img = cv2.imread(input_png_name)
    height, width, channels = img.shape
    if verbose==1:
        print("original picture's height, width, channels:", height, width, channels)
        cv2.imshow('original',img)
        cv2.waitKey(0)

    img_with_rectangles_TowelOnly = img.copy()
    
    # Threshold
    if sequence_name=='DeepCloth':
        threshold_chosen_TowelOnly=0 #10**(-20)
    elif sequence_name=='TowelWall':
        threshold_chosen_TowelOnly=40
    retval, threshold_TowelOnly = cv2.threshold(img, threshold_chosen_TowelOnly, 255, cv2.THRESH_BINARY)
    if verbose==1:
        cv2.imshow('threshold ' + str(threshold_chosen_TowelOnly) + ' - Towel only', threshold_TowelOnly)
        cv2.waitKey(0)
            
    # Dilate (default = no dilation)
    # kernel = np.ones((10,15), np.uint8)
    kernel = np.ones((1,1), np.uint8)
    img_dilation_TowelOnly = cv2.dilate(threshold_TowelOnly, kernel, iterations=1)
    if verbose==1:
        cv2.imshow('dilated - Towel only', img_dilation_TowelOnly)
        cv2.waitKey(0)

    # Turn into a grayscale image
    gray_TowelOnly = cv2.cvtColor(img_dilation_TowelOnly,cv2.COLOR_BGR2GRAY)
    if verbose==1:
        cv2.imshow('gray - Towel only', gray_TowelOnly)
        cv2.waitKey(0)

    # Turn into a binary image
    if sequence_name=='TowelWall':
        ret, binary_TowelOnly = cv2.threshold(gray_TowelOnly,220,255,cv2.THRESH_BINARY_INV)
        # I invert the colours here so that we can find the contours of the towel with 'cv2.findContours'
    else:
        ret, binary_TowelOnly = cv2.threshold(gray_TowelOnly,220,255,cv2.THRESH_BINARY)
    if verbose==1:
        cv2.imshow('binary - Towel only', binary_TowelOnly)
        cv2.waitKey(0)
        
    return binary_TowelOnly

def find_uv_from_binary_image(binary_TowelOnly, verbose=0, numpy_or_tensor=0, device="cpu", dtype=1):
    """ Find in the binary image named binary_TowelOnly the pixels occupied by the towel. 
    If numpy_or_tensor=0, it returns a numpy array of shape Nx2, where the 1st column corresponds to u and 2nd to v.
    If numpy_or_tensor=1, it returns a tensor with the same properties instead.
    
    dtype==0 -->float
    dtype==1 -->double
    """    
    v_values = range(binary_TowelOnly.shape[0])
    u_values = range(binary_TowelOnly.shape[1])
    if numpy_or_tensor==0:
        return np.array([[u, v] for v in v_values for u in u_values
                         if binary_TowelOnly[v, u]==255])
    else:
        if dtype==0: 
            return torch.tensor([[u, v] for v in v_values for u in u_values
                                 if binary_TowelOnly[v, u]==255], device=device).float() 
        else:
            return torch.tensor([[u, v] for v in v_values for u in u_values
                                 if binary_TowelOnly[v, u]==255], device=device).double() 
    
def find_uv_from_img_name(input_png_name, verbose=0, numpy_or_tensor=0, device="cpu", sequence_name='TowelWall'):
    """ Find in the image named input_png_name the pixels occupied by the towel. 
    If numpy_or_tensor=0, it returns a numpy array of shape Nx2, where the 1st column corresponds to u and 2nd to v.
    If numpy_or_tensor=1, it returns a tensor with the same properties instead.
    """
    # Binary image
    binary_TowelOnly = binary_img_from_towel_igm_name(input_png_name, verbose=0, sequence_name=sequence_name)
    
    return find_uv_from_binary_image(binary_TowelOnly, verbose, numpy_or_tensor, device)

def show_BnW_n_original_video(args, ms_per_frame=25, threshold_chosen_TowelOnly=40):
    """ Show a video of the original and black and white segmented versions of all the examples in a dataset.
    ms_per_frame=0 will display until a key is pressed"""
    root_dir = data_loading.root_dir_from_dataset(args.sequence_name, args.dataset_number)
    dataset_length = data_loading.dataset_length_from_name(args)
    for idx in range(dataset_length):
        if args.sequence_name == 'DeepCloth':
            group_number, animation_frame = None, None
            print("idx:", idx)
        else:
            group_number, animation_frame = data_loading.group_and_frame_from_idx(
                idx, num_groups=args.num_groups, num_animationFramesPerGroup = args.num_animationFramesPerGroup)
            group_number, animation_frame = data_loading.post_process_grp_and_frame(group_number, animation_frame)
            print("idx, group_number, animation_frame:", idx, group_number, animation_frame)

        # Image name from index
        img_name_end, crop_option = data_loading.imgNameEnd_from_cropOption(args.crop_centre_or_ROI)
        img_name = data_loading.imgName_from_idx(root_dir, args, img_name_end, idx)

        # Plot binary image
        binary_TowelOnly = binary_img_from_towel_igm_name(
            img_name, verbose=0, threshold_chosen_TowelOnly=threshold_chosen_TowelOnly, sequence_name=args.sequence_name)
        cv2.imshow('binary - Towel only', binary_TowelOnly)
        img = cv2.imread(img_name)
        cv2.imshow('original',img)
        cv2.waitKey(ms_per_frame)
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    import data_loading
    import functions_data_processing
    import functions_plot
    
    args = functions_data_processing.parser()
            
    DeepCloth_example = os.path.join('DeepCloth2', 'train', 'train_non-text', 'imgs', '027996.png')
    TowelWall_example = os.path.join('RendersTowelWall11', 'Group.003', '17.png')
    for input_png_name in [DeepCloth_example, TowelWall_example]:
        uv_towel = find_uv_from_img_name(input_png_name=input_png_name, verbose=0, sequence_name=args.sequence_name)
        print(uv_towel.shape)
        # Plot uv on RGB
        fig=functions_plot.plot_RGB_and_landmarks(u_visible=uv_towel[:,0], v_visible=uv_towel[:,1],image_path=input_png_name)
        plt.show()

        uv_towel = find_uv_from_img_name(input_png_name=input_png_name, verbose=0, numpy_or_tensor=1, sequence_name=args.sequence_name)
        print(uv_towel.shape)
        # Plot uv on RGB
        fig=functions_plot.plot_RGB_and_landmarks(u_visible=uv_towel[:,0], v_visible=uv_towel[:,1],image_path=input_png_name)
        plt.show()
    
    show_BnW_n_original_video(args, ms_per_frame=args.ms_per_frame, threshold_chosen_TowelOnly=threshold_chosen_TowelOnly)
