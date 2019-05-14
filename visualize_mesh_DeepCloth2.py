import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import functions_data_processing
import functions_plot

sequence_name = 'DeepCloth'
dataset_number = '2'
img_versions = ['depths', 'depths_occl', 'imgs', 'imgs_occl', 'imgs_spec', 'imgs_spec_occl', 'imgs_tmpl']
n_rows = 3
n_columns = 5

def reproject_3D_to_2D(camera_coord, resW=223., resH=223., f_u=1, f_v=1):
    """ Input: xyz camera coordinates
    Output: uv pixel coordinates"""
    if isinstance(camera_coord, np.ndarray):
        uv = np.zeros((camera_coord.shape[0], 2), dtype=camera_coord.dtype)
        uv[:, 0] = (camera_coord[:, 0] * f_u / camera_coord[:, 2] + resW / 2.0)
        uv[:, 1] = ((camera_coord[:, 1] + resH) * f_v / camera_coord[:, 2] + resH / 2.0)
    return uv

marker_size=15
for example_id in range(0, 1):
    fig = plt.figure()
    directory_name = os.path.join(sequence_name + dataset_number, 'train', 'train_non-text')
    vertex_filename = os.path.join(directory_name, 'poses', str(example_id).zfill(6) + '.csv')
    print('Loading data from', vertex_filename)
    uvD = np.genfromtxt(vertex_filename, delimiter=',') #.astype(np.float32)
    u, v, z = uvD[:, 0], uvD[:, 1], uvD[:, 2]
    
    # Plot uvD
    i=1
    ax = fig.add_subplot(n_rows,n_columns,i, projection='3d')

    xmin, ymin, zmin, xmax, ymax, zmax, fig, ax= functions_plot.plot_3D_vertices(
        X=u, Y=v, Z=z, sequence_name=sequence_name, dataset_number=dataset_number, marker_size = 25/9, swap_axes=1, 
        xmin=0, xmax=224, ymin=0, ymax=224, zmin=5, zmax=10, ax=ax, fig=fig, title='uv + depth',
        X_label='u', Y_label='v', Z_label='z')
    
    # Show uv on RGB
    img_type='imgs'
    i+=1
    ax = fig.add_subplot(n_rows,n_columns,i)
    image_path = os.path.join(directory_name, img_type, str(example_id).zfill(6) + '.png')
    print('Showing', image_path)
    print('Picture size:', functions_data_processing.get_picture_size(verbose=0, image_path=image_path))
    functions_plot.show_2D_animation_frame(image_path=image_path, show=0, ax=ax)
    uv = functions_data_processing.forget_depth(uvD=uvD)
    ax.scatter(uv[:,0], uv[:,1], marker='o', s=marker_size)
    plt.title("uv")
    
    # Show 2D reprojected vertices on RGB (note that GT is given as uvD, rather than xyz, so this does not work)
    img_type='imgs'
    i+=1
    ax = fig.add_subplot(n_rows,n_columns,i)
    image_path = os.path.join(directory_name, img_type, str(example_id).zfill(6) + '.png')
    print('Showing', image_path)
    print('Picture size:', functions_data_processing.get_picture_size(verbose=0, image_path=image_path))
    functions_plot.show_2D_animation_frame(image_path=image_path, show=0, ax=ax)
    uv = reproject_3D_to_2D(camera_coord=uvD)
    ax.scatter(uv[:,0], uv[:,1], marker='o', s=marker_size)
    plt.title("Reprojected vertices if GT were camera coord xyz")
    
    # Plot xyz
    for f in [1, 600.172]:
        f_u, f_v = f, f
        i+=1
        ax = fig.add_subplot(n_rows,n_columns,i, projection='3d')
        xyz = functions_data_processing.uvD_to_xyz(uvD, f_u=f_u, f_v=f_v)
        xmin, ymin, zmin, xmax, ymax, zmax, fig, ax= functions_plot.plot_3D_vertices(
            X=xyz[:,0], Y=xyz[:,1], Z=xyz[:,2], sequence_name=sequence_name, dataset_number=dataset_number, marker_size = 25/9, 
            swap_axes=1, ax=ax, fig=fig, title='uvD_to_xyz(camera), f_u=f_v=' + str(f))
        
    # Plot xyz using the conversor for tensor batches
    f_u, f_v = 600.172, 600.172
    i+=1
    ax = fig.add_subplot(n_rows,n_columns,i, projection='3d')
    # Batch of two copies of uv
    uv_batch = torch.zeros((2, uvD.shape[0], 2)) 
    for j in range(2):
        uv_batch[j,:,:] = torch.tensor(uvD[:,:2]).view(1, -1, 2)
    
    # Batch of two copies of D
    D_batch = torch.zeros((2, uvD.shape[0], 1)) 
    for j in range(2):
        D_batch[j,:,:] = torch.tensor(uvD[:,2]).view(1, -1, 1)
    
    xyz_batch = functions_data_processing.uvD_to_xyz_batch(uv_batch, D_batch, f_u=f_u, f_v=f_v)
    xyz_computed_via_batch_function = xyz_batch[0,:,:].view(-1, 3)
    xyz_computed_via_batch_function=xyz_computed_via_batch_function.numpy()
    print(abs(xyz_computed_via_batch_function-xyz)<10**(-5))
    xmin, ymin, zmin, xmax, ymax, zmax, fig, ax= functions_plot.plot_3D_vertices(
        X=xyz_computed_via_batch_function[:,0], Y=xyz_computed_via_batch_function[:,1], Z=xyz_computed_via_batch_function[:,2], sequence_name=sequence_name, dataset_number=dataset_number, marker_size = 25/9, 
        swap_axes=1, ax=ax, fig=fig, title='uvD_to_xyz via batch function')


    # Show corresponding image
    for img_type in img_versions:
        i+=1
        ax = fig.add_subplot(n_rows,n_columns,i)
        image_path = os.path.join(directory_name, img_type, str(example_id).zfill(6) + '.png')
        print('Showing', image_path)
        print('Picture size:', functions_data_processing.get_picture_size(verbose=0, image_path=image_path))
        functions_plot.show_2D_animation_frame(image_path=image_path, show=0, ax=ax)
        plt.title(img_type)

    # Maximise figure window
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    plt.show()
