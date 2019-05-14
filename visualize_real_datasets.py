import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

import functions_data_processing
import functions_plot

marker_size=15
sequence_name = 'kinect_tshirt'
# sequence_name = 'kinect_paper'
dataset_number = ''
root_dir = sequence_name

# Find uvD filenames
uvD_directory = os.path.join(root_dir, 'processed', 'vertexs')
onlyfiles = [f for f in os.listdir(uvD_directory) if os.path.isfile(os.path.join(uvD_directory, f))]
uvD_idx_list = [int(f.split('.')[0]) for f in onlyfiles]
uvD_idx_list.sort()
print(uvD_idx_list)
print("Number of processed files with uvD GT:", len(uvD_idx_list))
print("Therefore, the GT of the 1st picture is missing. I.e., the file " + os.path.join(uvD_directory, "1.csv") + " does not exist.", "\n")

example_id = 100

fig = plt.figure()
n_rows = 2
n_columns = 3
i=1

# Show original RGB
ax = fig.add_subplot(n_rows,n_columns,i)
image_path = os.path.join(root_dir, 'seq', 'frame_' + str(example_id).zfill(3) + '.png')
print('Showing', image_path)
print('Picture size:', functions_data_processing.get_picture_size(verbose=0, image_path=image_path), "\n")
functions_plot.show_2D_animation_frame(image_path=image_path, show=0, ax=ax)
plt.title('Original RGB')

# Show processed RGB
i+=1
ax = fig.add_subplot(n_rows,n_columns,i)
image_path = os.path.join(root_dir, 'processed', 'color', str(example_id) + '.png')
print('Showing', image_path)
print('Picture size:', functions_data_processing.get_picture_size(verbose=0, image_path=image_path), "\n")
functions_plot.show_2D_animation_frame(image_path=image_path, show=0, ax=ax)
plt.title('Backgd to black, resize 224x224')
# uv = forget_depth(uvD=uvD)
# ax.scatter(uv[:,0], uv[:,1], marker='o', s=marker_size)
# plt.title("uv")

# Loading uvD from processed images
filename = os.path.join(root_dir, 'processed', 'vertexs', str(example_id) + '.csv')
print('Showing', filename, "\n")
uvD = np.genfromtxt(filename, delimiter=',') # numpy array of shape (81, 3)

# Plot uv on processed RGB
i+=1
ax = fig.add_subplot(n_rows,n_columns,i)
functions_plot.show_2D_animation_frame(image_path=image_path, show=0, ax=ax)
uv = functions_data_processing.forget_depth(uvD=uvD)
ax.scatter(uv[:,0], uv[:,1], marker='o', s=marker_size)
plt.title('Processed RGB w/ uv')

# Plot uvD
i+=1
ax = fig.add_subplot(n_rows,n_columns,i, projection='3d')
print("On example " + str(example_id) + ":")
print('depth_min:', min(uvD[:,2]), ' - depth_max:', max(uvD[:,2]))
kwargs_axis_lim={'xmin':0, 'xmax':223, 'ymin':0, 'ymax':223, 'zmin':min(uvD[:,2]), 'zmax':max(uvD[:,2])}
xmin, ymin, zmin, xmax, ymax, zmax, fig, ax= functions_plot.plot_3D_vertices(
    X=uvD[:,0], Y=uvD[:,1], Z=uvD[:,2], sequence_name=sequence_name, dataset_number=dataset_number, marker_size = 25/9, swap_axes=1, ax=ax, fig=fig, title='uv + depth', X_label='u', Y_label='v', Z_label='Depth', **kwargs_axis_lim)

#### Plot triangular faces of mesh
# functions_plot.plot_vertices_and_edges_6edgeColours_from_vertexCoord(X=uvD[:,0], Y=uvD[:,1], Z=uvD[:,2], colour='b', ax=ax, swap_axes=0)
functions_plot.show_mesh_triangular_faces_from_coord(X=uvD[:,0], Y=uvD[:,1], Z=uvD[:,2],
                          submesh_num_vertices_vertical=9,
                          submesh_num_vertices_horizontal=9,
                          xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
                          swap_axes = 1, plot_in_existing_figure=1, fig=fig, 
                          ax=ax, colour='b', line_width=0.1, transparency=0.05)
# Hide grid lines
ax.grid(False)
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')
plt.title('')





# # I think this file may contain the RBG values of each 3D vertex, since (640, 480) is the size of the original pictures
# filename = os.path.join(root_dir, 'xyz', 'frame_001.mat')
# mat = scipy.io.loadmat(filename)
# xyz = mat['XYZ'] # numpy array of shape (480, 640, 3)
# print(np.min(xyz), np.max(xyz))
# print(xyz[:,:,2])

# Maximise figure window
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()