###
### Imports
###

import argparse
import imageio
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from numpy import genfromtxt
import pandas as pd
from PIL import Image
from skimage import io
import scipy.io
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
import functions_train
import submesh
        
def plot_faces_from_vertexCoord(X, Y, Z, faces, colour, swap_axes, ax):
    """
    3D plot of the faces of the meshes
    """
    if swap_axes==1:
        X, Y, Z = X, Z, -Y
    # print(type(X))
    # print(X.shape)

    for face in faces:
        x = X[face]
        y = Y[face]
        z = Z[face]
    #     print(type(x))
    #     print(x)

        verts = [list(zip(x, y, z))]
    #     print(verts)
    #     print(type(verts))
        poly3d = Poly3DCollection(verts, linewidths=1)
        poly3d.set_alpha(0.05) # if you don't set alpha individually and before settin facecolor, it doesn't work
        poly3d.set_facecolor(colour)
        ax.add_collection3d(poly3d)

        ax.add_collection3d(Line3DCollection(verts, colors=colour, linewidths=0.2, linestyles=':', alpha=1))

    if swap_axes==0:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    elif swap_axes==1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

def plot_faces(tensor_of_coordinates, faces, colour, swap_axes=0, sample_idx_within_batch=0):
    """
    3D plot of the faces of the meshes
    (The prediction on the 5356 vertices is awful)
    At the moment, it only works if num_selected_vertices==5356

    tensor_of_coordinates: tensor of shape 
    - batch_size x num_selected_vertices x 3, 
        if there are only the coordinates of the vertices.
    - batch_size x (num_selected_vertices*2) x 3, 
        if there are the coordinates of the vertices followed by those of the normal vectors.
    """
    X = tensor_of_coordinates[sample_idx_within_batch, :, 0].numpy()
    Y = tensor_of_coordinates[sample_idx_within_batch, :, 1].numpy()
    Z = tensor_of_coordinates[sample_idx_within_batch, :, 2].numpy()
    plot_faces_from_vertexCoord(X, Y, Z, faces, colour, swap_axes)
    
def connectpoints(x,y,z,p1,p2):
    """Plot a line segment between the p1-th and the p2-th point in the 
    3d coordinate list given by x,y,z"""
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1, z2 = z[p1], z[p2]
    plt.plot([x1,x2],[y1,y2],[z1,z2])       

def plot_vertices_and_edges(tensor_of_coordinates, colour, swap_axes=0, sample_idx_within_batch=0):
    """
    Plot ground truth and prediction vertex position and edges connecting them in separate figures.
    At the moment, it only works if num_selected_vertices==6.

    tensor_of_coordinates: tensor of shape 
    - batch_size x num_selected_vertices x 3, 
        if there are only the coordinates of the vertices.
    - batch_size x (num_selected_vertices*2) x 3, 
        if there are the coordinates of the vertices followed by those of the normal vectors.

    swap_axes=1 means performing an axis swap so that the camera coordinates plot look like the world coordinates."""
    X = tensor_of_coordinates[sample_idx_within_batch, :, 0].numpy()
    Y = tensor_of_coordinates[sample_idx_within_batch, :, 1].numpy()
    Z = tensor_of_coordinates[sample_idx_within_batch, :, 2].numpy()
    if swap_axes==1:
        X, Y, Z = X, Z, -Y
    # print(type(X))
    # print(X.shape)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c=colour, marker='o', s=50)

    # Plot edges between the 6 vertices
    # Run plot_vertices_and_edges_in_order.py to see the order of the 6 vertices
    # The following list contains the order in which the vertices should be connected
#         For more info on this, see my code My_Python_code/plot_line_segments_in_2D_and_3D.ipynb
    vertices_to_2wise_connect = [0, 4, 1, 3, 5, 2, 0]
    for i in range(len(vertices_to_2wise_connect)-1):
        connectpoints(X,Y,Z,vertices_to_2wise_connect[i],vertices_to_2wise_connect[i+1])

    if swap_axes==0:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    elif swap_axes==1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

#     fig.savefig('Prediction_plots/' + figure_name + '.png')
        
def connectpoints_specific_colour(x,y,z,p1,p2, colour = 'k', line_width = 0.2):
    """Plot a line segment between the p1-th and the p2-th point in the 
    3d coordinate list given by x,y,z.
    You can specify the colour of the segment."""
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1, z2 = z[p1], z[p2]
    handle, = plt.plot([x1,x2],[y1,y2],[z1,z2], colour, linewidth = line_width)       
    return handle
        
def plot_vertices_and_edges_6edgeColours_from_vertexCoord(X, Y, Z, colour, ax, swap_axes=0):
    """
  swap_axes=1 means performing an axis swap so that the camera coordinates plot look like the world coordinates.
  """
    if swap_axes==1:
        X, Y, Z = X, Z, -Y
    # print(type(X))
    # print(X.shape)

    ax.scatter(X, Y, Z, c=colour, marker='o', s=50)

    # Plot edges between the 6 vertices
    # Run plot_vertices_and_edges_in_order.py to see the order of the 6 vertices
    # The following list contains the order in which the vertices should be connected
#         For more info on this, see my code My_Python_code/plot_line_segments_in_2D_and_3D.ipynb
    colors_for_edges = ['b', 'r', 'y', 'k', 'm', 'g']
    #     b : blue.
    #     g : green.
    #     r : red.
    #     c : cyan.
    #     m : magenta.
    #     y : yellow.
    #     k : black.
    #     w : white.
    vertices_to_2wise_connect = [0, 4, 1, 3, 5, 2, 0]
    for i in range(len(vertices_to_2wise_connect)-1):
        connectpoints_specific_colour(X,Y,Z,vertices_to_2wise_connect[i],
                                      vertices_to_2wise_connect[i+1], colour=colors_for_edges[i])
    if swap_axes==0:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    elif swap_axes==1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')

#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])

def plot_vertices_and_edges_6edgeColours(tensor_of_coordinates, colour, swap_axes=0, sample_idx_within_batch=0, ax=None):
    """
   swap_axes=1 means performing an axis swap so that the camera coordinates plot look like the world coordinates.
   """
    X = tensor_of_coordinates[sample_idx_within_batch, :, 0].numpy()
    Y = tensor_of_coordinates[sample_idx_within_batch, :, 1].numpy()
    Z = tensor_of_coordinates[sample_idx_within_batch, :, 2].numpy()
    plot_vertices_and_edges_6edgeColours_from_vertexCoord(X, Y, Z, colour=colour, swap_axes=swap_axes, ax=ax)

def create_GIF(fig, GIF_name, degrees_of_each_rotation, crop_centre_or_ROI=0, full_GIF_name=None, ax=None, delay=0, fps=1):
    import imageio
    llista=[]
    # rotate the axes and update
    for angle in range(0, 360, degrees_of_each_rotation):
#         ax.view_init(30, angle)
        ax.view_init(0, angle + delay)
#                 plt.draw()
#                 plt.pause(.001)
        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        llista.append(image)
    kwargs_write = {'fps':fps, 'quantizer':'nq'}
    if full_GIF_name is None:
        imageio.mimsave('GIF/' + GIF_name +'.gif', llista, fps=fps)
    else:
        imageio.mimsave(full_GIF_name, llista, fps=fps)
        

def show_image_batch(sample_batched):
    """Show the transformed 2D image for a batch of samples, unnormalizeing the colours."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

#     print(type(images_batch))
#     print(images_batch.shape)

    # Unnormalize the pictures to plot them
    images_batch = list(map(data_loading.unnormalize_RGB_of_CHW_tensor, images_batch))

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
def show_unnormalized_image_batch(sample_batched):
    """Show the transformed 2D image for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

#     print(type(images_batch))
#     print(images_batch.shape)

    # Prepare format for grid
    images_batch = list(images_batch)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
#     plt.show()

def show_mesh_batch(sample_batched, swap_axes=1):
    """Show the first element of a batch of 3D meshes"""
    mesh_batch = sample_batched['Vertex_coordinates']
#     print(mesh_batch.shape)
    for i in range(mesh_batch.shape[0]):
        X = mesh_batch[i, :, 0]
        Y = mesh_batch[i, :, 1]
        Z = mesh_batch[i, :, 2]
        
        if swap_axes==1:
            X, Y, Z = X, Z, -Y

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X, Y, Z, c='b', marker='o', s=50)

        if swap_axes==0:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        elif swap_axes==1:
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('-Y')

        xmin, xmax = plt.xlim() # return the current xlim
        ymin, ymax = plt.ylim() 
        zmin, zmax = ax.set_zlim() 
#         print('xlim:', xmin, xmax)
#         print('ylim:', ymin, ymax)
#         print('zlim:', zmin, zmax)

        #ax.set_aspect(aspect = 'equal')

    plt.show()
    
    return xmin, xmax, ymin, ymax, zmin, zmax

# Helper function to show rectangular faces of the first element of a batch of meshes
def show_mesh_faces_batch(sample_batched,
                          submesh_num_vertices_vertical,
                          submesh_num_vertices_horizontal,
                          xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1,
                          sequence_name = 'TowelWall',
                          dataset_number = '2',
                          swap_axes=1):
    """Show the 3D mesh faces for a batch"""
    # 3D plot of the faces of the mesh
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    
    figure_list = []
    
    mesh_batch = sample_batched['Vertex_coordinates']
    #     print(mesh_batch.shape)
    for i in range(mesh_batch.shape[0]):
        X = mesh_batch[i, :, 0]
        Y = mesh_batch[i, :, 1]
        Z = mesh_batch[i, :, 2]

        if swap_axes==1:
            X, Y, Z = X, Z, -Y

        fig = plt.figure()
        ax = Axes3D(fig)
        # print(type(X))
        # print(X.shape)

        # Load the face file.
        # Each face is represented by 4 numbers in a row.
        # Each of these numbers represent a vertex.
        # Vertices are ordered by row in any file of the form 'RendersTowelWall/vertices_*****.txt'
        # There is no heading, so I will import it directly as a numpy array, rather than as a panda DataFrame
        filename = 'Renders' + sequence_name + dataset_number + '/faces_submesh_'
        filename += str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + '.txt'    
        faces = genfromtxt(filename, delimiter=' ')
        # print(type(faces))
        # print(faces[0:5, :])
        faces = faces.astype(int)
        # print(faces[0:5, :])

        for face in faces:
            x = X[face]
            y = Y[face]
            z = Z[face]
        #     print(type(x))
        #     print(x)

            verts = [list(zip(x, y, z))]
    #         print(verts)
    #         print(type(verts))
            poly3d = Poly3DCollection(verts, linewidths=1)
            poly3d.set_alpha(0.05) # if you don't set alpha individually and before settin facecolor, it doesn't work
            poly3d.set_facecolor('b')
            ax.add_collection3d(poly3d)

            ax.add_collection3d(Line3DCollection(verts, colors='b', linewidths=0.2, linestyles=':'))

        if swap_axes==0:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        elif swap_axes==1:
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('-Y')
#         ax.set_xlim([xmin, xmax])
#         ax.set_ylim([ymin, ymax])
#         ax.set_zlim([zmin, zmax])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        #ax.set_aspect(aspect = 'equal')
        title = 'Faces of mesh in ' + sequence_name + '_' + dataset_number
        title += ' - ' + str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + 'mesh'
        plt.title(title)
        figure_list.append(ax)

#     plt.show()

    # fig.savefig('VisualizationTest/meshFaces_' + sequence_name + dataset_number + '_Group' + group_number + '_frame' + animation_frame + '.png')
    return figure_list 
    
    
# Show triangular faces of a tensor of coordinates of points
def show_mesh_triangular_faces_from_coord(X, Y, Z,
                          submesh_num_vertices_vertical,
                          submesh_num_vertices_horizontal,
                          xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
                          sequence_name = 'TowelWall',
                          dataset_number = '2',
                          swap_axes = 1, plot_in_existing_figure=0, fig=None, 
                          ax=None, colour='b', line_width=0.1, transparency=0.05, triangular_faces=1):
    """Show the 3D mesh faces for a tensor of coordinates of points
    
    If triangular_faces==0, then squared faces are plotted."""
    if swap_axes==1:
        X, Y, Z = X, Z, -Y

    if plot_in_existing_figure==0:
        fig = plt.figure()
        ax = Axes3D(fig)
    # print(type(X))
    # print(X.shape)

    # Load the face file.
    # Each face is represented by 4 numbers in a row.
    # Each of these numbers represent a vertex.
    if sequence_name=='TowelWall':
        # Vertices are ordered by row in any file of the form 'RendersTowelWall/vertices_*****.txt'
        # There is no heading, and I will import it directly as a numpy array, rather than as a panda DataFrame
        filename = 'Renders' + sequence_name + dataset_number + '/faces_submesh_'
        filename += str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + '.txt'    
        faces = genfromtxt(filename, delimiter=' ')
        faces = faces.astype(int)
    else:
        faces=np.array([[submesh.idx_from_matrix_coord((ver, hor), 9), submesh.idx_from_matrix_coord((ver, hor+1), 9),
         submesh.idx_from_matrix_coord((ver+1, hor), 9), submesh.idx_from_matrix_coord((ver+1, hor+1), 9)]
         for ver in range(submesh_num_vertices_vertical-1) for hor in range(submesh_num_vertices_horizontal-1) ], dtype=int)
        
    if triangular_faces==1:
        for i, j in [(0, 3), (1, 4)]:
            for face in faces:
                x = X[face[i:j]]
                y = Y[face[i:j]]
                z = Z[face[i:j]]

                verts = [list(zip(x, y, z))]
                poly3d = Poly3DCollection(verts, linewidths=1)
                poly3d.set_alpha(transparency) # if you don't set alpha individually and before settin facecolor, it doesn't work
                poly3d.set_facecolor(colour)
                ax.add_collection3d(poly3d)

                # Plot line segments connecting consecutive vertices
                #                 ax.add_collection3d(Line3DCollection(verts, colors='k', linewidths=0.2, linestyles=':'))
                # The line above plots boundary of all triangles, but I only want boundary of "squares", so I'll perform differently
                
        # Plot edges
        for face in faces:
            x = X[face]
            y = Y[face]
            z = Z[face]
            # The following list contains the order in which the vertices should be connected
            # For more info on this, see my code My_Python_code/plot_line_segments_in_2D_and_3D.ipynb
            vertices_to_2wise_connect = [0, 1, 3, 2, 0, 3]
            for i in range(len(vertices_to_2wise_connect)-1):
                handle = connectpoints_specific_colour(x,y,z,vertices_to_2wise_connect[i],vertices_to_2wise_connect[i+1], colour=colour, line_width = line_width)
    else: 
        for face in faces:
            x = X[face]
            y = Y[face]
            z = Z[face]
        #     print(type(x))
        #     print(x)

            verts = [list(zip(x, y, z))]
    #         print(verts)
    #         print(type(verts))
            poly3d = Poly3DCollection(verts, linewidths=1)
            poly3d.set_alpha(transparency) # if you don't set alpha individually and before settin facecolor, it doesn't work
            poly3d.set_facecolor(colour)
            ax.add_collection3d(poly3d)

        # Plot edges
        for face in faces:
            x = X[face]
            y = Y[face]
            z = Z[face]
            # The following list contains the order in which the vertices should be connected
            # For more info on this, see my code My_Python_code/plot_line_segments_in_2D_and_3D.ipynb
            vertices_to_2wise_connect = [0, 1, 3, 2, 0]
            for i in range(len(vertices_to_2wise_connect)-1):
                handle = connectpoints_specific_colour(x,y,z,vertices_to_2wise_connect[i],vertices_to_2wise_connect[i+1], colour=colour, line_width = line_width)

            
            
            


    if swap_axes==0:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    elif swap_axes==1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')

    if xmin is not None:
        if swap_axes==0:
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
        elif swap_axes==1:
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([zmin, zmax])
            ax.set_zlim([ymin, ymax])

    #ax.set_aspect(aspect = 'equal')
    title = 'Faces of mesh in ' + sequence_name + '_' + dataset_number
    title += ' - ' + str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + 'mesh'
    plt.title(title)
    
    if plot_in_existing_figure==0:
        return ax, title, fig
    else:
        return ax, title, handle

# Show triangular faces of a tensor of coordinates of points
def show_mesh_triangular_faces_tensor(tensor_of_coordinates,
                          submesh_num_vertices_vertical,
                          submesh_num_vertices_horizontal,
                          xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
                          sequence_name = 'TowelWall',
                          dataset_number = '2',
                          swap_axes = 1, plot_in_existing_figure=0, fig=None, 
                          ax=None, colour='b', line_width=0.1, transparency=0.05, triangular_faces=1):
    """Show the 3D mesh faces for a tensor of coordinates of points"""
    X = tensor_of_coordinates[:, 0]
    Y = tensor_of_coordinates[:, 1]
    Z = tensor_of_coordinates[:, 2]
    return show_mesh_triangular_faces_from_coord(
        X, Y, Z, submesh_num_vertices_vertical, submesh_num_vertices_horizontal, xmin, xmax, ymin, ymax, zmin, zmax,
        sequence_name, dataset_number, swap_axes, plot_in_existing_figure, fig, ax, colour, line_width, transparency, triangular_faces)

# Helper function to show triangular faces of the first element of a batch of meshes
def show_mesh_triangular_faces_batch(sample_batched,
                          submesh_num_vertices_vertical,
                          submesh_num_vertices_horizontal,
                          xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1,
                          sequence_name = 'TowelWall',
                          dataset_number = '2',
                                     swap_axes = 1, triangular_faces=1):
    """Show the 3D mesh faces for a batch"""
    # 3D plot of the faces of the mesh
    figure_list=[]
    mesh_batch = sample_batched['Vertex_coordinates']
    #     print(mesh_batch.shape)
    for i in range(mesh_batch.shape[0]):
        tensor_of_coordinates = mesh_batch[i, :, :]
        ax = show_mesh_triangular_faces_tensor(tensor_of_coordinates=tensor_of_coordinates,
                          submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                          submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                          sequence_name = sequence_name,
                          dataset_number = dataset_number,
                          swap_axes = swap_axes, triangular_faces=triangular_faces)
        figure_list.append(ax)

    plt.show()

    # fig.savefig('VisualizationTest/meshFaces_' + sequence_name + dataset_number + '_Group' + group_number + '_frame' + animation_frame + '.png')
    return figure_list

def plot_camera_and_vertices(sequence_name, dataset_number, group_number, animation_frame, 
                             X, Y, Z, occlusion_mask_values, camera_x, camera_y, camera_z,
                             xmin=None, ymin=None, zmin=None, xmax=None, ymax=None, zmax=None,
                             swap_axes=0, vertex_size=0.5, title=None):
    """
    3D scatter plot of the vertices coloured by visibility.
    Plot as well camera position.
    
    - X, Y, Z can be either world coordinates or camera coordinates.
    - swap_axes=1 means performing an axis swap so that the camera coordinates plot look like the world coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if swap_axes==1:
        X, Y, Z = X, Z, -Y

    unique = list(set(occlusion_mask_values))
    #colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    colors= ['b', 'y']
    for i, un in enumerate(unique):
        xi = [X[j] for j  in range(len(X)) if occlusion_mask_values[j] == un]
        yi = [Y[j] for j  in range(len(Y)) if occlusion_mask_values[j] == un]
        zi = [Z[j] for j  in range(len(Z)) if occlusion_mask_values[j] == un]

        ax.scatter(xi, yi, zi, c=colors[i], marker='o', s=vertex_size)

    ax.scatter(camera_x, camera_y, camera_z, c='r', marker='o', s=100)

    ax.legend(['Visible', 'Occluded', 'Camera'], loc=3)

    if swap_axes==0:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    elif swap_axes==1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')
        
    if xmin!=None:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

    #ax.set_aspect(aspect = 'equal')
    
    if title==None:
        title = 'Vertex visibility of mesh in ' + sequence_name + '_' + dataset_number
        title += '\nGroup ' + group_number + ' - Animation frame ' + animation_frame
    if swap_axes==1:
        title=title+'_swapAxes'
        
    plt.title(title)

    plt.show()
    
#     figure_name = 'VisualizationTest/vertex_visibility_wCamera' + sequence_name + dataset_number + '_Group' + group_number + '_frame' + animation_frame
#     if swap_axes==1:
#         figure_name=figure_name+'_swapAxes'
#     fig.savefig(figure_name + '.png')

def plot_normal_vectors(sequence_name, dataset_number, group_number, animation_frame, 
                        X, Y, Z, nX, nY, nZ, occlusion_mask_values,
                        camera_x, camera_y, camera_z, 
                        xmin=None, ymin=None, zmin=None, xmax=None, ymax=None, zmax=None,
                        plot_camera=1, swap_axes=0, linewidths=1, length=1, headwidth=3):
    """
    - swap_axes=1 means performing an axis swap so that the camera coordinates plot look like the world coordinates.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if swap_axes==1:
        X, Y, Z = X, Z, -Y
        nX, nY, nZ = nX, nZ, -nY
        
#     ax.quiver(X, Y, Z, nX, nY, nZ, length=0.4, normalize=True, linewidths = 0.2)
    ax.quiver(X, Y, Z, nX, nY, nZ, length=length, normalize=True, linewidths = linewidths)


    if swap_axes==0:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    elif swap_axes==1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')

    if xmin!=None:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
    
    if plot_camera==1:
        ax.scatter(camera_x, camera_y, camera_z, c='r', marker='o', s=100)
        
    ax.view_init(0, -90)

    #ax.set_aspect(aspect = 'equal')

    plt.title('Normal vectors of mesh in ' + sequence_name + '_' + dataset_number + '\nGroup ' + group_number + ' - Animation frame ' + animation_frame)

    plt.show()

#     fig.savefig('VisualizationTest/meshNormals_' + sequence_name + dataset_number + '_Group' + group_number + '_frame' + animation_frame + '.png')

    return fig
   
def show_2D_animation_frame(sequence_name=None, dataset_number=None, group_number=None, animation_frame=None, image_path=None, show=1, ax=None):
    if ax is None:
        plt.figure()
    if image_path is None:
        image_path = os.path.join('Renders' + sequence_name + dataset_number, 'Group.' + group_number, str(int(animation_frame)) + '.png') 
    image = io.imread(image_path)
    plt.imshow(image)
    if show==1:
        plt.show()
    
def plot_RGB_and_landmarks_ax_as_input(u_visible, v_visible, ax, u_occluded=None, v_occluded=None,
                          sequence_name=None, dataset_number=None, group_number=None, animation_frame=None,
                          image_path=None, title=None, marker_size=25, image=None, binary=0, visible_colour='b', axis_on='on',
                                       show_vertices=1):
    """
    Plot the landmarks on top of the RGB on a given axis subplot

    2D scatter plot of the pixel location of each vertex
    Using both visible and occluded vertices
    """
    if show_vertices==1:
        # print(u.size == v.size)
        ax.scatter(u_visible, v_visible, c=visible_colour, marker='o', s=marker_size) # CAVEAT: I had to use '-v'
        if not(u_occluded is None):
            ax.scatter(u_occluded, v_occluded, c='y', marker='o', s=marker_size) # CAVEAT: I had to use '-v'

        plt.gca().invert_yaxis() # the origin of the (u, v) coordinates is top-left
        plt.axis(axis_on)

        if not(u_occluded is None):
            ax.legend(['Visible', 'occluded'], loc=3)

        ax.set_xlabel('u')
        ax.set_ylabel('v')

        ax.set_aspect(aspect = 'equal')

    if image is None:
        if image_path is None:
            if title is None:
                title='Pixel location of each vertex\n' + sequence_name + '_' + dataset_number 
                title+= ' - Group ' + group_number + ' - Animation frame ' + animation_frame
                plt.title(title)        
            image_path = os.path.join('Renders' + sequence_name + dataset_number, 'Group.' + group_number)
            image_path = os.path.join(image_path, str(int(animation_frame)) + '.png') 
        image = io.imread(image_path)
    if title is not None:
        plt.title(title)        
    # Keep the next lines. Otherwise, plot_RGB_and_landmarks_from_dataset does not plot the RGB image
#     plt.imshow(image)
    if binary==0:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
    return ax # Keep this line. Otherwise, uv_RGB_match_check.py does not work

def plot_RGB_and_landmarks_ax_as_input_GT_and_pred(u_pred, v_pred, ax, u_GT=None, v_GT=None,
                          sequence_name=None, dataset_number=None, group_number=None, animation_frame=None,
                          image_path=None, title=None, marker_size=25, image=None, binary=0, annotate=0, show_vertices=1):
    """
    Plot the landmarks on top of the RGB on a given axis subplot, for both prediction and Ground Truth

    2D scatter plot of the pixel location of each vertex
    """
    if show_vertices==1:
        # print(u.size == v.size)
        if u_GT is not None:
            ax.scatter(u_GT, v_GT, c='b', marker='o', s=marker_size) # CAVEAT: I had to use '-v'
        ax.scatter(u_pred, v_pred, c='y', marker='o', s=marker_size) # CAVEAT: I had to use '-v'
        if annotate==1:
            for i, txt in enumerate(range(len(u_pred))):
                ax.annotate(txt, (u_pred[i], v_pred[i]), size=15)

        plt.gca().invert_yaxis() # the origin of the (u, v) coordinates is top-left

        if not(u_GT is None):
            ax.legend(['Ground Truth', 'Prediction'], loc=3)

        ax.set_xlabel('u')
        ax.set_ylabel('v')

        ax.set_aspect(aspect = 'equal')

    if image is None:
        if image_path is None:
            title='Pixel location of each vertex\n' + sequence_name + '_' + dataset_number 
            title+= ' - Group ' + group_number + ' - Animation frame ' + animation_frame
            plt.title(title)        
            image_path = os.path.join('Renders' + sequence_name + dataset_number, 'Group.' + group_number)
            image_path = os.path.join(image_path, str(int(animation_frame)) + '.png') 
        image = io.imread(image_path)
    if title!=None:
        plt.title(title)       
        
    # Keep thie next lines. Otherwise, plot_RGB_and_landmarks_from_dataset does not plot the RGB image
#     plt.imshow(image)
    if binary==0:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
    return ax # Keep this line. Otherwise, uv_RGB_match_check.py does not work

def plot_RGB(ax, image_path=None, marker_size=25, image=None, binary=0, annotate=0, axis_on='on'):
    """
    Plot the RGB on a given axis subplot
    """
    if image is None:
        if image_path is None:
            title='Pixel location of each vertex\n' + sequence_name + '_' + dataset_number 
            title+= ' - Group ' + group_number + ' - Animation frame ' + animation_frame
            plt.title(title)        
            image_path = os.path.join('Renders' + sequence_name + dataset_number, 'Group.' + group_number)
            image_path = os.path.join(image_path, str(int(animation_frame)) + '.png') 
        image = io.imread(image_path)    
    plt.axis(axis_on)
        
    # Keep thie next lines. Otherwise, plot_RGB_and_landmarks_from_dataset does not plot the RGB image
#     plt.imshow(image)
    if binary==0:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
    return ax # Keep this line. Otherwise, uv_RGB_match_check.py does not work

def plot_RGB_and_landmarks(u_visible, v_visible, u_occluded=None, v_occluded=None,
                          sequence_name=None, dataset_number=None, group_number=None, animation_frame=None,
                          image_path=None, title=None, marker_size=25, image=None, binary=0, visible_colour='b',
                          axis_on='on'):
    """
    Plot the landmarks on top of the RGB

    2D scatter plot of the pixel location of each vertex
    Using both visible and occluded vertices
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax = plot_RGB_and_landmarks_ax_as_input(u_visible, v_visible, ax, u_occluded, v_occluded,
                          sequence_name, dataset_number, group_number, animation_frame,
                          image_path, title, marker_size, image, binary, visible_colour, axis_on)
    return fig # Keep this line. Otherwise, uv_RGB_match_check.py does not work
    
def plot_RGB_and_landmarks_from_dataset(
    dataset, observation_within_dataset, transformed_image_or_not=0, uv_normalization=0, binary=0):
    sample = dataset[observation_within_dataset]
    if transformed_image_or_not==0:
        loaded_image_wo_transforms_shape = functions_data_processing.get_picture_size(verbose=0, image_path=sample['img_name'])
        width=loaded_image_wo_transforms_shape[1] # width for the uv normalization
        # width = 960 for uncropped images. Smaller for cropped images.
        height=loaded_image_wo_transforms_shape[0] # height for the uv normalization
        # height = 540 for uncropped images. Smaller for cropped images.
#         print('width, height=', width, height)
        im = Image.open(sample['img_name'])
    else: # if resize to 224x224 was on
        im = sample['image']
#         print('channels, height, width:', im.shape)
        height = im.shape[1]
        width = im.shape[2]
#         print('width, height=', width, height)
        im = data_loading.unnormalize_RGB_of_CHW_tensor(im)
        im = transforms.ToPILImage()(im).convert("RGB")
        
    uv = sample['uv']
    if uv_normalization==1:
        uv = functions_data_processing.unnormalize_uv_01(uv=uv, width=width, height=height)
    else:
#         uv = uv.astype(int)
        uv = functions_data_processing.unnormalize_uv_01(uv=uv, width=width/functions_data_processing.get_picture_size(verbose=0, image_path=sample['img_name'])[1], height=height/functions_data_processing.get_picture_size(verbose=0, image_path=sample['img_name'])[0])
        
    u=uv[:,0]
    v=uv[:,1]
    plot_RGB_and_landmarks(u_visible=u, v_visible=v, image=im, binary=binary)
    
def plot_a_fixed_list_images_randomly_transformed(transformed_dataset):
    if transformed_dataset.transform:
        imglist = [transformed_dataset[i]['image'] for i in [0, 20, 40, 60, 80]]
    #     print(type(imglist))
    #     print(len(imglist))
    #     print(imglist[0])
        plt.figure()
        grid = utils.make_grid(imglist)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('A fixed list of normalized transformed images')
        plt.axis('off')
        plt.ioff()
        plt.show()

        imglist = list(map(data_loading.unnormalize_RGB_of_CHW_tensor, imglist))
        plt.figure()
        grid = utils.make_grid(imglist)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('A fixed list of unnormalized transformed images')
        plt.axis('off')
        plt.ioff()
        plt.show()
    else:
        i = 0

        while i < len(transformed_dataset):
            sample = transformed_dataset[i]

            plt.figure()
            plt.imshow(sample['image']) 
            group_number, animation_frame = group_and_frame_from_idx(i)
            plt.title('Transformed image. Group no.: ' + str(group_number) + '. Frame = ' + str(animation_frame))
            plt.show()

            i += 20

            if i > 98:
        #         plt.show()
                break
            
def plot_uv_on_RGB(u_visible, v_visible, u_occluded, v_occluded,
                  sequence_name=None, dataset_number=None, group_number=None, animation_frame=None):
    """2D scatter plot of the pixel location of each vertex.
    Using both visible and occluded vertices."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(u_visible, v_visible, c='b', marker='o', s=50) # CAVEAT: I had to use '-v'
    ax.scatter(u_occluded, v_occluded, c='y', marker='o', s=50) # CAVEAT: I had to use '-v'

    plt.gca().invert_yaxis() # the origin of the (u, v) coordinates is top-left

    ax.legend(['Visible', 'occluded'], loc=3)

    ax.set_xlabel('u')
    ax.set_ylabel('v')

    if not(sequence_name is None):
        plt.title('Pixel location of each vertex\n' + sequence_name + '_' + dataset_number + ' - Group ' + group_number + ' - Animation frame ' + animation_frame)

    ax.set_aspect(aspect = 'equal')
    
def create_title(sequence_name, dataset_number, group_number, animation_frame):
    title = 'Vertices of mesh in ' + sequence_name + '_' + dataset_number
    if (group_number is not None) and (animation_frame is not None):
        title += '\nGroup ' + group_number + ' - Animation frame ' + animation_frame
    return title
    
def plot_3D_vertices(X, Y, Z, sequence_name, dataset_number, group_number=None, 
                     animation_frame=None, marker_size = 50, swap_axes=0,
                     xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, 
                     ax=None, fig=None, title=None, colour='b', verbose=0,
                    X_label='X', Y_label='Y', Z_label='Z'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    if swap_axes==1:
        X, Y, Z = X, Z, -Y
    ax.scatter(X, Y, Z, c=colour, marker='o', s=marker_size)

    if swap_axes==0:
        ax.set_xlabel(X_label)
        ax.set_ylabel(Y_label)
        ax.set_zlabel(Z_label)
        if xmin is not None:
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
    elif swap_axes==1:
        ax.set_xlabel(X_label)
        ax.set_ylabel(Z_label)
        ax.set_zlabel('-'+Y_label)
        if xmin is not None:
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([zmin, zmax])
            ax.set_zlim([-ymax, -ymin])
    
    ax.view_init(0, -90)

    if title is None:
        title = create_title(sequence_name, dataset_number, group_number, animation_frame)
    plt.title(title)
    
    xmin, xmax = plt.xlim() # return the current xlim
    ymin, ymax = plt.ylim() 
    zmin, zmax = ax.set_zlim() 
    if verbose==1:
        print('xlim:', xmin, xmax)
        print('ylim:', ymin, ymax)
        print('zlim:', zmin, zmax)
    # fig.savefig('VisualizationTest/vertices_' + sequence_name + dataset_number + '_Group' + group_number + '_frame' + animation_frame + '.png')
    
    return xmin, ymin, zmin, xmax, ymax, zmax, fig, ax

def range_of_obs(args):
    if args.elt_within_batch is None:
        range_of_observations = range(args.batch_size_to_show)
    else:
        range_of_observations = [args.elt_within_batch]
    return range_of_observations

def visualize_uv_prediction(args, labels, outputs, sample_batched_img_name, model_directory, show_vertices=1):
    range_of_observations = range_of_obs(args)
    for sample_idx_within_batch in range_of_observations:    
        if show_vertices==1:
    #         for sample_idx_within_batch in [0]:            
            # Shape back to args.batch_size x args.num_selected_vertices x args.num_coord_per_vertex
            labels = functions_data_processing.reshape_labels_back(
                labels, args.batch_size, args.num_selected_vertices, num_coord_per_vertex=2) 
            u = labels[sample_idx_within_batch, :, 0].numpy()
            v = labels[sample_idx_within_batch, :, 1].numpy()
            outputs = functions_data_processing.reshape_labels_back(
                outputs, args.batch_size, args.num_selected_vertices, num_coord_per_vertex=2) 
            u_outputs = outputs[sample_idx_within_batch, :, 0].numpy()
            v_outputs = outputs[sample_idx_within_batch, :, 1].numpy()

            if args.crop_centre_or_ROI==2: # rectangular crop
                # Get group_number and animation frame from image_name
                image_name = sample_batched_img_name[sample_idx_within_batch] 
                # E.g., 'RendersTowelWall7/Group.004/70_rectROI.png'
                group_number = image_name.split('Group')[1][1:4]
                animation_frame = image_name.split(group_number)[1][1:]
                animation_frame = animation_frame.split('_')[0]

                directory_name = os.path.join('Renders' + args.sequence_name + args.dataset_number, 'Group.' + group_number)
                text_name = os.path.join(directory_name, 'uv_width_height_rectROI_' + animation_frame +'.txt')
                uvwh = np.genfromtxt(fname=text_name, dtype='int', delimiter=' ', skip_header=1) 
                u_crop_corner = uvwh[0]
                v_crop_corner = uvwh[1]
                width_crop = uvwh[2]
                height_crop = uvwh[3]

            if args.uv_normalization==1:
                if args.crop_centre_or_ROI==2: # rectangular crop
                    u, v = functions_data_processing.unnormalize_uv_01(u=u, v=v, width=width_crop, height=height_crop) 
                    u_outputs, v_outputs = functions_data_processing.unnormalize_uv_01(
                        u=u_outputs, v=v_outputs, width=width_crop, height=height_crop) 
                else:
                    u, v = functions_data_processing.unnormalize_uv_01(u=u, v=v) 
                    u_outputs, v_outputs = functions_data_processing.unnormalize_uv_01(u=u_outputs, v=v_outputs) 

            if args.crop_centre_or_ROI==2: # rectangular crop
                u += u_crop_corner
                v += v_crop_corner
                u_outputs += u_crop_corner
                v_outputs += v_crop_corner

        image_path = sample_batched_img_name[sample_idx_within_batch]
        if args.crop_centre_or_ROI!=3: # If there is some crop
            # Set as image_path that of the uncropped image
            image_path=image_path.split('_rectROI')
            image_path = image_path[0]+image_path[1]

        # CAVEAT!
        # In this first version, I'm gonna plot both visible and occluded as visible
        if show_vertices==1:
            plot_RGB_and_landmarks(u_visible=u, v_visible=v, 
                                                  image_path=image_path,
                                                  title='Ground Truth')
            plot_RGB_and_landmarks(u_visible=u_outputs, v_visible=v_outputs, 
                                                  image_path=image_path,
                                                  title='Predicted uv')
        else:
            pass
#             plot_RGB()
        plt.show()

# def plot_grid_uv_on_RGB(u_visible, v_visible, image_path, title):
#     """ visualize_uv_prediction_grid works and is based on this, but plot_grid_uv_on_RGB is just the idea of how to code visualize_uv_prediction_grid, and plot_grid_uv_on_RGB itself is not expected to be run """
#     fig = plt.figure()
#     if args.batch_size_to_show == 12:
#         n_rows = 3
#         n_columns = 4
#         for sample_idx_within_batch in range(args.batch_size_to_show):       
#             i = sample_idx_within_batch + 1
#             ax = fig.add_subplot(n_rows,n_columns,i)
#             ax = plot_RGB_and_landmarks_ax_as_input(u_visible=u_outputs, v_visible=v_outputs, ax=ax,
#                                               image_path=image_path,
#                                               title='Predicted uv')
# #             ax.imshow(...)

def visualize_uv_prediction_grid(args, labels, outputs, sample_batched_img_name, model_directory, plot_on_input=0, sample_batched=None, save_png=0):
    """If plot_on_input==0, the predictions are plot on the original RGB. 
    If plot_on_input==1, the predictions are plot on the transformed RGB (after potential crop, resize and normalization of colours). 
    If plot_on_input==2, the predictions are plot on the transformed RGB (after potential crop, resize and normalization of colours), and the predicted and ground truth 0<=u,v<=1 are multiplied by the width and height of their corresponding crop.
    
    If save_png==1, a png will be saved.
    """
    range_of_observations = range_of_obs(args)
#     if args.batch_size_to_show == 12:
    if True:
        if args.grid==1:
            fig = plt.figure()
            n_rows = 3
            n_columns = 4
        # Shape back to args.batch_size x args.num_selected_vertices x args.num_coord_per_vertex
        labels = functions_data_processing.reshape_labels_back(
            labels, args.batch_size, args.num_selected_vertices, num_coord_per_vertex=2) 
        outputs = functions_data_processing.reshape_labels_back(
            outputs, args.batch_size, args.num_selected_vertices, num_coord_per_vertex=2) 
        if args.test_uv_unnorm_batch==1:
            labels=functions_data_processing.unnormalize_uv_01_tensor_batch(labels, sequence_name=args.sequence_name)
            outputs=functions_data_processing.unnormalize_uv_01_tensor_batch(outputs, sequence_name=args.sequence_name)
        for sample_idx_within_batch in range_of_observations:    
    #         for sample_idx_within_batch in [0]:            
            u = labels[sample_idx_within_batch, :, 0].numpy()
            v = labels[sample_idx_within_batch, :, 1].numpy()
            u_outputs = outputs[sample_idx_within_batch, :, 0].numpy()
            v_outputs = outputs[sample_idx_within_batch, :, 1].numpy()

            if plot_on_input in [0,2]:
                if args.crop_centre_or_ROI==2: # rectangular crop
                    # Get group_number and animation frame from image_name
                    image_name = sample_batched_img_name[sample_idx_within_batch] 
                    # E.g., 'RendersTowelWall7/Group.004/70_rectROI.png'
                    group_number = image_name.split('Group')[1][1:4]
                    animation_frame = image_name.split(group_number)[1][1:]
                    animation_frame = animation_frame.split('_')[0]

                    directory_name = os.path.join('Renders' + args.sequence_name + args.dataset_number, 'Group.' + group_number)
                    text_name = os.path.join(directory_name, 'uv_width_height_rectROI_' + animation_frame +'.txt')
                    uvwh = np.genfromtxt(fname=text_name, dtype='int', delimiter=' ', skip_header=1) 
                    u_crop_corner = uvwh[0]
                    v_crop_corner = uvwh[1]
                    width_crop = uvwh[2]
                    height_crop = uvwh[3]

                if args.test_uv_unnorm_batch!=1:
                    if args.uv_normalization in [1,2]:
                        if args.crop_centre_or_ROI==2: # rectangular crop
                            if plot_on_input==0:
                                u, v = functions_data_processing.unnormalize_uv_01(u=u, v=v, width=width_crop, height=height_crop) 
                                u_outputs, v_outputs = functions_data_processing.unnormalize_uv_01(
                                    u=u_outputs, v=v_outputs, width=width_crop, height=height_crop) 
                            elif plot_on_input==2:
                                u, v = functions_data_processing.unnormalize_uv_01(u=u, v=v, width=224, height=224) 
                                u_outputs, v_outputs = functions_data_processing.unnormalize_uv_01(
                                    u=u_outputs, v=v_outputs, width=224, height=224) 
                        else:
                            u, v = functions_data_processing.unnormalize_uv_01(u=u, v=v, sequence_name=args.sequence_name) 
                            u_outputs, v_outputs = functions_data_processing.unnormalize_uv_01(u=u_outputs, v=v_outputs, sequence_name=args.sequence_name) 
                    if args.uv_normalization==2: 
                        u, v = functions_data_processing.normalize_uv_01(u=u, v=v, width=224, height=224)
                        u_outputs, v_outputs = functions_data_processing.normalize_uv_01(u=u_outputs, v=v_outputs, width=224, height=224)
            
            if plot_on_input == 0:
                if args.crop_centre_or_ROI==2: # rectangular crop
                    u += u_crop_corner
                    v += v_crop_corner
                    u_outputs += u_crop_corner
                    v_outputs += v_crop_corner

                image_path = sample_batched_img_name[sample_idx_within_batch]
                if args.crop_centre_or_ROI!=3: # If there is some crop
                    # Set as image_path that of the uncropped image
                    image_path=image_path.split('_rectROI')
                    image_path = image_path[0]+image_path[1]

            # CAVEAT!
            # In this first version, I'm gonna plot both visible and occluded as visible
#             plot_RGB_and_landmarks(u_visible=u, v_visible=v, 
#                                                   image_path=image_path,
#                                                   title='Ground Truth')
#             plot_RGB_and_landmarks(u_visible=u_outputs, v_visible=v_outputs, 
#                                                   image_path=image_path,
#                                                   title='Predicted uv')
            i = sample_idx_within_batch + 1
            if args.grid==1:
                ax = fig.add_subplot(n_rows,n_columns,i)
            else:
                fig=plt.figure()
                ax = fig.add_subplot(111)    
#             ax = plot_RGB_and_landmarks_ax_as_input(u_visible=u_outputs, v_visible=v_outputs, ax=ax,
#                                               image_path=image_path,
#                                               title='Predicted uv')
            if args.train_or_val==0:
                title='training set'
            if args.train_or_val==1:
                title='validation set'
            if args.train_or_val==2:
                title='test set'
            if args.show_vertices==1:
                if plot_on_input==0:
                    title+= '\npredicted uv on original RGB'
                if plot_on_input==1:
                    title+= '\npredicted uv on transformed RGB'
                if plot_on_input==2:
                    title+= '\nunnormalized predicted uv on transformed RGB'

            if plot_on_input==0:
                ax = plot_RGB_and_landmarks_ax_as_input_GT_and_pred(u_pred=u_outputs, v_pred=v_outputs, ax=ax, 
                                                                u_GT=u, v_GT=v,image_path=image_path,
                                                                title=title, marker_size=50/args.submesh_num_vertices_vertical, 
                                                                   annotate=args.annotate, show_vertices=args.show_vertices)
            elif plot_on_input in [1,2]:
                image = sample_batched['image'][sample_idx_within_batch].numpy().transpose((1, 2, 0))
                ax = plot_RGB_and_landmarks_ax_as_input_GT_and_pred(u_pred=u_outputs, v_pred=v_outputs, ax=ax, 
                                                                u_GT=u, v_GT=v, image=image,
                                                                title=title, marker_size=50/args.submesh_num_vertices_vertical,
                                                                   annotate=args.annotate, show_vertices=args.show_vertices)
    # Maximise figure window
    if args.grid==1:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    if (save_png==1) and (args.save_png_path is not None):
        fig.savefig(args.save_png_path)
# #             plt.get_current_fig_manager().window.showMaximized()
#         fig.set_size_inches(8, 6)
#         # when saving, specify the DPI
# #             plt.savefig("myplot.png", dpi = 100)
#         fig.savefig(args.save_png_path, dpi = 100)

#         plt.show()

###
### Visualize a batch of training images - under some transforms
### (moved here from transfer_learning_TowelWall.py)
### This is old code. I have more updated versions which work perfecly.
###
# if __name__ == '__main__':
    # # Skip this when running on the server
    # # In[ ]:
    # def imshow(inp, title=None):
    #     """Imshow for Tensor."""
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
    #     plt.imshow(inp)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.001)  # pause a bit so that plots are updated
    #    
    # # Get a batch of training data
    # sample_batched = next(iter(dataloaders['train']))
    # inputs = sample_batched['image']
    # labels = sample_batched['Vertex_coordinates'] # Shape batch_size x num_selected_vertices x 3
    # labels = functions_data_processing.reshape_labels(labels, batch_size, num_selected_vertices) # Shape batch_size x (num_selected_vertices * 3)
    # print('labels:', labels)
    #
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    #
    # # Show the grid
    # imshow(out) 
    
def GIF_falling_towel(sequence_name = 'TowelWall', dataset_number = '11', group_number = '003', GIF_name=None):
    llista=[]
    fig = plt.figure()

    for animation_frame in range(1, 100):
        img_name = os.path.join('Renders' + sequence_name + dataset_number, 'Group.' + group_number,
                                str(animation_frame) + '.png')
        image = io.imread(img_name)
        plt.axis('off')
        plt.imshow(image)
        # plt.show()
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        llista.append(image)

    kwargs_write = {'fps':5.0, 'quantizer':'nq'}
    if GIF_name is None:
        GIF_name = 'falling_towel_' + sequence_name + dataset_number + '_Group' + group_number
        GIF_name = os.path.join('GIF_dataset', GIF_name + '.gif')
    imageio.mimsave(GIF_name, llista, fps=5)
    
    
    
    
# Polygonal faces on RGB
def polygonal_faces(sample_batched,
                          submesh_num_vertices_vertical,
                          submesh_num_vertices_horizontal,
                          xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1,
                          sequence_name = 'TowelWall',
                          dataset_number = '2',
                          swap_axes=1):
    """Show the 3D mesh faces for a batch"""
    # 3D plot of the faces of the mesh
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    
    figure_list = []
    
    mesh_batch = sample_batched['Vertex_coordinates']
    #     print(mesh_batch.shape)
    for i in range(mesh_batch.shape[0]):
        X = mesh_batch[i, :, 0]
        Y = mesh_batch[i, :, 1]
        Z = mesh_batch[i, :, 2]

        if swap_axes==1:
            X, Y, Z = X, Z, -Y

        fig = plt.figure()
        ax = Axes3D(fig)
        # print(type(X))
        # print(X.shape)

        # Load the face file.
        # Each face is represented by 4 numbers in a row.
        # Each of these numbers represent a vertex.
        # Vertices are ordered by row in any file of the form 'RendersTowelWall/vertices_*****.txt'
        # There is no heading, so I will import it directly as a numpy array, rather than as a panda DataFrame
        filename = 'Renders' + sequence_name + dataset_number + '/faces_submesh_'
        filename += str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + '.txt'    
        faces = genfromtxt(filename, delimiter=' ')
        # print(type(faces))
        # print(faces[0:5, :])
        faces = faces.astype(int)
        # print(faces[0:5, :])

        for face in faces:
            x = X[face]
            y = Y[face]
            z = Z[face]
        #     print(type(x))
        #     print(x)

            verts = [list(zip(x, y, z))]
    #         print(verts)
    #         print(type(verts))
            poly3d = Poly3DCollection(verts, linewidths=1)
            poly3d.set_alpha(0.05) # if you don't set alpha individually and before settin facecolor, it doesn't work
            poly3d.set_facecolor('b')
            ax.add_collection3d(poly3d)

            ax.add_collection3d(Line3DCollection(verts, colors='b', linewidths=0.2, linestyles=':'))

        if swap_axes==0:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        elif swap_axes==1:
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('-Y')
#         ax.set_xlim([xmin, xmax])
#         ax.set_ylim([ymin, ymax])
#         ax.set_zlim([zmin, zmax])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        #ax.set_aspect(aspect = 'equal')
        title = 'Faces of mesh in ' + sequence_name + '_' + dataset_number
        title += ' - ' + str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + 'mesh'
        plt.title(title)
        figure_list.append(ax)

#     plt.show()

    # fig.savefig('VisualizationTest/meshFaces_' + sequence_name + dataset_number + '_Group' + group_number + '_frame' + animation_frame + '.png')
    return figure_list 

def visualize_xyz_prediction_grid(args, labels, outputs, sample_batched=None, save_png=0, transparency=0.05, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, plot_pred=1, plot_GT=1):
    """    
    If save_png==1, a png will be saved.
    """
    line_width = args.line_width if args.line_width is not None else 0.2
    range_of_observations = range_of_obs(args)
#     if args.batch_size_to_show in [3, 12]:
    if True:
        if args.grid==1:
            fig = plt.figure()
            n_rows = 3
            n_columns = 4
        for sample_idx_within_batch in range_of_observations:    
            if args.sequence_name=='TowelWall':
                create_face_file_from_num_vertices(submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
                                           submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
                                           sequence_name = args.sequence_name,
                                           dataset_number = args.dataset_number)

    #         # Whole batch
    #         figure_list=show_mesh_triangular_faces_batch(sample_batched=sample_batched,
    #                                  submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
    #                                  submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
    #                                  sequence_name = args.sequence_name,
    #                                  dataset_number = args.dataset_number)
    #         print(type(figure_list[0]))
            i = sample_idx_within_batch + 1
            if args.grid==1:
                ax = fig.add_subplot(n_rows,n_columns,i, projection='3d')
            else:
                fig=plt.figure()
                ax = Axes3D(fig)            

            plot_in_existing_figure=1
            if plot_GT==1:
                ax, title, handle_GT = show_mesh_triangular_faces_tensor(
                    tensor_of_coordinates=labels[sample_idx_within_batch,:,:],
                    submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
                    submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                    sequence_name = args.sequence_name, dataset_number = args.dataset_number,
                    swap_axes = args.swap_axes, plot_in_existing_figure=plot_in_existing_figure, fig=fig,ax=ax, colour='b',
                    line_width=line_width, transparency=transparency, triangular_faces=args.triangular_faces)

            if plot_pred==1:
                ax, title, handle_pred = show_mesh_triangular_faces_tensor(
                    tensor_of_coordinates=outputs[sample_idx_within_batch,:,:],
                    submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
                    submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                    sequence_name = args.sequence_name, dataset_number = args.dataset_number,
                    swap_axes = args.swap_axes, plot_in_existing_figure=plot_in_existing_figure, fig=fig,ax=ax, colour='y', 
                    line_width=line_width, transparency=transparency, triangular_faces=args.triangular_faces)
            if plot_GT==1 and plot_pred==1:
                ax.legend([handle_GT, handle_pred], ['Ground Truth', 'Prediction'], loc=3)
                
                        
            # Hide grid lines
            ax.grid(False)
            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off') 
            
            if args.param_plots_4_paper == 1:
                if 'tshirt' in args.sequence_name:
                    xmin, xmax, umin, ymax, zmin, zmax = -4, 0.5, -0.5, 4.5, 23, 29
                if 'paper' in args.sequence_name: # Parameters not checked yet for paper dataset
                    xmin, xmax, umin, ymax, zmin, zmax = -4, 0.5, -0.5, 4.5, 23, 29
                if args.swap_axes==0:
                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([ymin, ymax])
                    ax.set_zlim([zmin, zmax])
                elif args.swap_axes==1:
                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([zmin, zmax])
                    ax.set_zlim([ymin, ymax])
            
            if args.train_or_val==0:
                title='training set'
            if args.train_or_val==1:
                title='validation set'
            if args.train_or_val==2:
                title='test set'
            title+= '\npredicted xyz'
            if args.title_binary==0: title=''
            plt.title(title)
            ax.view_init(elev=-30., azim=-90)
            
        # Maximise figure window
        if args.grid==1:
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
        if (save_png==1) and (args.save_png_path is not None):
            fig.savefig(args.save_png_path)
    # #             plt.get_current_fig_manager().window.showMaximized()
    #         fig.set_size_inches(8, 6)
    #         # when saving, specify the DPI
    # #             plt.savefig("myplot.png", dpi = 100)
    #         fig.savefig(args.save_png_path, dpi = 100)
        if args.auto_save == 1:
            var_string = args.sequence_name
            var_string += "_" + args.models_lengths_proportion_train_nonAnnotated.replace('.', ',')
            if plot_pred==1 and plot_GT==1:
                var_string += '_GT_and_pred'
            elif plot_GT==1:
                var_string += '_GT'
            elif plot_pred==1:
                var_string += '_pred'
                
            # Create directory if it does not exist
            dir_name = os.path.join('Paper_figs', 'Compare_prediction', 'Auto_save', "Batch" + str(args.batch_to_show))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print('\nCreating directory:\n' + dir_name + "\n")
#             if args.grid==1: plt.show(fig)
            fig.savefig(os.path.join(dir_name, var_string + '.png'))
#             if args.grid==1: plt.close('all')
    else:
        for sample_idx_within_batch in range_of_observations:    
#         for sample_idx_within_batch in [0]:
            # Create face file for the chosen submesh
            create_face_file_from_num_vertices(submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
                                       submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
                                       sequence_name = args.sequence_name,
                                       dataset_number = args.dataset_number)

    #         # Whole batch
    #         figure_list=show_mesh_triangular_faces_batch(sample_batched=sample_batched,
    #                                  submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
    #                                  submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
    #                                  sequence_name = args.sequence_name,
    #                                  dataset_number = args.dataset_number)
    #         print(type(figure_list[0]))
            ax, title, fig = show_mesh_triangular_faces_tensor(
                tensor_of_coordinates=labels[sample_idx_within_batch,:,:],
                submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
                submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                sequence_name = args.sequence_name, dataset_number = args.dataset_number,
                swap_axes = args.swap_axes, transparency=transparency, triangular_faces=args.triangular_faces)
            title += ' - GT'
            if args.title_binary==0: title=''
            plt.title(title)
            
            ax, title, handle = show_mesh_triangular_faces_tensor(
                tensor_of_coordinates=outputs[sample_idx_within_batch,:,:],
                submesh_num_vertices_vertical=args.submesh_num_vertices_vertical,
                submesh_num_vertices_horizontal=args.submesh_num_vertices_horizontal,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                sequence_name = args.sequence_name, dataset_number = args.dataset_number,
                swap_axes = args.swap_axes, plot_in_existing_figure=1, fig=fig,ax=ax, colour='y', 
                line_width=line_width, transparency=transparency, triangular_faces=args.triangular_faces)
            title += ' - Prediction'
            if args.title_binary==0: title=''
            plt.title(title)
    if args.auto_save == 0:
        plt.show()

def plot_GT_surface_grid():
    args = functions_data_processing.parser()
    
    # Load 1st batch
    sample_batched = data_loading.get_1st_batch(args)
    labels = sample_batched['Vertex_coordinates']
    visualize_xyz_prediction_grid(args, labels, labels, sample_batched=sample_batched, plot_pred=0)
    plt.show()
    
# if __name__=='__main__':
#     plot_GT_surface_grid()
    
def GIF_append(fig, GIF_image_list):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    GIF_image_list.append(image)
    return GIF_image_list

def GIF_save(GIF_name, GIF_image_list, fps=5.0):
    kwargs_write = {'fps':fps, 'quantizer':'nq'}
    imageio.mimsave(GIF_name, GIF_image_list, fps=fps)
    
def show_for_seconds(milliseconds=3000):
    def close_event():
        plt.close('all') #timer calls this function after 3 seconds and closes the window 
    fig=plt.gcf()
    timer = fig.canvas.new_timer(interval = milliseconds) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)
    timer.start()
    plt.show()
    
def plot_performance_graph_wrt_annot_ratio(x = range(0, 100, 2), y=np.cos(range(0, 100, 2)), x_both=range(0, 100, 2),
                                           y_both=np.sin(range(0, 100, 2)), real_dataset='tshirt', show=1, finetune_baseline=None, 
                                           finetune_NonAnn=None, verbose=1, x_both_finetuned=None, y_both_finetuned=None):
    """
    (x, y) graph info of annotated data only
    (x_both, y_both) graph info of both annotated and non-annotated data
    """
#     figsize=(6.4, 10)
    figsize=None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)   
    import pylab as plot
    #     params = {'legend.fontsize': 40,
    #           'legend.handlelength': 3}
    #     plot.rcParams.update(params)
    x=[int(m*100) for m in x]
    x_both=[int(m*100) for m in x_both]
    plt.plot(x, y, linewidth=5, color='y') 
    plt.plot(x_both, y_both, linewidth=5, color='b', linestyle='dashed') 
    
    if x_both_finetuned is not None:
        x_both_finetuned=[int(m*100) for m in x_both_finetuned]
        plt.plot(x_both_finetuned, y_both_finetuned, linewidth=5, color='r', linestyle='dotted') 
        legend_list=['Annotated only', 'Annotated and non-annotated', 'Annotated and non-annotated (unsup. finetune)']
    else:
        legend_list=['Annotated only', 'Annotated and non-annotated']
    if verbose==1:
        legend_list[0] += ' - finetune: ' + str(finetune_baseline)
        legend_list[1] += ' - finetune: ' + str(finetune_NonAnn)
    plt.legend(legend_list, loc='upper right')
#     plt.xlabel('Percentage (%) of annotated training data used', fontsize='x-large') 
    plt.xlabel('Size of annotated training set (in thousands)', fontsize='x-large') 
#     plt.xlabel('Percentage (%) of annotated training data used') 
    ax.set_xlim([0, 100])
    plt.ylabel('3D error', fontsize='x-large') 
    ax2 = ax.twiny()
#     ax2.plot(x_both, y_both, linewidth=5, color='b', linestyle='dashed') # Create a dummy plot
    ax2.scatter([0], [4], color='w') # Create a dummy plot
    ax2.set_xlim(ax.get_xlim())
    ax2.invert_xaxis()
#     ax2.cla()
    ax2.set_xlabel('Size of non-annotated training set (in thousands)', fontsize='x-large')
    
#     SMALL_SIZE = 8
#     MEDIUM_SIZE = 10
#     BIGGER_SIZE = 50
#     plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
#     # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#     font = {'family' : 'normal',  'weight' : 'bold',   'size'   : 22}
#     matplotlib.rc('font', **font)
    
    title = 'Moving T-shirt dataset' if real_dataset=='tshirt' else 'Bending paper dataset'
    figure_name = 'MovingTshirt' if real_dataset=='tshirt' else 'BendingPaper'
#     plt.title(title, y=1.08)
    plt.text(0.5, 1.28, title,
         horizontalalignment='center',
         fontsize=20,
         transform = ax2.transAxes)
#     # Get figure size
#     fig = plt.gcf()
#     size = fig.get_size_inches()*fig.dpi # size in pixels
#     size = fig.get_size_inches()
#     print(size) #[640. 480.] in pixels, [6.4 4.8] in inches
    
#     xmin, xmax, ymin, ymax = 0, 11, 0, 0.12
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
    # Maximise figure window
#     mng = plt.get_current_fig_manager()
#     mng.full_screen_toggle()
    if show==1: plt.show()
    fig.savefig(os.path.join('Paper_figs', 'Error_Graph', 'Auto_save', figure_name + '.png'))
    
# if __name__=='__main__':
#     plot_performance_graph_wrt_annot_ratio()