import numpy as np
from numpy import genfromtxt
import pandas as pd

from plot_vertices_in_order import plot_vertices_in_order_function
import functions_data_processing

def reorder_vertices_from_Blender():
    # Choosing sequence and animation frame
    sequence_name = 'TowelWall'
    animation_frame = '00001'

    # Load the vertices files disregarding the string '# ' at the beginning of the file
    f = open('Renders' + sequence_name + '/vertices_' + animation_frame + '.txt', 'r')
    line1 = f.readline()
    df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())

    # Extract the x, y, z coordinates of the vertices
    df_X = df_vertices_all_data['x']
    df_Y = df_vertices_all_data['y']
    df_Z = df_vertices_all_data['z']
    X = df_X.values
    Y = df_Y.values
    Z = df_Z.values

    # The top-left vertex is the third one in the order given by Blender
    idx_1st_vertex_in_row=2

    # Find all vertices with same x coordinate as the top-left vertex
    # These correspond to the left-most vertex of each row
    column = [i for i in range(len(X)) if X[i] == X[idx_1st_vertex_in_row]]

    # Select the z coordinate of these vertices to sort them from top to bottom
    column_z = [Z[i] for i in column]
    column_reordering_idx = np.argsort(column_z)[::-1] # [::-1] used to reverse the order
    column_new_vertex_order = [column[i] for i in column_reordering_idx]

    num_vertices_height = (2**2)*13
#     print('True number of vertices in each column:', num_vertices_height)
#     print('Number of vertices found in the left-most column:', len(column_new_vertex_order))
#     print()

    # Initialize the list which will contain the indices of the vertices in the new order
    new_vertex_order = []

    tol = 10**-3 # Tolerance to compare the equality of vertex coordinates
#     j=0
    for idx_1st_vertex_in_row in column_new_vertex_order:
        # Find all vertices with same y, z coordinates as the left-most vertex of the row
        row = [ i for i in range(len(X)) if (abs(Y[i]-Y[idx_1st_vertex_in_row])<tol and 
                                             abs(Z[i]-Z[idx_1st_vertex_in_row])<tol) ]

        # Select the x coordinates of these vertices to sort them left to right
        row_x = [X[i] for i in row]
        row_reordering_idx = np.argsort(row_x)
        row_new_vertex_order = [row[i] for i in row_reordering_idx]

        new_vertex_order += row_new_vertex_order
#         num_vertices_width = 103
#         print(j, len(row), len(new_vertex_order))
#         j+=1

    # Convert the idx list to a numpy array to make the reordering of X much faster when running X = X[new_vertex_order]
    # https://stackoverflow.com/questions/26194389/numpy-rearrange-array-based-upon-index-array
    new_vertex_order = np.array(new_vertex_order) 

    return new_vertex_order

new_vertex_order = reorder_vertices_from_Blender()

def vertex_old_idx_to_new_idx(n, new_vertex_order=new_vertex_order):
#     print(type(new_vertex_order)) # <class 'numpy.ndarray'>
#     return new_vertex_order.index(n) # since new_vertex_order is a numpy array instead of a list, we do as follows:
    return int(np.where(new_vertex_order==n)[0][0])

def face_file_old_idx_to_new_idx(faces, new_vertex_order=new_vertex_order):
    f = np.vectorize(vertex_old_idx_to_new_idx, otypes=[np.int])
    return f(faces)
    
if __name__ == "__main__":
    ###
    ### Parser for entering arguments and setting default ones
    ###
    import argparse
    import torch
    args = functions_data_processing.parser()

    plot_or_GIF = args.plot_or_GIF

    if plot_or_GIF in [0, 1]: # 1=GIF, 0=plot
        group_number = 1
        animation_frame = '00001'   
    #     animation_frame = str(int(animation_frame))            

        # Load the vertices files disregarding the string '# ' at the beginning of the file
        filename = 'Renders' + args.sequence_name + args.dataset_number + '/Group.' + str(group_number).zfill(3) + '/vertices_' + animation_frame + '.txt'
        f = open(filename, 'r')
        line1 = f.readline()
        df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())

        # Extract the x, y, z coordinates of the vertices
        df_X = df_vertices_all_data['x']
        df_Y = df_vertices_all_data['y']
        df_Z = df_vertices_all_data['z']
        X = df_X.values
        Y = df_Y.values
        Z = df_Z.values

        X = X[new_vertex_order]
        Y = Y[new_vertex_order]
        Z = Z[new_vertex_order]
        plot_vertices_in_order_function(plot_or_GIF, X, Y, Z, gif_name='GIF/vertices_in_new_order.gif') 
    elif plot_or_GIF==2: # 2=reorder vertex dataset
        ###
        ### Save reordered vertex data files
        ###
#         for group_number in range(1, 2):
#             for animation_frame in range(1, 2):
        for group_number in range(1, args.num_groups+1):
            for animation_frame in range(1, args.num_animationFramesPerGroup+1):

                # Load the vertices files disregarding the string '# ' at the beginning of the file
                filename = 'Renders' + args.sequence_name + args.dataset_number + '/Group.' + str(group_number).zfill(3) 
                filename = filename + '/vertices_' + str(animation_frame).zfill(5)
                f = open(filename + '.txt', 'r')
                line1 = f.readline()
                df_vertices_all_data = pd.read_csv(f, sep = ' ',  names = line1.replace('# ', '').split())

#                 print(df_vertices_all_data.head())
                df_vertices_all_data_reordered = df_vertices_all_data.reindex(new_vertex_order)
#                 print(df_vertices_all_data_reordered.head())

                # Save reordered vertex dataset
                df_vertices_all_data_reordered.to_csv(path_or_buf=filename + '_reordered.txt', sep=' ', na_rep='', 
                                                      header=True, index=False, line_terminator='\n', decimal='.')
                # to keep the old indexing, set index=True
                # Notice that we do not include the has sign # at the beginning of the file, so we must take this into account when loading the file

#                 # Load vertex dataset to check it was saved properly
#                 # It does work propery
#                 f = open(filename + '_reordered.txt', 'r')
#                 line1 = f.readline()
#                 df_vertices_all_data_reordered_loaded = pd.read_csv(f, sep = ' ',  names = line1.split()) # Notice that we do not include the has sign # at the beginning of the file, so we must take this into account when loading the file
#                 print(df_vertices_all_data_reordered_loaded.head())
    elif plot_or_GIF==3: # 3=reorder face dataset
        ###
        ### Save reordered face data file
        ###
        # Load the faces file.
        # Each face is represented by 4 numbers in a row.
        # Each of these numbers represent a vertex.
        # Vertices are ordered by row in any file of the form 'RendersTowelWall/vertices_*****.txt'
        # There is no heading, so I will import it directly as a numpy array, rather than as a panda DataFrame
        filename = 'Renders' + args.sequence_name + args.dataset_number + '/faces_mesh'
        faces = genfromtxt(filename + '.txt', delimiter=' ')
        # print(type(faces))
        # print(faces[0:5, :])
        faces = faces.astype(int)
        # print(faces[0:5, :])

#         # In the faces file, the vertices are indexed starting from 0
#         print('Number of vertices = ' + str(X.size))
#         # print('min_index in faces file = ' + str(np.min(faces))) # = 0
#         # print('max_index in faces file = ' + str(np.max(faces))) # = X.size - 1

#         print('Number of faces = ' + str(faces.size))
        
#         print(vertex_old_idx_to_new_idx(faces[0,0], new_vertex_order=new_vertex_order))
        faces_reordered = face_file_old_idx_to_new_idx(faces, new_vertex_order=new_vertex_order)
#         print(type(faces_reordered))
        print(faces_reordered[0:5, :])

        np.savetxt(filename + '_reordered.txt', faces_reordered, delimiter=' ', fmt='%d')