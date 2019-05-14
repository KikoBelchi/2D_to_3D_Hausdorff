# Plot vertices of the mesh in the order in which they appear in the csv files of 3d coordinates

# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import io

plot_or_GIF = 1 # GIF
# plot_or_GIF = 0 # plot

def plot_vertices_in_order_function(plot_or_GIF, X, Y, Z, gif_name):
    num_vertices = X.size
    num_of_vertices_to_show = num_vertices

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 1])
    ax.set_zlim([1, 3])

    # # For plotting all selected vertices from jupyter notebook or terminal at the same time
    # ax.scatter(X[0:num_of_vertices_to_show], Y[0:num_of_vertices_to_show], Z[0:num_of_vertices_to_show], 
    #            c='b', marker='o', s=50)

    if plot_or_GIF==1: # GIF
        import imageio
        llista=[]
        # Creating a GIF with all vertices appearing in the order given by Blender
        for i in range(0, min(12, num_of_vertices_to_show)):
            ax.scatter(X[i], Y[i], Z[i], 
                   c='b', marker='o', s=50)  
            # Used to return the plot as an image rray
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            llista.append(image)
        for i in range(min(12, num_of_vertices_to_show), min(700, num_of_vertices_to_show)):
            ax.scatter(X[i], Y[i], Z[i], 
                   c='b', marker='o', s=50)
            # Used to return the plot as an image rray
            if i%30==0:
                fig.canvas.draw()       # draw the canvas, cache the renderer
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                llista.append(image)
        for i in range(min(700, num_of_vertices_to_show), num_of_vertices_to_show):
            ax.scatter(X[i], Y[i], Z[i], 
                   c='b', marker='o', s=50)
            # Used to return the plot as an image rray
            if i%300==0:
                fig.canvas.draw()       # draw the canvas, cache the renderer
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                llista.append(image)
        kwargs_write = {'fps':1.0, 'quantizer':'nq'}
        imageio.mimsave(gif_name, llista, fps=1)
    
    if plot_or_GIF==0: # Plot
        # For plotting all vertices one by one as an animation. You need to export to .py and run in terminal
        for i in range(0,6):
            ax.scatter(X[i], Y[i], Z[i], 
                   c='b', marker='o', s=50)  
            plt.pause(1.5)
        for i in range(6, num_of_vertices_to_show):
            ax.scatter(X[i], Y[i], Z[i], 
                   c='b', marker='o', s=50)
            plt.pause(0.00000000001)

if __name__ == '__main__':
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
    
    plot_vertices_in_order_function(plot_or_GIF, X, Y, Z, gif_name='GIF/vertices_in_Blender_order.gif')


