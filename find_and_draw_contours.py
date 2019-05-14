import numpy as np
import cv2
import os
import torch

import binary_image

def find_and_draw_contour_of_towel_from_binary(binary_TowelOnly, verbose=0):
    # Find contours
    im2, ctrs_TowelOnly, hier = cv2.findContours(binary_TowelOnly.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours (from left to right and top to bottom)
    sorted_ctrs_TowelOnly = sorted(ctrs_TowelOnly, key=lambda ctr: cv2.boundingRect(ctr)[0])
#     print('There is/are ' + str(len(sorted_ctrs_TowelOnly)) + ' contour(s) in Towel only.')
    
    # Find the bounding box of the contour closest to the top left corner
    # which has no size 1x1
    i, w, h = 0, 1, 1
    while i<len(sorted_ctrs_TowelOnly) and (w<=10 or h<=10):
        ctr = sorted_ctrs_TowelOnly[i] 
        x, y, w, h = cv2.boundingRect(ctr) # Get bounding box
        if verbose==1:
            print("(u,v) of upper-left corner of the box = (" + str(x) + ", " + str(y) + ")")
            print("width and height of the box :", w, h)
        i+=1
   
    # Draw contours
    if verbose==1:
        # Draw the 0th contour:
        cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

        # Draw all contours
    #     cv2.drawContours(img,ctrs_TowelOnly,-1,(0,0,255),1)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
    return [ctr] # list of a single element

def find_and_draw_contour_of_towel(input_png_name, verbose=0):
    # Binary image
    binary_TowelOnly = binary_image.binary_img_from_towel_igm_name(input_png_name, verbose)

    return find_and_draw_contour_of_towel_from_binary(binary_TowelOnly, verbose)
    
if __name__=='__main__':
    find_and_draw_contour_of_towel(input_png_name='RendersTowelWall11/Group.003/17.png', verbose=0)

def convert_contour_to_uv(ctrs_TowelOnly):
    """CAVEAT: convert_contour_to_uv only returns some uv pixels. 
    The missing ones correspond to straight lines connecting the reported uv pixels"""
    xcnts = np.vstack((x.reshape(-1,2) for x in ctrs_TowelOnly[0]))
    u = xcnts[:,0]
    v = xcnts[:,1]
    return u, v

def convert_contour_to_uv_from_img_name(input_png_name, verbose=0):
    """CAVEAT: convert_contour_to_uv_from_img_name only returns some uv pixels. 
    The missing ones correspond to straight lines connecting the reported uv pixels"""
    ctrs_TowelOnly = find_and_draw_contour_of_towel(input_png_name, verbose)
    return convert_contour_to_uv(ctrs_TowelOnly)

def convert_contour_to_uv_from_binary(binary_TowelOnly, verbose=0):
    """CAVEAT: convert_contour_to_uv_from_binary only returns some uv pixels. 
    The missing ones correspond to straight lines connecting the reported uv pixels"""
    ctrs_TowelOnly = find_and_draw_contour_of_towel_from_binary(binary_TowelOnly, verbose)
    return convert_contour_to_uv(ctrs_TowelOnly)
    
def convert_contour_to_uv_from_img_name_slower(input_png_name, verbose=0):
    """CAVEAT: convert_contour_to_uv_from_img_name_slower only returns some uv pixels. 
    The missing ones correspond to straight lines connecting the reported uv pixels"""
    ctrs_TowelOnly = find_and_draw_contour_of_towel(input_png_name, verbose)
    u=[]
    v=[]
    for i in range(len(ctrs_TowelOnly[0])):
        u.append(ctrs_TowelOnly[0][i][0][0])
    for i in range(len(ctrs_TowelOnly[0])):
        v.append(ctrs_TowelOnly[0][i][0][1])
    return u, v

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import functions_plot
    sequence_name = 'TowelWall'
    dataset_number = '11'
    group_number = '003'
    animation_frame = '17'
#     animation_frame = '1'
    submesh_num_vertices_horizontal = 52
    submesh_num_vertices_vertical = 103
    verbose=0

    input_png_name = os.path.join('Renders' + sequence_name + dataset_number, 'Group.'+ group_number, animation_frame + '.png')
    u, v = convert_contour_to_uv_from_img_name(input_png_name=input_png_name, verbose=verbose)


    # Plot uv of contour without straight lines on RGB
    fig=functions_plot.plot_RGB_and_landmarks(u_visible=u, v_visible=v, 
                                          sequence_name=sequence_name, dataset_number=dataset_number, 
                                          group_number=group_number, animation_frame=animation_frame,
                                         marker_size=50/submesh_num_vertices_vertical)
    plt.show()




# Access all pixels from contour, without omitting the ones which describe straight lines
def append_missing_new_value(w, i, l):
    """ update w (u or v) adding the missing points between w[i] and w[i+1].
    l = len of the original w before undergoing any updates at all"""
    sign = 1 if w[(i+1)%l]>w[i%l] else -1
    for k in range(sign, w[(i+1)%l]-w[i%l], sign):
        w=np.append(w, w[i]+k)
    return w

def append_missing_copied_value(w, i, l, w_complementary):
    """ - update w (u or v) adding the missing points between w[i] and w[i+1].
    - l = len of the original w before undergoing any updates at all.
    - w_complementary = u if w is v and v if w is u. 
    This is used to compute the number of copies to append to w. """
    n_copies = abs(w_complementary[(i+1)%l]-w_complementary[i%l])-1 # number of copies to append
    w = np.append(w, w[i]*np.ones((n_copies,), dtype=int))
    return w

def full_list_of_contour_pixels(u, v):
    """ This assumes the contour pixels are ordered but the output ones are not, 
    since the new ones are appended at the end """
    l = len(u)
    for i in range(l):
        if abs(u[i%l]-u[(i+1)%l])>1 or abs(v[i%l]-v[(i+1)%l])>1:
            if abs(u[i%l]-u[(i+1)%l])==abs(v[i%l]-v[(i+1)%l]): # append diagonal
                u = append_missing_new_value(u, i, l)
                v = append_missing_new_value(v, i, l)
            elif u[i%l]-u[(i+1)%l]==0: # append vertical line
                u = append_missing_copied_value(u, i, l, v)
                v = append_missing_new_value(v, i, l)
            elif v[i%l]-v[(i+1)%l]==0: # append horizontal line
                u = append_missing_new_value(u, i, l)
                v = append_missing_copied_value(v, i, l, u)
    return u, v

if __name__=='__main__':
    u, v = full_list_of_contour_pixels(u, v)

    # Plot uv of full contour on RGB
    fig=functions_plot.plot_RGB_and_landmarks(u_visible=u, v_visible=v, 
                                          sequence_name=sequence_name, dataset_number=dataset_number, 
                                          group_number=group_number, animation_frame=animation_frame,
                                         marker_size=50/submesh_num_vertices_vertical)
    plt.show()




# Putting it all together
def full_list_of_contour_pixels_from_towel_imgname(input_png_name, verbose=0):
    u, v = convert_contour_to_uv_from_img_name(input_png_name, verbose)
    u, v = full_list_of_contour_pixels(u, v)
    return u, v

if __name__=='__main__':
    u, v = full_list_of_contour_pixels_from_towel_imgname(input_png_name, verbose)

    # Plot uv of full contour on RGB
    fig=functions_plot.plot_RGB_and_landmarks(u_visible=u, v_visible=v, 
                                          sequence_name=sequence_name, dataset_number=dataset_number, 
                                          group_number=group_number, animation_frame=animation_frame,
                                         marker_size=50/submesh_num_vertices_vertical)
    plt.show()

def full_list_of_contour_pixels_from_binary(binary_TowelOnly, verbose=0):
    u, v = convert_contour_to_uv_from_binary(binary_TowelOnly, verbose)
    if verbose==1:
        print("len(u):", len(u))
    u, v = full_list_of_contour_pixels(u, v)
    return u, v

if __name__=='__main__':
    binary_TowelOnly = binary_image.binary_img_from_towel_igm_name(input_png_name, verbose)

    u, v = full_list_of_contour_pixels_from_binary(binary_TowelOnly, verbose=0)

    # Plot uv of full contour on RGB
    fig=functions_plot.plot_RGB_and_landmarks(u_visible=u, v_visible=v, 
                                          sequence_name=sequence_name, dataset_number=dataset_number, 
                                          group_number=group_number, animation_frame=animation_frame,
                                         marker_size=50/submesh_num_vertices_vertical,
                                              title='full_list_of_contour_pixels_from_binary')
    plt.show()
    
def full_tensor_of_contour_pixels_from_binary(binary_TowelOnly, device='cpu', verbose=0, numpy_or_tensor=1, dtype=1):
    """ Find in the binary image named binary_TowelOnly the contour of the region of pixels occupied by the towel. 
    If numpy_or_tensor=0, it returns a numpy array of shape Nx2, where the 1st column corresponds to u and 2nd to v.
    If numpy_or_tensor=1, it returns a tensor with the same properties instead.
    
    dtype==0 -->float
    dtype==1 -->double
    """    
    u, v =full_list_of_contour_pixels_from_binary(binary_TowelOnly, verbose)
    
    if numpy_or_tensor==1:
        if dtype==0: 
            return torch.from_numpy(np.concatenate((np.reshape(u, (-1, 1)),np.reshape(v, (-1, 1))), axis=1)).to(device).float().view(-1, 2) 
        else:
            return torch.from_numpy(np.concatenate((np.reshape(u, (-1, 1)),np.reshape(v, (-1, 1))), axis=1)).to(device).double().view(-1, 2) 

if __name__=='__main__':
    binary_TowelOnly = binary_image.binary_img_from_towel_igm_name(input_png_name, verbose)
    print(full_tensor_of_contour_pixels_from_binary(binary_TowelOnly, verbose=0, numpy_or_tensor=1))