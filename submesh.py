""" Functions to create submeshes and find neighbours within it. """

import numpy as np
import torch

import functions_data_processing

def matrix_coord_from_idx(idx, num_vertices_width = 103):
    """ Input: the index of a vertex within the reordered vertex list 
    (i.e., not from the vertex list ordered as directly given by Blender)
    
    Output: matrix_coord is a tuple of the form (x, y), 
    where these are the matrix coordinates of a vertex seen as a vertex of the original mesh (not as a vertex of a submesh).
    These (x, y) matrix coordinates correspond to vertical downwards and horizontal rightwards, respectively. """
    matrix_coord_from_idx_0 = int(int(idx) // num_vertices_width)
    matrix_coord_from_idx_1 = idx%num_vertices_width
    return matrix_coord_from_idx_0, matrix_coord_from_idx_1

def CHECK_matrix_coord_from_idx():
    """ Function to check that matrix_coord_from_idx() works """
    print(matrix_coord_from_idx(idx=0))
    print(matrix_coord_from_idx(idx=102))
    print(matrix_coord_from_idx(idx=103))
    print(matrix_coord_from_idx(idx=205))

def idx_from_matrix_coord(matrix_coord, num_vertices_width = 103):
    """ Input: matrix_coord is a tuple of the form (x, y), where these are the matrix coordinates of a vertex seen as a vertex of the original mesh (not as a vertex of a submesh).
    These (x, y) matrix coordinates correspond to vertical downwards and horizontal rightwards, respectively.
    
    Output: the index of the vertex within the reordered vertex list 
    (i.e., not from the vertex list ordered as directly given by Blender) """
    return matrix_coord[1] + matrix_coord[0]*num_vertices_width

def CHECK_idx_from_matrix_coord():
    """ Function to check that idx_from_matrix_coord() works """
    print(idx_from_matrix_coord(matrix_coord=(0, 0)))
    print(idx_from_matrix_coord(matrix_coord=(0, 102)))
    print(idx_from_matrix_coord(matrix_coord=(1, 0)))
    print(idx_from_matrix_coord(matrix_coord=(1, 102)))

def submesh_idx_from_num_vertices_in_each_direction(submesh_num_vertices_vertical = 2,
                                                    submesh_num_vertices_horizontal = 3,
                                                    num_vertices_width = 103,
                                                    num_vertices_height = 52):
    """ Input:
    - num_vertices_width: number of vertices in the horizontal direction in the original mesh. 
    - num_vertices_height: number of equidistant vertices in the vertical direction in the original mesh.
    - submesh_num_vertices_horizontal: number of vertices in the horizontal direction which want to be selected.
      The vertices will be chosen so that the distance between any consecutive vertices is the same.
    - submesh_num_vertices_vertical: number of vertices in the vertical direction which want to be selected.
      The vertices will be chosen so that the distance between any consecutive vertices is the same.
    Output:
    - List containing the indices which would select the wanted vertices from the reordered vertex list 
    (i.e., not from the vertex list ordered as directly given by Blender).
    """
    submesh_idx_vertical = np.round(np.linspace(0, num_vertices_height-1, submesh_num_vertices_vertical)).astype(int)
    # This is the matrix coordinate 0
#     print(submesh_idx_vertical)

    submesh_idx_horizontal = np.round(np.linspace(0, num_vertices_width-1, submesh_num_vertices_horizontal)).astype(int)
    # This is the matrix coordinate 1
#     print(submesh_idx_horizontal)

    submesh_matrix_coord = [(a, b) for a in submesh_idx_vertical for b in submesh_idx_horizontal]
#     print(submesh_matrix_coord)
    submesh_idx = [idx_from_matrix_coord(matrix_coord) for matrix_coord in submesh_matrix_coord]
    return submesh_idx
    
def CHECK_submesh_idx_from_num_vertices_in_each_direction():
    """ Function to check that submesh_idx_from_num_vertices_in_each_direction() works """
    submesh_idx = submesh_idx_from_num_vertices_in_each_direction(submesh_num_vertices_vertical = 2,
                                                                  submesh_num_vertices_horizontal = 3)
    print(submesh_idx)
    print([matrix_coord_from_idx(i) for i in submesh_idx])
    
def Allen_neighbours_of_generic_vertex_in_mesh(matrix_coord, num_vertices_width = 103, num_vertices_height = 52):
    """ Input: matrix_coord is a tuple of the form (x, y), 
    where these are the matrix coordinates of a vertex of a mesh.
    These (x, y) matrix coordinates correspond to vertical downwards and horizontal rightwards, respectively.
    
    Output: list of matrix coordinates (as tuples (x, y)) of the neighbours of the input vertex which sit:
    - on top, 
    - on top and on the right, 
    - on the right and 
    - below and on the right, making an Allen key shape hanging on top of the center vertex. 
    This way, we only select 4 of the 8 neighbours of the vertex, 
    but once we run this for all the vertices, 
    we will have all the pairs of neighbours and no pair will be repeated.
    
    Note: (original mesh vs submesh) 
    The mesh all vertices are considered in for the matrix coordinates and for the notion of neighbour is the mesh with:
    - number of vertices in horizontal direction: num_vertices_width.
    - number of vertices in vertical direction: num_vertices_height. """
    neighbours = []
    if matrix_coord[0]!=0: 
        # If vertex is not in the top row --> it has a neighbour right on top
        neighbours.append((matrix_coord[0]-1, matrix_coord[1]))
    if matrix_coord[0]!=0 and matrix_coord[1]!=(num_vertices_width-1): 
        # If vertex not in top row nor in last column --> it has a neighbour on top, on the right
        neighbours.append((matrix_coord[0]-1, matrix_coord[1]+1))
    if matrix_coord[1]!=(num_vertices_width-1): 
        # If vertex not in last column --> it has a neighbour on the right
        neighbours.append((matrix_coord[0], matrix_coord[1]+1))
    if matrix_coord[0]!=(num_vertices_height-1) and matrix_coord[1]!=(num_vertices_width-1): 
        # If vertex not in last row nor in last column --> it has a neighbour below, on the right
        neighbours.append((matrix_coord[0]+1, matrix_coord[1]+1))
    return neighbours

def CHECK_Allen_neighbours_of_generic_vertex_in_mesh():
    """Function to check that Allen_neighbours_of_generic_vertex_in_mesh() works"""
    list_of_matrix_coord = [(0, 0), (0, 102), (1, 0), (1, 102), (51, 0), (51, 102)]
    for matrix_coord in list_of_matrix_coord:
        print('neighbours of', matrix_coord, ': ', end='')
        print(Allen_neighbours_of_generic_vertex_in_mesh(matrix_coord, num_vertices_width = 103, num_vertices_height = 52))
        
def idx_within_submesh_of_Allen_neighbours_within_submesh(v_idx_submesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3):
    """ Input: 
    - v_idx_mesh: index within the submesh of a vertex v of the submesh.
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    This function finds a list of the Allen-key neighbours of v within the submesh.
    The notion of Allen-key neighbour is explained in the description of Allen_neighbours_of_generic_vertex_in_mesh().
    Output: 
    - list of indices within the submesh of these neighbours. """   
    # Convert index of v within the submesh to matrix coordinates within the submesh
    v_matrix_coord_submesh = matrix_coord_from_idx(v_idx_submesh, num_vertices_width = submesh_num_vertices_horizontal)
    
    # Find list of matrix coordinates within submesh of allen neighbours of v
    neighbours_matrix_coord_submesh = Allen_neighbours_of_generic_vertex_in_mesh(
        v_matrix_coord_submesh, num_vertices_width = submesh_num_vertices_horizontal,
        num_vertices_height = submesh_num_vertices_vertical)
    
    # Convert matrix coordinates of neighbours within submesh to indices within submesh
    neighbours_idx_submesh = [idx_from_matrix_coord(matrix_coord, num_vertices_width = submesh_num_vertices_horizontal) 
                              for matrix_coord in neighbours_matrix_coord_submesh]
    
    return neighbours_idx_submesh

def idx_within_mesh_of_Allen_neighbours_within_submesh(v_idx_mesh, submesh_idx_within_mesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3):
    """ Input: 
    - v_idx_mesh: index within the original mesh of a vertex v of the submesh.
    - submesh_idx_within_mesh: list containing the indices within the (reordered, not directly from Blender) original mesh 
    of the vertices that form the submesh.
    This is the output of submesh_idx_from_num_vertices_in_each_direction().
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    This function finds a list of the Allen-key neighbours of v within the submesh.
    The notion of Allen-key neighbour is explained in the description of Allen_neighbours_of_generic_vertex_in_mesh().
    Output: 
    - list of indices within the original mesh of these neighbours. """
    # Convert index of v within the original mesh to index within the submesh
    v_idx_submesh = submesh_idx_within_mesh.index(v_idx_mesh)
    
    # Find indices within the submesh of the allen-neighbours of v
    neighbours_idx_submesh=idx_within_submesh_of_Allen_neighbours_within_submesh(v_idx_submesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3)
    
    # Convert indices of neighbours within the submesh to indices within the original mesh
    neighbours_idx_mesh = [submesh_idx_within_mesh[neighbour_idx_submesh] for neighbour_idx_submesh in neighbours_idx_submesh]
    
    return neighbours_idx_mesh

def CHECK_idx_within_mesh_of_Allen_neighbours_within_submesh():
    """Function to check that idx_within_mesh_of_Allen_neighbours_within_submesh() works"""
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 3
    # Find idx within the original mesh of all the vertices of the submesh
    submesh_idx_within_mesh = submesh_idx_from_num_vertices_in_each_direction(
        submesh_num_vertices_vertical = submesh_num_vertices_vertical,
        submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
    print('submesh_idx_within_mesh:', submesh_idx_within_mesh)
    for v_idx_mesh in submesh_idx_within_mesh:
        print('Allen-neighbours of', v_idx_mesh, '--> ' , end='')
        print(idx_within_mesh_of_Allen_neighbours_within_submesh(
            v_idx_mesh = v_idx_mesh, submesh_idx_within_mesh = submesh_idx_within_mesh, 
            submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal))

# CHECK_idx_within_mesh_of_Allen_neighbours_within_submesh()

def squared_l2_distance_of_two_vertices_from_submesh(v_idx_submesh, w_idx_submesh, Vertex_coordinates_submesh):
    """ Input: 
    - v_idx_submesh: index within the submesh of a vertex v.
    - w_idx_submesh: index within the submesh of a vertex w.
    - Vertex_coordinates_submesh: numpy array of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., 1. Instanciate the dataset class to transformed_dataset.
    2. Get the first example in the dataset via template_submesh = transformed_dataset[0].
    3. Then, 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.

    Output:
    - squared_l2: squared Euclidean distance of vertices v and w within submesh of a given pose of the towel 
    (this submesh should be obtained from the dataloader and is given by the list of vertex coordinates
    Vertex_coordinates_submesh). This refers to the xyz coordinates of the vertices or the uv coordinates,
    depending on what the input Vertex_coordinates_submesh is.
    See submesh_CHECKS.CHECK_squared_l2_distance_of_two_vertices_from_submesh() for more details and an examle of both cases."""
    
    v_Vertex_coordinates = Vertex_coordinates_submesh[v_idx_submesh, :]
    w_Vertex_coordinates = Vertex_coordinates_submesh[w_idx_submesh, :]
    squared_l2 = np.sum(np.square(v_Vertex_coordinates - w_Vertex_coordinates))
    return squared_l2

# CHECK_squared_l2_distance_of_two_vertices_from_submesh() is in the file submesh_CHECKS.py   

def squared_l2_distance_of_two_vertices_from_submesh_tensor(v_idx_submesh, w_idx_submesh, Vertex_coordinates_submesh, squared=1):
    """ Input: 
    - v_idx_submesh: index within the submesh of a vertex v.
    - w_idx_submesh: index within the submesh of a vertex w.
    - Vertex_coordinates_submesh: tensor of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., instanciate the dataset class to transformed_dataset,
    Get the first example in the dataset: template_submesh = transformed_dataset[0]
    then 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.

    Output:
    - squared_l2: squared Euclidean distance of vertices v and w within submesh of a given pose of the towel 
    (this submesh should be obtained from the dataloader and is given by the list of vertex coordinates
    Vertex_coordinates_submesh). This refers to the xyz coordinates of the vertices or the uv coordinates,
    depending on what the input Vertex_coordinates_submesh is.
    See submesh_CHECKS.CHECK_squared_l2_distance_of_two_vertices_from_submesh() for more details and an example of both cases."""
    
    v_Vertex_coordinates = Vertex_coordinates_submesh[v_idx_submesh, :]
    w_Vertex_coordinates = Vertex_coordinates_submesh[w_idx_submesh, :]
    if squared==1: # square of L2 distance
        squared_l2 = torch.sum(torch.pow(v_Vertex_coordinates - w_Vertex_coordinates, 2))
    else: # L2 distance
        squared_l2 = torch.sqrt(torch.sum(torch.pow(v_Vertex_coordinates - w_Vertex_coordinates, 2)))
    return squared_l2

# CHECK_squared_l2_distance_of_two_vertices_from_submesh_tensor() is in the file submesh_CHECKS.py   
        
def squared_l2_distance_of_adjacent_vertices_of_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
                                                       Vertex_coordinates_submesh, squared=1):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh: numpy array of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., instanciate the dataset class to transformed_dataset,
    Get the first example in the dataset: template_submesh = transformed_dataset[0]
    then 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.
    
    Output: 
    - squared_l2_adjacent_vertices: list containing the squared Euclidean distance of all adjacent vertices of a submesh
    of a given pose of the towel 
    (this submesh should be obtained from the dataloader and is given by the list of vertex coordinates
    Vertex_coordinates_submesh)
    """
    submesh_num_vertices = submesh_num_vertices_vertical * submesh_num_vertices_horizontal
    squared_l2_adjacent_vertices = [
        squared_l2_distance_of_two_vertices_from_submesh(v_idx_submesh, w_idx_submesh, Vertex_coordinates_submesh, squared)
        for v_idx_submesh in range(submesh_num_vertices)
        for w_idx_submesh in idx_within_submesh_of_Allen_neighbours_within_submesh(
            v_idx_submesh, submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)]
    return squared_l2_adjacent_vertices

# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh() is in the file submesh_CHECKS.py

def squared_l2_distance_of_adjacent_vertices_of_submesh_tensor(submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
                                                       Vertex_coordinates_submesh, squared=1):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh: tensor of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., instanciate the dataset class to transformed_dataset,
    Get the first example in the dataset: template_submesh = transformed_dataset[0]
    then 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.
    
    Output: 
    - squared_l2_adjacent_vertices: tensor containing the squared Euclidean distance of all adjacent vertices of a submesh
    of a given pose of the towel 
    (this submesh should be obtained from the dataloader and is given by the list of vertex coordinates
    Vertex_coordinates_submesh)
    """
    submesh_num_vertices = submesh_num_vertices_vertical * submesh_num_vertices_horizontal
    squared_l2_adjacent_vertices = torch.tensor([
        squared_l2_distance_of_two_vertices_from_submesh_tensor(
            v_idx_submesh, w_idx_submesh, Vertex_coordinates_submesh, squared).item()
        for v_idx_submesh in range(submesh_num_vertices)
        for w_idx_submesh in idx_within_submesh_of_Allen_neighbours_within_submesh(
            v_idx_submesh, submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)], 
        dtype=Vertex_coordinates_submesh.dtype, requires_grad=Vertex_coordinates_submesh.requires_grad)
    return squared_l2_adjacent_vertices

# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_tensor() is in the file submesh_CHECKS.py

def n_pairs_of_adjacent_vertices_of_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    Output: 
    - n_pairs: number of pairs of adjacent vertices
    """
    submesh_num_vertices = submesh_num_vertices_vertical * submesh_num_vertices_horizontal
    n_pairs = 0
    for v_idx_submesh in range(submesh_num_vertices):
        for w_idx_submesh in idx_within_submesh_of_Allen_neighbours_within_submesh(
            v_idx_submesh, submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal):
            n_pairs += 1
    return n_pairs

def squared_l2_distance_of_adjacent_vertices_of_submesh_batch(args, Vertex_coordinates_submesh_batch):
    """ Apply squared_l2_distance_of_adjacent_vertices_of_submesh() to a batch.
    
    Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh_batch: tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex), 
    containing the num_coord_per_vertex vertex coordinates of the num_selected_vertices vertices in the submesh.
    E.g., instanciate the dataset class to transformed_dataset, create dataloaders, get first instance of dataloader and the vertex coordinates five labels as the correct format needed for Vertex_coordinates_submesh_batch:
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)
    labels = sample_batched['Vertex_coordinates'] 
    
    Output: 
    - square_distance_tensor: tensor of shape (batch_size, n_pairs) such that
    square_distance_tensor[i, :] contains the square (if args.squared==1) or non-squared (if args.squared==0) distance of the adjacent vertices of the submesh in Vertex_coordinates_submesh_batch[i, :, :]. 
    If args.normalize_distance_adj_vertices==1, this (square or not) distance is divided by the distance between the element of all coordinates 0 and the element of all coordinates 1.
    """
#     batch_size = Vertex_coordinates_submesh_batch.shape[0]
    num_selected_vertices=args.submesh_num_vertices_vertical*args.submesh_num_vertices_horizontal
    num_coord_per_vertex = Vertex_coordinates_submesh_batch.shape[1]//args.num_selected_vertices
    n_pairs = n_pairs_of_adjacent_vertices_of_submesh(args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal)
    square_distance_tensor = torch.zeros([args.batch_size, n_pairs], dtype=Vertex_coordinates_submesh_batch.dtype,
                                         requires_grad=False, device=Vertex_coordinates_submesh_batch.device)
    
    # I had to set
    # square_distance_tensor.requires_grad = False
    # instead of
    # square_distance_tensor.requires_grad = Vertex_coordinates_submesh_batch.requires_grad
    # to avoid the following error during training:
    # RuntimeError: leaf variable has been moved into the graph interior
    # More details here:
    # https://discuss.pytorch.org/t/leaf-variable-has-been-moved-into-the-graph-interior/18679

    for i in range(args.batch_size):
#         # tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex)
#         Vertex_coordinates_submesh = functions_data_processing.reshape_labels_back(
#             Vertex_coordinates_submesh_batch[i,:], batch_size=batch_size,
#             num_selected_vertices=num_selected_vertices, num_coord_per_vertex=num_coord_per_vertex)
        square_distance_tensor[i,:] = squared_l2_distance_of_adjacent_vertices_of_submesh_tensor(
            args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal,
            Vertex_coordinates_submesh=Vertex_coordinates_submesh_batch[i,:,:], squared=args.squared)
        if args.normalize_distance_adj_vertices==1:
            if args.squared==1:
                square_distance_tensor[i,:] = square_distance_tensor[i,:]/args.num_coord_per_vertex
            elif args.squared==0:
                square_distance_tensor[i,:] = square_distance_tensor[i,:]/np.sqrt(args.num_coord_per_vertex)
    return square_distance_tensor
        
        
# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_batch() is in the file submesh_CHECKS.py







###
### Angle between consecutive edges. All code made for tensors.
###
def vector_from_points(x, y):
    """Output: vector from x to y"""
    return y-x

# # Checks
# if __name__=='__main__':
#     x=torch.ones([1,3])
#     y=torch.zeros([1,3])
#     y[0,0], y[0,1], y[0,2] = 0, 1, 2
#     print("x:", x, "\ny:", y, "\nvector_from_points(x, y):", vector_from_points(x, y), "\nvector_from_points(x, y).shape:", vector_from_points(x, y).shape, "\n")
    
#     x=torch.ones([3])
#     y=torch.zeros([3])
#     y[0], y[1], y[2] = 0, 1, 2
#     print("x:", x, "\ny:", y, "\nvector_from_points(x, y):", vector_from_points(x, y), "\nvector_from_points(x, y).shape:", vector_from_points(x, y).shape, "\n")
    
def cos_angle(u, v):
    """ cosine of the angle between vectors u and v"""
    return torch.div(torch.dot(u, v),(torch.norm(u)*torch.norm(v)))
    
def sin_angle(u, v):
    """ sine of the angle between vectors u and v"""
#     print(u.shape)
#     u=torch.squeeze(u)
#     print(u.shape)
#     v=torch.squeeze(v)
    return torch.div(torch.norm(torch.cross(u, v)), (torch.norm(u)*torch.norm(v)))

# # Checks
# if __name__=='__main__':
#     x=torch.zeros([3])
#     y=torch.zeros([3])
#     y[0], y[1], y[2] = 1, 0, 0
#     z=torch.zeros([3])
#     z[0], z[1], z[2] = 0, 0, 1

#     u = vector_from_points(x, y)
#     v = vector_from_points(x, z)
#     print("u:", u, "\nv:", v, "\ntorch.dot(u, v):", torch.dot(u, v), "\ncos_angle(u, v):", cos_angle(u, v), "\n")
#     print("u:", u, "\nv:", v, "\ntorch.cross(u, v):", torch.cross(u, v), "\nsin_angle(u, v):", sin_angle(u, v), "\n")

#     z[0], z[1], z[2] = 2, 0, 0
#     v = vector_from_points(x, z)
#     print("u:", u, "\nv:", v, "\ntorch.dot(u, v):", torch.dot(u, v), "\ncos_angle(u, v):", cos_angle(u, v), "\n")
#     print("u:", u, "\nv:", v, "\ntorch.cross(u, v):", torch.cross(u, v), "\nsin_angle(u, v):", sin_angle(u, v), "\n")
    
def cos_angle_given_3_points(a, b, c):
    """ Given 3D points a, b, c, return the cosine of the angle between the vectors a-->b and b-->c. """
    u = vector_from_points(a, b)
    v = vector_from_points(b, c)
    return cos_angle(u, v)

def sine_angle_given_3_points(a, b, c):
    """ Given 3D points a, b, c, return the sine of the angle between the vectors a-->b and b-->c. """
    u = vector_from_points(a, b)
    v = vector_from_points(b, c)
    return sin_angle(u, v)

# # Checks
# if __name__=='__main__':
#     a=torch.tensor([1, 0, 0]).float()
#     b=torch.tensor([0, 0, 0]).float()
#     c=torch.tensor([0, 0, 1]).float() 
#     print("a:", a, "\nb:", b, "\nc:", c, "\nsine_angle_given_3_points(a, b, c):", sine_angle_given_3_points(a, b, c), "\n")    
#     print("a:", a, "\nb:", b, "\nc:", c, "\ncos_angle_given_3_points(a, b, c):", cos_angle_given_3_points(a, b, c), "\n")

#     a=torch.tensor([1, 0, 0]).float()
#     b=torch.tensor([0, 0, 0]).float()
#     c=torch.tensor([-1, 0, 0]).float() 
#     print("a:", a, "\nb:", b, "\nc:", c, "\nsine_angle_given_3_points(a, b, c):", sine_angle_given_3_points(a, b, c), "\n")
#     print("a:", a, "\nb:", b, "\nc:", c, "\ncos_angle_given_3_points(a, b, c):", cos_angle_given_3_points(a, b, c), "\n")
    
#     a=torch.tensor([ -0.8455,   0.8455,  12.0570])
#     b=torch.tensor([ -0.6147,   1.0874,  11.8395])
#     c=torch.tensor([ -0.4634,   2.7659,  11.7806])
#     print("a:", a, "\nb:", b, "\nc:", c, "\nsine_angle_given_3_points(a, b, c):", sine_angle_given_3_points(a, b, c), "\n")
#     print("a:", a, "\nb:", b, "\nc:", c, "\ncos_angle_given_3_points(a, b, c):", cos_angle_given_3_points(a, b, c), "\n")
    
def consecutive_3_vertices_horizontally_in_mesh(matrix_coord, num_vertices_width = 103, num_vertices_height = 52):
    """ Input: matrix_coord is a tuple of the form (x, y), 
    where these are the matrix coordinates of a vertex of a mesh.
    These (x, y) matrix coordinates correspond to vertical downwards and horizontal rightwards, respectively.
    
    Output: list of matrix coordinates (as tuples (x, y)) of the given vertex and the next two vertices moving horizontally right-wards in the mesh.
    If the vertex is either on the last or second-to-last column, the output is [None, None, None].
    
    Note: (original mesh vs submesh) 
    The mesh all vertices are considered in for the matrix coordinates and for the notion of neighbour is the mesh with:
    - number of vertices in horizontal direction: num_vertices_width.
    - number of vertices in vertical direction: num_vertices_height. 
    """
    if matrix_coord[1]+2 < num_vertices_width:
        return [(matrix_coord[0], matrix_coord[1]+i) for i in range(3)]
    else:
        return [None, None, None]

def CHECK_consecutive_3_vertices_horizontally_in_mesh():
    """Function to check that consecutive_3_vertices_horizontally_in_mesh() works"""
    list_of_matrix_coord = [(0, 0), (0, 102), (1, 0), (1, 101), (51, 0), (51, 100)]
    num_vertices_width = 103
    num_vertices_height = 52
    print("num_vertices_width:", num_vertices_width, "; num_vertices_height:", num_vertices_height)
    for matrix_coord in list_of_matrix_coord:
        print('consecutive_3_vertices_horizontally_in_mesh from', matrix_coord, ' on: ', end='')
        print(consecutive_3_vertices_horizontally_in_mesh(matrix_coord, num_vertices_width, num_vertices_height))
    print()

# CHECK_consecutive_3_vertices_horizontally_in_mesh()

def idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh(v_idx_submesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3):
    """ Input: 
    - v_idx_mesh: index within the submesh of a vertex v of the submesh.
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    This function builds the list of the given vertex and the next two vertices moving horizontally right-wards in the mesh.
    Output: 
    - list of indices within the submesh of these 3 vertices. """   
    # Convert index of v within the submesh to matrix coordinates within the submesh
    v_matrix_coord_submesh = matrix_coord_from_idx(v_idx_submesh, num_vertices_width = submesh_num_vertices_horizontal)
    
    # Find list of matrix coordinates within submesh of 3 consecutive vertices
    consecutive3_coord_submesh = consecutive_3_vertices_horizontally_in_mesh(
        v_matrix_coord_submesh, num_vertices_width = submesh_num_vertices_horizontal,
        num_vertices_height = submesh_num_vertices_vertical)
    
    # Convert matrix coordinates of neighbours within submesh to indices within submesh
    if consecutive3_coord_submesh[0] is not None:
        consecutive3_idx_submesh = [idx_from_matrix_coord(matrix_coord, num_vertices_width = submesh_num_vertices_horizontal) 
                              for matrix_coord in consecutive3_coord_submesh]
    else: 
        consecutive3_idx_submesh = [None, None, None]
    
    return consecutive3_idx_submesh


def CHECK_idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh():
    """Function to check that idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh() works"""
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 3
    print("submesh_num_vertices_vertical:", submesh_num_vertices_vertical, "; submesh_num_vertices_horizontal:", submesh_num_vertices_horizontal)

    for v_idx_submesh in range(submesh_num_vertices_vertical*submesh_num_vertices_horizontal):
        print('3 consecutive vertices horizontally from ', v_idx_submesh, 'on --> ' , end='')
        print(idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh(
            v_idx_submesh = v_idx_submesh, 
            submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal))

# CHECK_idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh()

def idx_within_mesh_of_consecutive_3_vertices_horizontally_within_submesh(v_idx_mesh, submesh_idx_within_mesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3):
    """ Input: 
    - v_idx_mesh: index within the original mesh of a vertex v of the submesh.
    - submesh_idx_within_mesh: list containing the indices within the (reordered, not directly from Blender) original mesh 
    of the vertices that form the submesh.
    This is the output of submesh_idx_from_num_vertices_in_each_direction().
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    This function builds the list of the given vertex and the next two vertices moving horizontally right-wards in the mesh.
    Output: 
    - list of indices within the original mesh of these 3 vertices. 
    """
    # Convert index of v within the original mesh to index within the submesh
    v_idx_submesh = submesh_idx_within_mesh.index(v_idx_mesh)
    
    # Find indices within the submesh of the 3 consecutive vertices
    consecutive3_idx_submesh=idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh(v_idx_submesh,
                                                          submesh_num_vertices_vertical = submesh_num_vertices_vertical,
                                                          submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
    
    # Convert indices of neighbours within the submesh to indices within the original mesh
    if consecutive3_idx_submesh[0] is not None:
        consecutive3_idx_mesh = [submesh_idx_within_mesh[consecutive3_idx_submesh] for consecutive3_idx_submesh in consecutive3_idx_submesh]
    else:
        consecutive3_idx_mesh = [None, None, None]
    
    return consecutive3_idx_mesh

def CHECK_idx_within_mesh_of_consecutive_3_vertices_horizontally_within_submesh():
    """Function to check that idx_within_mesh_of_consecutive_3_vertices_horizontally_within_submesh() works"""
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 3
    print("submesh_num_vertices_vertical:", submesh_num_vertices_vertical, "; submesh_num_vertices_horizontal:", submesh_num_vertices_horizontal)

    # Find idx within the original mesh of all the vertices of the submesh
    submesh_idx_within_mesh = submesh_idx_from_num_vertices_in_each_direction(
        submesh_num_vertices_vertical = submesh_num_vertices_vertical,
        submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
    print('submesh_idx_within_mesh:', submesh_idx_within_mesh)
    for v_idx_mesh in submesh_idx_within_mesh:
        print('3 consecutive vertices horizontally from ', v_idx_mesh, 'on --> ' , end='')
        print(idx_within_mesh_of_consecutive_3_vertices_horizontally_within_submesh(
            v_idx_mesh = v_idx_mesh, submesh_idx_within_mesh = submesh_idx_within_mesh, 
            submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal))

# CHECK_idx_within_mesh_of_consecutive_3_vertices_horizontally_within_submesh()

def sin_3_vert_from_submesh(a_idx_submesh, b_idx_submesh, c_idx_submesh, Vertex_coordinates_submesh):
    """ Input: 
    - a_idx_submesh: index within the submesh of a vertex a.
    - b_idx_submesh: index within the submesh of a vertex b.
    - c_idx_submesh: index within the submesh of a vertex c.
    - Vertex_coordinates_submesh: tensor of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., instanciate the dataset class to transformed_dataset,
    Get the first example in the dataset: template_submesh = transformed_dataset[0]
    then 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.

    Output:
    - Sine of the angle between the vectors a-->b and b-->c.

    See submesh_CHECKS.CHECK_sin_3_vert_from_submesh() for an example on how to use it."""
    
    a_Vertex_coordinates = Vertex_coordinates_submesh[a_idx_submesh, :]
    b_Vertex_coordinates = Vertex_coordinates_submesh[b_idx_submesh, :]
    c_Vertex_coordinates = Vertex_coordinates_submesh[c_idx_submesh, :]
    return sine_angle_given_3_points(a_Vertex_coordinates, b_Vertex_coordinates, c_Vertex_coordinates)
    
# CHECK_sin_3_vert_from_submesh() is in the file submesh_CHECKS.py  

def sin_3horizConsec_vert_from_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
                                                       Vertex_coordinates_submesh):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh: tensor of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., instanciate the dataset class to transformed_dataset,
    Get the first example in the dataset: template_submesh = transformed_dataset[0]
    then 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.
    
    Output: 
    - sine_horiz_adjacent_edges: tensor containing the absolute value of the sine of the angle between the vectors a-->b and b-->c, 
    for all triples (a, b, c) of horizontally consecutive vertices (rightwards) in the submesh of a given pose of the towel
    (this submesh should be obtained from the dataloader and is given by the list of vertex coordinates
    Vertex_coordinates_submesh)
    """
    submesh_num_vertices = submesh_num_vertices_vertical * submesh_num_vertices_horizontal
    return torch.abs(torch.tensor([
        sin_3_vert_from_submesh(a_idx_submesh, b_idx_submesh, c_idx_submesh, Vertex_coordinates_submesh).item()
        for a_idx_submesh_1 in range(submesh_num_vertices)
        for [a_idx_submesh, b_idx_submesh, c_idx_submesh] in [idx_within_submesh_of_consecutive_3_vertices_horizontally_within_submesh(
            a_idx_submesh_1, submesh_num_vertices_vertical, submesh_num_vertices_horizontal)] if a_idx_submesh is not None], 
        dtype=Vertex_coordinates_submesh.dtype, requires_grad=Vertex_coordinates_submesh.requires_grad))

# CHECK_sin_3horizConsec_vert_from_submesh() is in the file submesh_CHECKS.py  

def n_horiz_consec_edges_of_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    Output: 
    - n_pairs: number of pairs of horizontally consecutive edges
    
    Since the last two columns of vertices of the mesh do not have two vertices on the right, 
    those are the 2 columns that do not produce a pair of horizontally consecutive edges.
    """
    return submesh_num_vertices_vertical * (submesh_num_vertices_horizontal-2)
    
def sin_3horizConsec_vert_from_submesh_batch(args, Vertex_coordinates_submesh_batch):
    """ Apply sin_3horizConsec_vert_from_submesh() to a batch.
    
    Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh_batch: tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex), 
    containing the num_coord_per_vertex vertex coordinates of the num_selected_vertices vertices in the submesh.
    E.g., instanciate the dataset class to transformed_dataset, create dataloaders, get first instance of dataloader and the vertex coordinates five labels as the correct format needed for Vertex_coordinates_submesh_batch:
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)
    labels = sample_batched['Vertex_coordinates'] 
    
    Output: 
    - sine_horiz_adjacent_edges: tensor of shape (batch_size, n_pairs) such that
    square_distance_tensor[i, :] contains the absolute value of the sine of the angle between the vectors a-->b and b-->c, 
    for all triples (a, b, c) of horizontally consecutive vertices (rightwards)
    of the submesh in Vertex_coordinates_submesh_batch[i, :, :]. 
    """
#     batch_size = Vertex_coordinates_submesh_batch.shape[0]
    num_selected_vertices=args.submesh_num_vertices_vertical*args.submesh_num_vertices_horizontal
    num_coord_per_vertex = Vertex_coordinates_submesh_batch.shape[1]//args.num_selected_vertices
    n_pairs = n_horiz_consec_edges_of_submesh(args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal)
    sine_horiz_adjacent_edges = torch.zeros([args.batch_size, n_pairs], dtype=Vertex_coordinates_submesh_batch.dtype,
                                         requires_grad=False, device=Vertex_coordinates_submesh_batch.device)
    # I had to set
    # square_distance_tensor.requires_grad = False
    # instead of
    # square_distance_tensor.requires_grad = Vertex_coordinates_submesh_batch.requires_grad
    # to avoid the following error during training:
    # RuntimeError: leaf variable has been moved into the graph interior
    # More details here:
    # https://discuss.pytorch.org/t/leaf-variable-has-been-moved-into-the-graph-interior/18679

    for i in range(args.batch_size):
#         # tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex)
#         Vertex_coordinates_submesh = functions_data_processing.reshape_labels_back(
#             Vertex_coordinates_submesh_batch[i,:], batch_size=batch_size,
#             num_selected_vertices=num_selected_vertices, num_coord_per_vertex=num_coord_per_vertex)
        sine_horiz_adjacent_edges[i,:] = sin_3horizConsec_vert_from_submesh(
            args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal, Vertex_coordinates_submesh_batch[i,:,:])
    return sine_horiz_adjacent_edges
        
# CHECK_sin_3horizConsec_vert_from_submesh_batch() is in the file submesh_CHECKS.py

### Code adapted from horizontally to vertically adjacent edges

def consecutive_3_vertices_vertically_in_mesh(matrix_coord, num_vertices_width = 103, num_vertices_height = 52):
    """ Input: matrix_coord is a tuple of the form (x, y), 
    where these are the matrix coordinates of a vertex of a mesh.
    These (x, y) matrix coordinates correspond to vertical downwards and horizontal rightwards, respectively.
    
    Output: list of matrix coordinates (as tuples (x, y)) of the given vertex and the next two vertices moving vertically down-wards in the mesh.
    If the vertex is either on the last or second-to-last row, the output is [None, None, None].
    
    Note: (original mesh vs submesh) 
    The mesh all vertices are considered in for the matrix coordinates and for the notion of neighbour is the mesh with:
    - number of vertices in horizontal direction: num_vertices_width.
    - number of vertices in vertical direction: num_vertices_height. 
    """
    if matrix_coord[0]+2 < num_vertices_height:
        return [(matrix_coord[0]+i, matrix_coord[1]) for i in range(3)]
    else:
        return [None, None, None]

def CHECK_consecutive_3_vertices_vertically_in_mesh():
    """Function to check that consecutive_3_vertices_vertically_in_mesh() works"""
    list_of_matrix_coord = [(0, 0), (51, 0), (50, 0), (50, 102), (4, 102)]
    num_vertices_width = 103
    num_vertices_height = 52
    print("num_vertices_width:", num_vertices_width, "; num_vertices_height:", num_vertices_height)
    for matrix_coord in list_of_matrix_coord:
        print('consecutive_3_vertices_vertically_in_mesh from', matrix_coord, ' on: ', end='')
        print(consecutive_3_vertices_vertically_in_mesh(matrix_coord, num_vertices_width, num_vertices_height))
    print()

# CHECK_consecutive_3_vertices_vertically_in_mesh()

def idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh(v_idx_submesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3):
    """ Input: 
    - v_idx_mesh: index within the submesh of a vertex v of the submesh.
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    This function builds the list of the given vertex and the next two vertices moving vertically down-wards in the mesh.
    Output: 
    - list of indices within the submesh of these 3 vertices. """   
    # Convert index of v within the submesh to matrix coordinates within the submesh
    v_matrix_coord_submesh = matrix_coord_from_idx(v_idx_submesh, num_vertices_width = submesh_num_vertices_horizontal)
    
    # Find list of matrix coordinates within submesh of 3 consecutive vertices
    consecutive3_coord_submesh = consecutive_3_vertices_vertically_in_mesh(
        v_matrix_coord_submesh, num_vertices_width = submesh_num_vertices_horizontal,
        num_vertices_height = submesh_num_vertices_vertical)
    
    # Convert matrix coordinates of neighbours within submesh to indices within submesh
    if consecutive3_coord_submesh[0] is not None:
        consecutive3_idx_submesh = [idx_from_matrix_coord(matrix_coord, num_vertices_width = submesh_num_vertices_horizontal) 
                              for matrix_coord in consecutive3_coord_submesh]
    else: 
        consecutive3_idx_submesh = [None, None, None]
    
    return consecutive3_idx_submesh

def CHECK_idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh():
    """Function to check that idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh() works"""
    submesh_num_vertices_vertical = 5
    submesh_num_vertices_horizontal = 3
    print("submesh_num_vertices_vertical:", submesh_num_vertices_vertical, "; submesh_num_vertices_horizontal:", submesh_num_vertices_horizontal)

    for v_idx_submesh in range(submesh_num_vertices_vertical*submesh_num_vertices_horizontal):
        print('3 consecutive vertices horizontally from ', v_idx_submesh, 'on --> ' , end='')
        print(idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh(
            v_idx_submesh = v_idx_submesh, 
            submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal))

# CHECK_idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh()

def idx_within_mesh_of_consecutive_3_vertices_vertically_within_submesh(v_idx_mesh, submesh_idx_within_mesh,
                                                          submesh_num_vertices_vertical = 2,
                                                          submesh_num_vertices_horizontal = 3):
    """ Input: 
    - v_idx_mesh: index within the original mesh of a vertex v of the submesh.
    - submesh_idx_within_mesh: list containing the indices within the (reordered, not directly from Blender) original mesh 
    of the vertices that form the submesh.
    This is the output of submesh_idx_from_num_vertices_in_each_direction().
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    This function builds the list of the given vertex and the next two vertices moving vertically down-wards in the mesh.
    Output: 
    - list of indices within the original mesh of these 3 vertices. 
    """
    # Convert index of v within the original mesh to index within the submesh
    v_idx_submesh = submesh_idx_within_mesh.index(v_idx_mesh)
    
    # Find indices within the submesh of the 3 consecutive vertices
    consecutive3_idx_submesh=idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh(v_idx_submesh,
                                                          submesh_num_vertices_vertical = submesh_num_vertices_vertical,
                                                          submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
    
    # Convert indices of neighbours within the submesh to indices within the original mesh
    if consecutive3_idx_submesh[0] is not None:
        consecutive3_idx_mesh = [submesh_idx_within_mesh[consecutive3_idx_submesh] for consecutive3_idx_submesh in consecutive3_idx_submesh]
    else:
        consecutive3_idx_mesh = [None, None, None]
    
    return consecutive3_idx_mesh

def CHECK_idx_within_mesh_of_consecutive_3_vertices_vertically_within_submesh():
    """Function to check that idx_within_mesh_of_consecutive_3_vertices_vertically_within_submesh() works"""
    submesh_num_vertices_vertical = 5
    submesh_num_vertices_horizontal = 3
    print("submesh_num_vertices_vertical:", submesh_num_vertices_vertical, "; submesh_num_vertices_horizontal:", submesh_num_vertices_horizontal)

    # Find idx within the original mesh of all the vertices of the submesh
    submesh_idx_within_mesh = submesh_idx_from_num_vertices_in_each_direction(
        submesh_num_vertices_vertical = submesh_num_vertices_vertical,
        submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
    print('submesh_idx_within_mesh:', submesh_idx_within_mesh)
    for v_idx_mesh in submesh_idx_within_mesh:
        print('3 consecutive vertices vertically from ', v_idx_mesh, 'on --> ' , end='')
        print(idx_within_mesh_of_consecutive_3_vertices_vertically_within_submesh(
            v_idx_mesh = v_idx_mesh, submesh_idx_within_mesh = submesh_idx_within_mesh, 
            submesh_num_vertices_vertical = submesh_num_vertices_vertical,
            submesh_num_vertices_horizontal = submesh_num_vertices_horizontal))

# CHECK_idx_within_mesh_of_consecutive_3_vertices_vertically_within_submesh()

def sin_3verConsec_vert_from_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
                                                       Vertex_coordinates_submesh):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh: tensor of shape num_vertices_in_submesh x D, 
    containing the D vertex coordinates of the vertices in the submesh.
    If we are dealing with xyz coordinates, then D=3, if dealing with uv, then D=2.
    E.g., instanciate the dataset class to transformed_dataset,
    Get the first example in the dataset: template_submesh = transformed_dataset[0]
    then 
    template_submesh['Vertex_coordinates'] or 
    template_submesh['uv'] 
    have the format needed for Vertex_coordinates_submesh.
    
    Output: 
    - sine_ver_adjacent_edges: tensor containing the absolute value of the sine of the angle between the vectors a-->b and b-->c, 
    for all triples (a, b, c) of vertically consecutive vertices (downwards) in the submesh of a given pose of the towel
    (this submesh should be obtained from the dataloader and is given by the list of vertex coordinates
    Vertex_coordinates_submesh)
    """
    submesh_num_vertices = submesh_num_vertices_vertical * submesh_num_vertices_horizontal
    return torch.abs(torch.tensor([
        sin_3_vert_from_submesh(a_idx_submesh, b_idx_submesh, c_idx_submesh, Vertex_coordinates_submesh).item()
        for a_idx_submesh_1 in range(submesh_num_vertices)
        for [a_idx_submesh, b_idx_submesh, c_idx_submesh] in [idx_within_submesh_of_consecutive_3_vertices_vertically_within_submesh(
            a_idx_submesh_1, submesh_num_vertices_vertical, submesh_num_vertices_horizontal)] if a_idx_submesh is not None], 
        dtype=Vertex_coordinates_submesh.dtype, requires_grad=Vertex_coordinates_submesh.requires_grad))

# CHECK_sin_3verConsec_vert_from_submesh() is in the file submesh_CHECKS.py  

def n_ver_consec_edges_of_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal):
    """ Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    
    Output: 
    - n_pairs: number of pairs of vertically consecutive edges
    
    Since the last two rows of vertices of the mesh do not have two vertices below, 
    those are the 2 rows that do not produce a pair of vertically consecutive edges.
    """
    return (submesh_num_vertices_vertical-2) * submesh_num_vertices_horizontal

def sin_3verConsec_vert_from_submesh_batch(args, Vertex_coordinates_submesh_batch):
    """ Apply sin_3verConsec_vert_from_submesh() to a batch.
    
    Input:
    - submesh_num_vertices_horizontal: number of vertices in the submesh in the horizontal direction.
    - submesh_num_vertices_vertical: number of vertices in the submesh in the vertical direction.
    - Vertex_coordinates_submesh_batch: tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex), 
    containing the num_coord_per_vertex vertex coordinates of the num_selected_vertices vertices in the submesh.
    E.g., instanciate the dataset class to transformed_dataset, create dataloaders, get first instance of dataloader and the vertex coordinates five labels as the correct format needed for Vertex_coordinates_submesh_batch:
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)
    labels = sample_batched['Vertex_coordinates'] 
    
    Output: 
    - sine_ver_adjacent_edges: tensor of shape (batch_size, n_pairs) such that
    square_distance_tensor[i, :] contains the absolute value of the sine of the angle between the vectors a-->b and b-->c, 
    for all triples (a, b, c) of vertically consecutive vertices (downwards)
    of the submesh in Vertex_coordinates_submesh_batch[i, :, :]. 
    """
#     batch_size = Vertex_coordinates_submesh_batch.shape[0]
    num_selected_vertices=args.submesh_num_vertices_vertical*args.submesh_num_vertices_horizontal
    num_coord_per_vertex = Vertex_coordinates_submesh_batch.shape[1]//args.num_selected_vertices
    n_pairs = n_ver_consec_edges_of_submesh(args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal)
    sine_ver_adjacent_edges = torch.zeros([args.batch_size, n_pairs], dtype=Vertex_coordinates_submesh_batch.dtype,
                                         requires_grad=False, device=Vertex_coordinates_submesh_batch.device)
    # I had to set
    # square_distance_tensor.requires_grad = False
    # instead of
    # square_distance_tensor.requires_grad = Vertex_coordinates_submesh_batch.requires_grad
    # to avoid the following error during training:
    # RuntimeError: leaf variable has been moved into the graph interior
    # More details here:
    # https://discuss.pytorch.org/t/leaf-variable-has-been-moved-into-the-graph-interior/18679

    for i in range(args.batch_size):
#         # tensor of shape (batch_size, num_selected_vertices, num_coord_per_vertex)
#         Vertex_coordinates_submesh = functions_data_processing.reshape_labels_back(
#             Vertex_coordinates_submesh_batch[i,:], batch_size=batch_size,
#             num_selected_vertices=num_selected_vertices, num_coord_per_vertex=num_coord_per_vertex)
        sine_ver_adjacent_edges[i,:] = sin_3verConsec_vert_from_submesh(
            args.submesh_num_vertices_vertical, args.submesh_num_vertices_horizontal, Vertex_coordinates_submesh_batch[i,:,:])
    return sine_ver_adjacent_edges
        
# CHECK_sin_3verConsec_vert_from_submesh_batch() is in the file submesh_CHECKS.py
   
def is_in_mesh_contour(idx, args):
    """ Given an idx of a vertex in the mesh, return True or False indicating whether that vertex is in the contour of the mesh.
    This does not depend on the embedding in 3D nor in the projection in 2D. Here we only use the mesh graph."""
    # x --> vertical downwards; y --> horizontal rightwards
    x, y = matrix_coord_from_idx(idx, num_vertices_width = args.submesh_num_vertices_horizontal) 
    return (x==0) or (x==args.submesh_num_vertices_vertical-1) or (y==0) or (y==args.submesh_num_vertices_horizontal-1)
    
def contour_of_mesh(args):
    """ Output: list of the idx of all the vertices in contour of the mesh.
    This does not depend on the embedding in 3D nor in the projection in 2D. Here we only use the mesh graph."""
    contour_list = [idx for idx in range(args.num_selected_vertices) if is_in_mesh_contour(idx, args)]
    return contour_list    