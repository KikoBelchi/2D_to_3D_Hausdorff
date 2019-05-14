import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


import data_loading
import functions_data_processing
import functions_plot
import functions_train
import submesh

def instanciate_dataset_example(submesh_num_vertices_vertical=None, submesh_num_vertices_horizontal=None,
                           predict_uv_or_xyz=None, return_args=0):
#     num_vertices = functions_data_processing.num_vertices_in_the_mesh()
    num_vertices_height = (2**2)*13
    num_vertices_width = 103
    
    # Parser for entering training setting arguments and setting default ones
    args = functions_data_processing.parser()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if submesh_num_vertices_vertical is not None:
        args.submesh_num_vertices_vertical = submesh_num_vertices_vertical
    if not (submesh_num_vertices_horizontal is None):
        args.submesh_num_vertices_horizontal = submesh_num_vertices_horizontal
    if not (predict_uv_or_xyz is None):
        args.predict_uv_or_xyz = predict_uv_or_xyz
            
    # Instanciate un-normalized dataset
    transformed_dataset = data_loading.instanciate_dataset(args)
    
    if return_args==0:
        return transformed_dataset
    else:
        return transformed_dataset, args

def create_template_submesh(submesh_num_vertices_vertical=None, submesh_num_vertices_horizontal=None,
                           predict_uv_or_xyz=None, return_args=0, template_idx=0):
    if return_args==0:
        transformed_dataset = instanciate_dataset_example(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                                      submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                                      predict_uv_or_xyz=predict_uv_or_xyz, return_args=return_args)        
    else:
        transformed_dataset, args = instanciate_dataset_example(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                                      submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                                      predict_uv_or_xyz=predict_uv_or_xyz, return_args=return_args)        
    template_submesh = transformed_dataset[template_idx]
    if return_args==0:
        return template_submesh
    else:
        return template_submesh, args

def CHECK_create_template_submesh(predict_uv_or_xyz='xyz'):
    template_submesh = create_template_submesh(predict_uv_or_xyz=predict_uv_or_xyz)
    if predict_uv_or_xyz=='xyz':
        # Plot 3D vertex coordinates
        X = template_submesh['Vertex_coordinates'][:, 0]
        Y = template_submesh['Vertex_coordinates'][:, 1]
        Z = template_submesh['Vertex_coordinates'][:, 2] 
        swap_axes=1
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

        # Plot RGB input image
        plt.figure()
        image = io.imread(template_submesh['img_name'])
        plt.imshow(image)
        plt.show()
    elif predict_uv_or_xyz=='uv':
        transformed_dataset = instanciate_dataset_example(predict_uv_or_xyz=predict_uv_or_xyz) 
        observation_within_dataset=0
        functions_plot.plot_RGB_and_landmarks_from_dataset(dataset = transformed_dataset, observation_within_dataset = observation_within_dataset, transformed_image_or_not=0, uv_normalization=0)
        plt.show()

# CHECK_create_template_submesh(predict_uv_or_xyz='xyz')
# CHECK_create_template_submesh(predict_uv_or_xyz='uv')


def CHECK_squared_l2_distance_of_two_vertices_from_submesh(predict_uv_or_xyz='xyz'):
    template_submesh = create_template_submesh(predict_uv_or_xyz=predict_uv_or_xyz)
#     print(template_submesh)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = template_submesh['Vertex_coordinates']
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = template_submesh['uv']
    print(Vertex_coordinates_submesh)
    
    v_idx_submesh = 0
    w_idx_submesh = 1
    print(submesh.squared_l2_distance_of_two_vertices_from_submesh(v_idx_submesh = v_idx_submesh, w_idx_submesh = w_idx_submesh,
                                                          Vertex_coordinates_submesh = Vertex_coordinates_submesh))

# CHECK_squared_l2_distance_of_two_vertices_from_submesh(predict_uv_or_xyz='xyz')
# CHECK_squared_l2_distance_of_two_vertices_from_submesh(predict_uv_or_xyz='uv')

def CHECK_squared_l2_distance_of_two_vertices_from_submesh_tensor(predict_uv_or_xyz='xyz'):
    template_submesh = create_template_submesh(predict_uv_or_xyz=predict_uv_or_xyz)
#     print(template_submesh)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['Vertex_coordinates'])
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['uv'])
    print(Vertex_coordinates_submesh)
    
    v_idx_submesh = 0
    w_idx_submesh = 1
    print(submesh.squared_l2_distance_of_two_vertices_from_submesh_tensor(
        v_idx_submesh = v_idx_submesh, w_idx_submesh = w_idx_submesh,
        Vertex_coordinates_submesh = Vertex_coordinates_submesh))

# CHECK_squared_l2_distance_of_two_vertices_from_submesh_tensor(predict_uv_or_xyz='xyz')
# CHECK_squared_l2_distance_of_two_vertices_from_submesh_tensor(predict_uv_or_xyz='uv')

def CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh(predict_uv_or_xyz='xyz'):
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 3
    template_submesh = create_template_submesh(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                               submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                               predict_uv_or_xyz=predict_uv_or_xyz)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = template_submesh['Vertex_coordinates']
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = template_submesh['uv']
    print(Vertex_coordinates_submesh)   
    print(submesh.squared_l2_distance_of_adjacent_vertices_of_submesh(submesh_num_vertices_vertical,
                                                                      submesh_num_vertices_horizontal,
                                                                      Vertex_coordinates_submesh = Vertex_coordinates_submesh))

# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh(predict_uv_or_xyz='xyz')
# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh(predict_uv_or_xyz='uv')

def CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_tensor(predict_uv_or_xyz='xyz'):
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 3
    template_submesh = create_template_submesh(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                               submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                               predict_uv_or_xyz=predict_uv_or_xyz)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['Vertex_coordinates'])
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['uv'])
#     print(template_submesh['img_name'])    
#     print(Vertex_coordinates_submesh)   
    print(submesh.squared_l2_distance_of_adjacent_vertices_of_submesh_tensor(
        submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
        Vertex_coordinates_submesh = Vertex_coordinates_submesh))

# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_tensor(predict_uv_or_xyz='xyz')
# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_tensor(predict_uv_or_xyz='uv')




def CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_batch(predict_uv_or_xyz='xyz'):
    """
    Instanciate dataste
    Create dataloaders
    Iterate over dataloaders
    Get one batch of labels
    Compute distance between adjacent vertices
    Reshape labels and append to them the distances between adjacent vertices
    """
    transformed_dataset = instanciate_dataset_example(predict_uv_or_xyz=predict_uv_or_xyz) 
    args = functions_data_processing.parser()
    transformed_dataset_parts, dataset_sizes, dataloaders = data_loading.random_split_notMixingSequences( 
            dataset=transformed_dataset, args=args)
    print('Length of transformed_dataset:', len(transformed_dataset))
    print('Length of each part:', dataset_sizes)
    
#     for batch_idx, sample_batched in enumerate(dataloaders['train'], 1):
#         if batch_idx==1:
# Pick the first validation batch
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)

    inputs = sample_batched['image']
    inputs = inputs.to(args.device)
        
    labels, labels_uv_coord, labels_xyz_coord = functions_train.create_labels_and_prediction(args, sample_batched=sample_batched)
    print("labels_xyz_coord.shape:", labels_xyz_coord.shape)
    
    print(submesh.squared_l2_distance_of_adjacent_vertices_of_submesh_batch(args, Vertex_coordinates_submesh_batch=labels_xyz_coord))

# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_batch(predict_uv_or_xyz='xyz')

# CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_batch(predict_uv_or_xyz='uv') 
# to run CHECK_squared_l2_distance_of_adjacent_vertices_of_submesh_batch(predict_uv_or_xyz='uv'), type on command line:
# python submesh_CHECKS.py --predict-uv-or-xyz 'uv' --neighbour-dist 'uv'







### Angles between adjacent edges in the mesh
def CHECK_sin_3_vert_from_submesh(predict_uv_or_xyz='xyz', submesh_num_vertices_vertical=None,
                                             submesh_num_vertices_horizontal=None,     
                                             a_idx_submesh = 0, b_idx_submesh = 1, c_idx_submesh = 2):
    template_submesh = create_template_submesh(submesh_num_vertices_vertical, submesh_num_vertices_horizontal, predict_uv_or_xyz)
#     print(template_submesh)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['Vertex_coordinates'])
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['uv'])
#     print(Vertex_coordinates_submesh)
    print("submesh_num_vertices_vertical, submesh_num_vertices_horizontal:", 
          submesh_num_vertices_vertical, submesh_num_vertices_horizontal)
    print("vertex_ids:", a_idx_submesh, b_idx_submesh, c_idx_submesh)
    print("sin:", submesh.sin_3_vert_from_submesh(a_idx_submesh, b_idx_submesh, c_idx_submesh, Vertex_coordinates_submesh))

# CHECK_sin_3_vert_from_submesh(predict_uv_or_xyz='xyz', submesh_num_vertices_vertical=2, submesh_num_vertices_horizontal=3)
# CHECK_sin_3_vert_from_submesh(predict_uv_or_xyz='xyz', submesh_num_vertices_vertical=2, submesh_num_vertices_horizontal=3, a_idx_submesh = 0, b_idx_submesh = 1, c_idx_submesh = 4)

def CHECK_sin_3horizConsec_vert_from_submesh(predict_uv_or_xyz='xyz'):
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 3
    template_submesh = create_template_submesh(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                               submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                               predict_uv_or_xyz=predict_uv_or_xyz)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['Vertex_coordinates'])
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['uv'])
#     print(template_submesh['img_name'])    
#     print(Vertex_coordinates_submesh)   
    print(submesh.sin_3horizConsec_vert_from_submesh(
        submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
        Vertex_coordinates_submesh = Vertex_coordinates_submesh))

# CHECK_sin_3horizConsec_vert_from_submesh(predict_uv_or_xyz='xyz')

def CHECK_sin_3verConsec_vert_from_submesh(predict_uv_or_xyz='xyz'):
    submesh_num_vertices_vertical = 5
    submesh_num_vertices_horizontal = 3
    template_submesh = create_template_submesh(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                               submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                               predict_uv_or_xyz=predict_uv_or_xyz)
    if predict_uv_or_xyz=='xyz':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['Vertex_coordinates'])
    elif predict_uv_or_xyz=='uv':
        Vertex_coordinates_submesh = torch.from_numpy(template_submesh['uv'])
#     print(template_submesh['img_name'])    
#     print(Vertex_coordinates_submesh)   
    print(submesh.sin_3verConsec_vert_from_submesh(
        submesh_num_vertices_vertical, submesh_num_vertices_horizontal,
        Vertex_coordinates_submesh = Vertex_coordinates_submesh))

# CHECK_sin_3verConsec_vert_from_submesh(predict_uv_or_xyz='xyz')

def CHECK_sin_3horizConsec_vert_from_submesh_batch(predict_uv_or_xyz='xyz'):
    """
    Instanciate dataste
    Create dataloaders
    Iterate over dataloaders
    Get one batch of labels
    Compute sine of angles between adjacent edges
    Reshape labels and append to them the distances between adjacent vertices
    """
    transformed_dataset = instanciate_dataset_example(predict_uv_or_xyz=predict_uv_or_xyz) 
    args = functions_data_processing.parser()
    transformed_dataset_parts, dataset_sizes, dataloaders = data_loading.random_split_notMixingSequences( 
            dataset=transformed_dataset, args=args)
    print('Length of transformed_dataset:', len(transformed_dataset))
    print('Length of each part:', dataset_sizes)
    
    # Pick the first training batch
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)

    inputs = sample_batched['image']
    inputs = inputs.to(args.device)
        
    labels, labels_uv_coord, labels_xyz_coord = functions_train.create_labels_and_prediction(args, sample_batched=sample_batched)
    print("labels_xyz_coord:", labels_xyz_coord)
    print("labels_xyz_coord.shape:", labels_xyz_coord.shape)
    
    print(submesh.sin_3horizConsec_vert_from_submesh_batch(args, Vertex_coordinates_submesh_batch=labels_xyz_coord))

# CHECK_sin_3horizConsec_vert_from_submesh_batch(predict_uv_or_xyz='xyz')

def CHECK_sin_3verConsec_vert_from_submesh_batch(predict_uv_or_xyz='xyz'):
    """
    Instanciate dataste
    Create dataloaders
    Iterate over dataloaders
    Get one batch of labels
    Compute sine of angles between adjacent edges
    Reshape labels and append to them the distances between adjacent vertices
    """
    transformed_dataset = instanciate_dataset_example(predict_uv_or_xyz=predict_uv_or_xyz, 
                                                      submesh_num_vertices_vertical=5, submesh_num_vertices_horizontal=3) 
    args = functions_data_processing.parser()
    args.submesh_num_vertices_vertical=5
    args.submesh_num_vertices_horizontal=3
    transformed_dataset_parts, dataset_sizes, dataloaders = data_loading.random_split_notMixingSequences( 
            dataset=transformed_dataset, args=args)
    print('Length of transformed_dataset:', len(transformed_dataset))
    print('Length of each part:', dataset_sizes)
    
    # Pick the first training batch
    iterable_dataloaders = iter(dataloaders['train'])
    sample_batched = next(iterable_dataloaders)

    inputs = sample_batched['image']
    inputs = inputs.to(args.device)
        
    labels, labels_uv_coord, labels_xyz_coord = functions_train.create_labels_and_prediction(args, sample_batched=sample_batched)
    print("labels_xyz_coord:", labels_xyz_coord)
    print("labels_xyz_coord.shape:", labels_xyz_coord.shape)
    
    print(submesh.sin_3verConsec_vert_from_submesh_batch(args, Vertex_coordinates_submesh_batch=labels_xyz_coord))

# CHECK_sin_3verConsec_vert_from_submesh_batch(predict_uv_or_xyz='xyz')

