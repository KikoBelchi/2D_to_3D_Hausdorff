"""
Here are some functions to define the Hausdorff loss in
https://github.com/javiribera/weighted-hausdorff-loss/blob/master/object-locator/losses.py.

I can either use them as inspiration to see how to define losses properly with nn.Module or I can simply use their losses.

CAVEAT on their losses:
Batches are not supported, so squeeze your inputs first!
Therefore, if I want to use their loss, I have to proceed as follows:
1. Take a batch and the 1st observation
2. Unsqueeze the observation, so that the shape of the point cloud is (n_points, D), rather than (1, n_points, D)
3. apply AveragedHausdorffLoss as in:
criterion = AveragedHausdorffLoss()
loss = criterion(observation1, observation1_GT)
4. Repeat with all other observations in the batch and average the loss result, since I am considering batch loss as average loss of each observation

DEVICE:
I have checked that all the operations in this script are made in the device of the input tensors.
"""
import torch
import torch.nn as nn

import functions_train

def cdist(x, y):
    '''
    Source: https://github.com/javiribera/weighted-hausdorff-loss/blob/master/object-locator/losses.py.

    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

class AveragedHausdorffLoss(nn.Module):
    """ Source: https://github.com/javiribera/weighted-hausdorff-loss/blob/master/object-locator/losses.py. 
    I modified it to allow for proper Hausdorff computations, rather than only average Hausdorff,
    and to allow for the computation of directional Hausdorff as well, as in returning separately each of the two summands if needed."""
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2, args):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        
        Extra input: 
        args.hausdorff==0 computes average hausdorff.
        args.hausdorff==1 computes proper hausdorff.
        
        If the loss_weight of one direction of this Hausdorff distance is 0, 
        the distance returned is the summand of the direction with non-zero weight.
        
        If args.n_outputs==1, the sum of the two directional Hausdorff distances is returned.
        If args.n_outputs==2, the two directional Hausdorff distances are returned separately.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)
        
        if args.contour==0:
            w_set1_set2 = args.loss_w_chamfer_GT_pred
            w_set2_set1 = args.loss_w_chamfer_pred_GT
        else:
            w_set1_set2 = args.loss_w_chamfer_GTcontour_pred
            w_set2_set1 = args.loss_w_chamfer_pred_GTcontour
        
        # Modified Chamfer Loss
        if args.hausdorff==0: # Average Hausdorff distance   
            if w_set1_set2==0:
                d_from_each_pt_in_set1_to_set2 = torch.zeros([], device=d2_matrix.device, dtype=d2_matrix.dtype)
            else: 
                d_from_each_pt_in_set1_to_set2 = torch.mean(torch.min(d2_matrix, 1)[0])                
            
            if w_set2_set1==0:
                d_from_each_pt_in_set2_to_set1 = torch.zeros([], device=d2_matrix.device, dtype=d2_matrix.dtype)
            else:
                d_from_each_pt_in_set2_to_set1 = torch.mean(torch.min(d2_matrix, 0)[0])
        else: # Classical Hausdorff distance       
            if w_set1_set2==0:
                d_from_each_pt_in_set1_to_set2 = torch.zeros([], device=d2_matrix.device, dtype=d2_matrix.dtype)
            else: 
                d_from_each_pt_in_set1_to_set2 = torch.max(torch.min(d2_matrix, 1)[0])[0]
            
            if w_set2_set1==0:
                d_from_each_pt_in_set2_to_set1 = torch.zeros([], device=d2_matrix.device, dtype=d2_matrix.dtype)
            else:
                d_from_each_pt_in_set2_to_set1 = torch.max(torch.min(d2_matrix, 0)[0])[0]

        if args.n_outputs==1:
            return d_from_each_pt_in_set1_to_set2 + d_from_each_pt_in_set2_to_set1
        else:
            return d_from_each_pt_in_set1_to_set2, d_from_each_pt_in_set2_to_set1

def loss_Hausdorff_4_batches(criterion, outputs_uv_coord, sample_batched, args):
    """
    CAVEAT on their losses in https://github.com/javiribera/weighted-hausdorff-loss/blob/master/object-locator/losses.py:
    Batches are not supported, so squeeze your inputs first!
    In this function, we do this adaptation for batches as follows:
    1. Take a batch and the 1st observation
    2. Unsqueeze the observation, so that the shape of the point cloud is (n_points, D), rather than (1, n_points, D)
    3. apply AveragedHausdorffLoss as in:
    criterion = AveragedHausdorffLoss()
    loss = criterion(observation1, observation1_GT)
    4. Repeat with all other observations in the batch and average the loss result, 
    since I am considering batch loss as average loss of each observation
    
    Input:
    - outputs_uv_coord: tensor of shape (batch_size, num_selected_vertices, 2) 
        containing the predicted uv cocordinates of all the vertices in the batch.
    - labels_uv_coord: tensor of shape (batch_size, num_selected_vertices, 2) 
        containing the GT uv cocordinates of all the vertices in the batch.
        
    If args.n_outputs==1, the loss using the sum of the two directional Hausdorff distances is returned.
    If args.n_outputs==2, the two directional Hausdorff distance losses are returned separately.
    """
    
    loss, loss_GT_pred, loss_pred_GT = 0, 0, 0
    
    # Note on torch.squeeze:
    # The returned tensor shares the storage with the input tensor, 
    # so changing the contents of one will change the contents of the other.
    for i in range(args.batch_size):
        if args.towelPixelSample==0:
            image_path=sample_batched['img_name'][i]
            pt_cld_GT = functions_train.sample_of_towel_pixels(image_path, args) # shape (M, 2)
        else:
            if args.contour==0:
                pt_cld_GT = sample_batched['towel_pixel_subsample'][i]
            else: 
                pt_cld_GT = sample_batched['towel_pixel_contour_subsample'][i]

        if args.verbose==1:
            print("Shape of squeezed predicted uv:", torch.squeeze(outputs_uv_coord[i, :, :]).shape)
            print("Shape of subsampled uv:", pt_cld_GT.shape)
            
        loss_batch = criterion.forward(pt_cld_GT, torch.squeeze(outputs_uv_coord[i, :, :]), args)
        # I had to change criterion by criterion.forward to avoid the error message
        # AttributeError: 'AveragedHausdorffLoss' object has no attribute '_forward_pre_hooks'
        
        if args.n_outputs==1:
            loss += loss_batch
        else:
            loss_GT_pred += loss_batch[0]
            loss_pred_GT += loss_batch[1]
    
    if args.n_outputs==1:
        return loss/args.batch_size
    else:
        return loss_GT_pred/args.batch_size, loss_pred_GT/args.batch_size
    
    
    
        

    
    