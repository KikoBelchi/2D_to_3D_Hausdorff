# Use the same virtual environment used for training the network

###
### Imports
###
import torch
import torch.nn as nn
import argparse




###
### Parser for entering arguments and setting default ones
###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for setting arguments')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                        help='number of workers to train (default: 4)')
    parser.add_argument('--vertices', type=int, default=6, metavar='N',
                        help='Number of vertices to use. It selects the first ones returned by blender. Number between 6 and 5356 with the current dataset')    
    parser.add_argument('--verbose', type=int, default=1, metavar='N',
                        help='verbose==1 will print comments about the loss, etc. To turn them off, set verbose=0 (default: 1)')    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    batch_size = args.batch_size
    num_selected_vertices = args.vertices    
    verbose = args.verbose
                
    ###
    ### Torch device (CPU vs GPU)
    ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nCuda device =', device)

    
    
    
    
###
### What this script is about
###   
if __name__ == "__main__":
    print("There are different behaviours of the MSELoss function from torch.nn for different versions of torch, so I will show some of these differences and create my own MSELoss, more stable to changing versions")
    print("I will use my function and plot step by step to prove that it works for the example I give,")
    print("and show how different the result given by torch.nn is")
    print()




###
### Playing around with MSELoss from torch
###
if __name__ == '__main__':
    if verbose==1:
        # Read the following to make sure to understand how nn.MSELoss works: <br>
        # https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        #  
        # The following few cells on playing around with MSELoss have versions for torch 0.4.0 and 0.4.1.<br>
        # For
        # torch==0.4.0 <br>
        # nn.MSELoss().reduce and nn.MSELoss().size_average work, and nn.MSELoss().reduction does not.
        # 
        # Using <br>
        # torch==0.4.1 <br>
        # reduction works and reduce and size_average have been deprecated.

        # The default behaviour of nn.MSELoss() is taking the average of the losses of the batch.





        # # Version for torch==0.4.0
        # criterion = nn.MSELoss()
        # print(criterion.reduce)
        # print(criterion.size_average)
        # # Since 'criterion.reduce == 1' and 'criterion.size_average == 1', 
        # # the resulting loss below is the average of the losses of the batch, 
        # # instead of 1 loss per element in the batch.

        # v= torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
        # w= torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        # print(v.shape)
        # loss = criterion(v, w)
        # print('loss =', loss.item())
        # print()

        # print('Notice that this seems to be computing the loss of {[v_1^1, v_1^2], [v_2^1, v_2^2]} against {[w_1^1, w_1^2], [w_2^1, w_2^2]} as $1/4 \sum_{1 \leq i, j \leq 2} (v_i^j - w_i^j)^2$.')
        # print('This is different from computing the loss as $1/2 \sum_{1 \leq i \leq 2} d(v_i, w_i)^2$. If d(v_i, w_i) were computed as the Euclidean distance, we would obtain a result of 8. If d(v_i, w_i) were computed as the square of the Euclidean distance, we would obtain a result of 64.')
        # print('In a sense, is like taking the MSE of the MSE, as in doig MSE of d(v_i,w_i), where d(v,w) is defined as the average of the square of the component-wise differences of the vectors v and w.')
        # print()

        # # Another example corroborating this
        # v= torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
        # w= torch.tensor([[2, 2], [4, 4]], dtype=torch.float32)
        # loss = criterion(v, w)
        # print('loss =', loss.item())





        # Version for torch==0.4.1
        criterion = nn.MSELoss()
        print('criterion.reduction:', criterion.reduction)
        # Since 'criterion.reduction == 'elementwise_mean'', 
        # the resulting loss below is the average of the losses of the batch, 
        # instead of 1 loss per element in the batch.

        v= torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
        w= torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        print(v.shape)
        loss = criterion(v, w)
        print('loss =', loss.item())
        print()

        print('Notice that this seems to be computing the loss of {[v_1^1, v_1^2], [v_2^1, v_2^2]} against {[w_1^1, w_1^2], [w_2^1, w_2^2]} as $1/4 \sum_{1 \leq i, j \leq 2} (v_i^j - w_i^j)^2$.')
        print('This is different from computing the loss as $1/2 \sum_{1 \leq i \leq 2} d(v_i, w_i)^2$. If d(v_i, w_i) were computed as the Euclidean distance, we would obtain a result of 8. If d(v_i, w_i) were computed as the square of the Euclidean distance, we would obtain a result of 64.')
        print('In a sense, it is like taking the MSE of the MSE, as in doing MSE of d(v_i,w_i), where d(v,w) is defined as the average of the square of the component-wise differences of the vectors v and w.')
        print()

        # Another example corroborating this
        print('Another example corroborating this:')
        v= torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
        w= torch.tensor([[2, 2], [4, 4]], dtype=torch.float32)
        loss = criterion(v, w)
        print('loss =', loss.item())
        print()

        # More tests on the loss
        v= torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float64)
        w= torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.float64)
        print(v.shape)
        print(v)
        loss = criterion(v, w)
        print('loss =', loss.item())
        print()



        # Version for torch==0.4.0:
        # 
        # To avoid nn.MSELoss() taking the average of the losses of the batch:
        # 
        # criterion_noAverage = nn.MSELoss(size_average=False, reduce=False) <br>
        # returns the Square Error of each component, without taking the mean.
        # 
        # When reduce is False, it ignores size_average, so <br>
        # criterion_noAverage = nn.MSELoss(size_average=True, reduce=False) <br>
        # yields the same result





        # # Version for torch==0.4.0
        # criterion_noAverage = nn.MSELoss(size_average=False, reduce=False)
        # loss = criterion_noAverage(v, w)
        # print('loss =', loss)

        # criterion_noAverage = nn.MSELoss(size_average=True, reduce=False)
        # loss = criterion_noAverage(v, w)
        # print('loss =', loss)





        # Version for torch==0.4.1:
        # 
        # To avoid nn.MSELoss() taking the average of the losses of the batch:
        # 
        # criterion_noAverage = nn.MSELoss(reduction = 'none') <br>
        # returns the Square Error of each component, without taking the mean.





        # Version for torch==0.4.1
        criterion_noAverage = nn.MSELoss(reduction = 'none')
        print('criterion_noAverage.reduction:', criterion_noAverage.reduction)
        loss = criterion_noAverage(v, w)
        print('loss =', loss)
        #
        ###
        ### But what I want is a MSE of each vector in the batch, not of each component of each vector in the batch. 
        ### I would need to do that by hand, since there is no option to do that with nn.MSELoss(), but for the moment, I will stay with the standar computation of nn.MSELoss().
        ###

        
        
        
        
        
###    
### Creating my own MSE loss function
###
def MSE_loss(labels, outputs, batch_size, num_selected_vertices):
    """We are defining the loss of each observation (i.e., a mesh) as the MSE of each vertex,
    I.e., for a given vertex v=(x,y,z) and its prediction v'=(x',y',z'),
    error(v,v')=l_2(v,v')=sqrt((x-x')**2 + (y-y')**2 + (z-z')**2).
    Hence, square_error(v,v')=(l_2(v,v'))**2=(x-x')**2 + (y-y')**2 + (z-z')**2,
    and MSE of a whole mesh (set of vertices) and its prediction becomes the mean of these values.
    Then, to compute the loss on a whole batch, we do MSE of the computed (MSE) loss of each observation.
    Input:
    - labels and outputs are torch tensors with shape [batch_size, num_selected_vertices*3],
    preferably with dtype=torch.float64, device=device.
    Each row in labels represents a batch of the form x_0 y_0 z_0 x_1 y_1 z_1 ..."""
    MSE_of_batch = (1/batch_size)*torch.sum(torch.pow((1/num_selected_vertices)*torch.sum(torch.pow(labels-outputs,2), dim=1),2)).item()
    return MSE_of_batch

###    
### Explaining my own MSE loss function
###
if __name__ == '__main__':
    if verbose==1:
        # Explanation of the function
        print('\nCreating my own loss function:')
        print("Explaining the MSE loss I defined as MSE_of_batch = (1/batch_size)*torch.sum(torch.pow((1/num_selected_vertices)*torch.sum(torch.pow(labels-outputs,2), dim=1),2)).item()")

        print('We are defining the loss of each observation as the MSE of each vertex')
        print("I.e., for a given vertex v=(x,y,z) and its prediction v'=(x',y',z'),")
        print("error(v,v')=l_2(v,v')=sqrt((x-x')**2 + (y-y')**2 + (z-z')**2)")
        print("Hence, square_error(v,v')=(l_2(v,v'))**2=(x-x')**2 + (y-y')**2 + (z-z')**2,")
        print("And MSE of a whole mesh (set of vertices) and its prediction becomes the mean of these values.")
        print("Example with batch_size=" + str(batch_size) + " and num_selected_vertices=" + str(num_selected_vertices) + ":")
        labels = torch.zeros([batch_size, num_selected_vertices*3], dtype=torch.float64, device=device)
    #     print('labels.device:', labels.device)
        outputs = torch.ones([batch_size, num_selected_vertices*3], dtype=torch.float64, device=device)
        outputs[0,:] *= 0
        outputs[1,:] *= 1
        outputs[2,:] *= 2
        outputs[3,:] *= 3
        print('labels.shape:', labels.shape)
        print('Each row in labels represents a batch of the form x_0 y_0 z_0 x_1 y_1 z_1 ...')
        print('labels=\n', labels)
        print('outputs=\n', outputs)
        elt_wise_l2_squared = torch.pow(labels-outputs,2) # each element is of the form (x-x')**2, (y-y')**2 or (z-z')**2
        print('elt_wise_l2_squared(labels, outputs)=\n', elt_wise_l2_squared)
        MSE_of_each_observation_in_batch = (1/num_selected_vertices)*torch.sum(elt_wise_l2_squared, dim=1) 
        print('MSE_of_each_observation_in_batch(labels,outputs):', MSE_of_each_observation_in_batch)
    #     print('12/num_selected_vertices=', 12/num_selected_vertices)
        # Column tensor where MSE_of_each_observation_in_batch[i] = MSE(labels[i,:], outputs[i,:]).
        # Indeed, it is a column tensor where each element is the sum 
        # of the elements in the corresponding row of elt_wise_l2_squared,
        # then divided by the number of selected vertices. 
        # I.e., each element is of the form 
        # 1/num_selected_vertices * 
        # ( (x_0-x_0')**2 + (y_0-y_0')**2 + (z_0-z_0')**2 + (x_1-x_1')**2 + (y_1-y_1')**2 + (z_1-z_1')**2 + ... )

        # Technically, it is not a column tensor, but rather a 1-dim tensor of size=batch_size.
        print('MSE_of_each_observation_in_batch.shape:', MSE_of_each_observation_in_batch.shape)

        print('The loss of a batch will be the MSE of the loss of each observation')
        MSE_of_batch = (1/batch_size)*torch.sum(torch.pow(MSE_of_each_observation_in_batch,2)).item()
    #     print(torch.pow(MSE_of_each_observation_in_batch,2))
    #     print(torch.sum(torch.pow(MSE_of_each_observation_in_batch,2)))
        print("Hence, loss(labels, outputs) is:")
        print('MSE_of_batch:', MSE_of_batch)
        #Equivalently,
        MSE_of_batch = (1/batch_size)*torch.sum(torch.pow((1/num_selected_vertices)*torch.sum(torch.pow(labels-outputs,2), dim=1),2)).item()
        print('MSE_of_batch (computed with 1 line of code):', MSE_of_batch)
        #Equivalently,
        MSE_of_batch = MSE_loss(labels, outputs, batch_size=batch_size, num_selected_vertices=num_selected_vertices)
        print('MSE_of_batch (using my custom defined function):', MSE_of_batch)
        print()

        criterion = nn.MSELoss()
        print('criterion.reduction:', criterion.reduction)
        # Since 'criterion.reduction == 'elementwise_mean'', 
        # the resulting loss below is the average of the losses of the batch, 
        # instead of 1 loss per element in the batch.
        loss = criterion(labels, outputs)
        print("Compare this to the loss using nn.MSELoss():", loss.item())
        print("which is precisely the average of all elements in elt_wise_l2_squared.")

        
        
        
        
        
        
        
