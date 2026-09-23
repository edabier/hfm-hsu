import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import scipy.io as io

from src.utils import utils
import src.models.foundation_models as hfm

def standardise(Y):
    """
    For FMs, the HSI must be standardised before being forwarded
    """
    mean = Y.mean()
    std = Y.std()

    return (Y-mean)/(std + 1e-8)

def order_endmembers(tensor_gt, tensor_hat, tensor2_hat=None):
    """
    Uses scipy linear_sum_assignement algorithm to reorder tensor_hat columns to match tensor_gt
    Tensors must be of shape (batch, D, X) or (D, X) where D is the axis along which to reorder
    tensor_2_hat is another tensor that can be reordered based on the tensor_hat reordering (for abundances)
    """
    is_batched = True
    if tensor_hat.dim() == 2:
        is_batched = False
        tensor_hat = tensor_hat.unsqueeze(0)
    if tensor_gt.dim() == 2:
        is_batched = False
        tensor_gt = tensor_gt.unsqueeze(0)

    tensor_hat_ordered = torch.zeros_like(tensor_hat)
        
    if tensor2_hat != None:
        if tensor2_hat.dim() < 4:
            tensor2_hat = tensor2_hat.unsqueeze(0)

        tensor2_hat_ordered = torch.zeros_like(tensor2_hat)
    
    for b in range(tensor_hat.size()[0]):

        # Normalise the tensors
        tensor_gt_norm = F.normalize(tensor_gt[b], p=2.0, dim=1)  # Normalise along reordered axis
        tensor_hat_norm = F.normalize(tensor_hat[b], p=2.0, dim=1)

        # Compute cost matrix (cosine distance)
        cost_matrix = torch.acos(torch.clamp(tensor_gt_norm @ tensor_hat_norm.T, -1.0, 1.0))
        cost_matrix_np = cost_matrix.cpu().numpy() 

        # Solve assignment problem
        _, col_ind = linear_sum_assignment(cost_matrix_np)

        # Reorder E_hat to match E_gt
        tensor_hat_ordered[b] = tensor_hat[b, col_ind]

        if tensor2_hat != None:
            tensor2_hat_ordered[b] = tensor2_hat[b, col_ind]

    if tensor2_hat != None:
        if is_batched:
            return tensor_hat_ordered, tensor2_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], tensor2_hat_ordered[0], col_ind
    else:
        if is_batched:
            return tensor_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], col_ind
   
def oneD_to_2d(Y, H=None, W=None):
    """
    Reshapes 1D tensors to 2D image
    If H and W are not passed, assumes the image is square
    
    Args:
        Y: input tensor to be reshaped
    """
    is_batched = Y.dim()==3
    if is_batched:
        batch, B, N = Y.shape
        if H != None:
            return Y.reshape(batch, B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(batch, B, H, H)
    else:
        B, N = Y.shape
        if H != None:
            return Y.reshape(B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(B, H, H)

def normalise(X, is_endmember=False):
    """
    Normalises batched tensors

    If is_endmember -> set max to 1 for every endmember
    If is Y or A (must be of shape (batch, B/c, H, W)) -> divides by frobenius norm
    """

    if is_endmember:
        X_norm = X/torch.max(X, dim=0).values

    else:
        X_norm = X/torch.norm(X)

    return X_norm

class HSI_dataset(Dataset):    
    def __init__(self, dataset, path="/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/"):
        
        self.dataset_name = dataset
        dtype = torch.float
        data_path = path + dataset + ".mat"
        data = io.loadmat(data_path)
        
        self.Y = torch.tensor(data['Y'], dtype=dtype)
        self.A = torch.tensor(data['A'], dtype=dtype)
        self.Y, self.A = hfm.reshape_Y("DOFA", utils.oneD_to_2d(self.Y).unsqueeze(0), 224, utils.oneD_to_2d(self.A).unsqueeze(0))
        self.Y, self.A = self.Y[0], self.A[0]
        self.E = torch.tensor(data["E"], dtype=dtype)
        
        self.total_patches = 1

    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        return self.Y, self.E, self.A
   
def create_dataloader(dataset_name, batch_size=1, path="", device="cpu", split=1.0):
    """
    Creates dataloader(s) for a given dataset
    
    Args:
        dataset_name (str): the name of the dataset to use
        dev: the device on which to pass the loaders
        train_split (float, optional): how much of the dataset to use for the training and testing sets
        patch_size (int): whether or not to patch the input HSI
    """
    path += "data/"
    dataset = HSI_dataset(dataset_name, path=path)
    generator = torch.Generator(device=device)
    train_set, test_set = random_split(dataset, lengths=[split, 1-split], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, generator=generator)
    return train_loader, test_loader
  