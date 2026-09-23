import torch.nn as nn
import torch
import torch.nn.functional as F

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_pred.dim() == 1:
            y_true = F.normalize(y_true, dim=0, p=2)
            y_pred = F.normalize(y_pred, dim=0, p=2)
        else:
            y_true = F.normalize(y_true, dim=1, p=2)
            y_pred = F.normalize(y_pred, dim=1, p=2)

        A = torch.mul(y_true, y_pred)

        if y_true.dim() == 1:
            A = torch.sum(A, dim=0)
        else:
            A = torch.sum(A, dim=1)
            
        sad = torch.acos(A)
        loss = torch.mean(sad)
        return loss
