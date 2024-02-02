import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.name           = 'mset'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM



    def forward(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

        full_T = kwargs['T']
        T = torch.tensor(full_T.loc[labels, labels].values).cuda().float()
        

        n_classes, n_characters = batch.shape

        _X = batch - batch.mean(dim=0, keepdim=True)
        cov_q = (
            torch.matmul(_X, _X.T)# + torch.eye(n_classes).cuda() * 1e-4
        ) / n_characters

        return torch.nn.functional.mse_loss(cov_q, T)

