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
        self.name           = 'klcov'

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
        #cov_q = torch.diagonal(torch.einsum('ij,kl->ikjl',_X,_X), dim1=0,dim2=1).permute(2,0,1) + torch.eye(n_classes).cuda() * 1e-4

        mu_q = batch.mean(dim=[1], keepdim=True).repeat((1,n_classes)).cuda().float() # mean character should be 0... repeat it 144 times

        inv_T = T.inverse()

        kld_loss = 0.5 * n_characters * torch.mean(
            torch.log(torch.trace(T)) # https://math.stackexchange.com/questions/202248/upper-bound-on-determinant-of-matrix-in-terms-of-trace 
            - torch.log(torch.trace(cov_q)) # trace is an upper bound on the determinant  - otherwise can use torch.linalg.slogdet[-1] directly 
            - n_classes
            + torch.trace(torch.matmul(torch.matmul(mu_q, inv_T), mu_q.T)) / n_characters
            + torch.trace(torch.matmul(inv_T, cov_q)))
        return kld_loss

