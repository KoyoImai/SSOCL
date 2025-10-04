
import torch
import torch.nn as nn






class MultiCropContrastiveLoss(nn.Module):

    def __init__(self, temp):

        super(MultiCropContrastiveLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.temp = temp

    
    def forward(self, features):


        assert False




class TotalCodingRateLoss(nn.Module):

    def __init__(self, eps=0.01, alpha=0.01):

        super(TotalCodingRateLoss, self).__init__()

        self.eps = eps
        self.alpha = alpha
    

    def compute_discrimn_loss(self, W):
        
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.


    def forward(self, X):

        assert False
        return - self.compute_discrimn_loss(X.T)




