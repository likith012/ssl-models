import torch
import torch.nn.functional as F

class loss_fn(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device

    def forward(self, p1, p2, z1, z2):

        # L2 normalize
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # mutual prediction
        l_pos1 = torch.einsum('nc,nc->n', [p1, z2.detach()]).unsqueeze(-1) # Using stop gradients, z2 doesn't involve in the computation, only p1 is responsible for learning
        l_pos2 = torch.einsum('nc,nc->n', [p2, z1.detach()]).unsqueeze(-1)

        loss = - (l_pos1.mean() + l_pos2.mean()) / 2 # using mean
                
        return loss