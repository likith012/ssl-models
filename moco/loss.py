import torch
import torch.nn.functional as F

class loss_fn(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=1.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device

    def forward(self, anchor, positive, queue):
        
        # L2 normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        queue = F.normalize(queue, p=2, dim=1)

        # positive logits: Nx1, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [anchor, positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [anchor, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # loss
        loss = F.cross_entropy(logits, labels)
        
        return loss # mean
