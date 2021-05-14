import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_Loss(nn.Module):
    # gamma and alpha from https://arxiv.org/pdf/1708.02002.pdf
    def __init__(self, gamma=2., alpha=1., average=True):
        super(Focal_Loss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        
        self.average = average
        
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        # if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
    
    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        
        pt = labels * preds + (1 - labels) * (1 - preds)
        loss = -(1 - pt)**self.gamma * torch.log(pt + 1e-10) * self.alpha
        
        loss = torch.sum(loss, dim=1)
        
        if self.average: 
            return loss.mean().float()
        else: 
            return loss.sum().float()

def L1_Loss(A_tensors, B_tensors):
    return torch.abs(A_tensors - B_tensors)

def L2_Loss(A_tensors, B_tensors):
    return torch.pow(A_tensors - B_tensors, 2)

# ratio = 0.2, top=20%
def Online_Hard_Example_Mining(values, ratio=0.2):
    b, c, h, w = values.size()
    return torch.topk(values.reshape(b, -1), k=int(c * h * w * ratio), dim=-1)[0]

def shannon_entropy_loss(logits, activation=torch.sigmoid, epsilon=1e-5):
    v = activation(logits)
    return -torch.sum(v * torch.log(v+epsilon), dim=1).mean()
