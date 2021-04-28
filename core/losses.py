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

class LSEP_Loss(nn.Module):
    def __init__(self):
        super(LSEP_Loss, self).__init__()
    
    def forward(self, inputs, targets):
        """"
        Parameters
        ----------
        inputs : logits
        targets : multi-label binarized vector
        """
        
        batch_size, classes = targets.size()[:2]

        positive_indices = targets.gt(0).float() # value > 0
        negative_indices = targets.eq(0).float() # value == 0

        # print(inputs)
        # print(targets)
        # print(positive_indices)
        # print(negative_indices)

        loss = 0.

        for i in range(batch_size):
            positive_mask = positive_indices[i].nonzero(as_tuple=False)
            negative_mask = negative_indices[i].nonzero(as_tuple=False)

            positive_examples = inputs[i, positive_mask]
            negative_examples = inputs[i, negative_mask]

            positive_examples = torch.transpose(positive_examples, 0, 1)

            exp_sub = torch.exp(negative_examples - positive_examples)
            exp_sum = torch.sum(exp_sub)

            loss += torch.log(1 + exp_sum)
            
            # print(positive_examples, positive_examples.size())
            # print(negative_examples, negative_examples.size())
            # print(exp_sub)
            # print(exp_sum)
            # print(torch.log(1 + exp_sum))
            # input()
        
        return loss / batch_size