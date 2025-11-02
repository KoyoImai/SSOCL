
import torch
import torch.nn as nn
import torch.nn.functional as F



class SupConLoss(nn.Module):
    def __init__(self, stream_bsz, temperature=0.07, base_temperature=0.07):
        
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.stream_bsz = stream_bsz


    def forward(self, z_stu, z_tch, labels=None, mask=None, device=None):

        """
        args:
            z_stu: student features of shape [bsz, D]
            z_tch: teacher features of shape [bsz, D]
            labels: ground truth of shape [N]
            mask: contrastive mask of shape [N, N], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.  
        returns:
            A loss scalar.      
        """

        batch_size = z_stu.shape[0]

        all_features = torch.cat((z_stu, z_tch), dim=0)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        # print(mean_log_prob_pos.shape, mean_log_prob_pos.max().item(), mean_log_prob_pos.mean().item(), mean_log_prob_pos.min().item())

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(2, batch_size)
        stream_mask = torch.zeros_like(loss).float().to(device)
        stream_mask[:, :self.stream_bsz] = 1
        loss = (stream_mask * loss).sum() / stream_mask.sum()
        
        return loss











class IRDLoss(nn.Module):
    def __init__(self, current_temperature=0.2, past_temperature=0.01):

        super(IRDLoss, self).__init__()

        self.curr_temp = current_temperature
        self.past_temp = past_temperature

    def forward(self, cur_features, past_features, device):

        cur_features_sim = torch.div(torch.matmul(cur_features, cur_features.T), self.curr_temp)
        
        logits_mask = torch.scatter(
            torch.ones_like(cur_features_sim),
            1,
            torch.arange(cur_features_sim.size(0)).view(-1, 1).to(device),
            0
        )

        cur_logits_max, _ = torch.max(cur_features_sim * logits_mask, dim=1, keepdim=True)
        cur_features_sim = cur_features_sim - cur_logits_max.detach()
        row_size =cur_features_sim.size(0)
        cur_logits = torch.exp(cur_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            cur_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
    
        past_features_sim = torch.div(torch.matmul(past_features, past_features.T), self.past_temp)
        past_logits_max, _ = torch.max(past_features_sim * logits_mask, dim=1, keepdim=True)
        past_features_sim = past_features_sim - past_logits_max.detach()
        past_logits = torch.exp(past_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            past_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        loss_distill = (- past_logits * torch.log(cur_logits)).sum(1).mean()
        #return loss_distill

        return cur_logits, loss_distill







