
import torch
import torch.nn as nn
import torch.nn.functional as F






class MultiCropDistillationLoss(nn.Module):

    def __init__(self, temp):

        super(MultiCropDistillationLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.temp = temp

    
    def forward(self, features, ema_features):
        
        num_patch = len(features)
        batch_size = features[0].shape[0]
        
        # 特徴量の平均 EMA model の出力特徴から計算
        z_list = torch.stack(list(ema_features), dim=0)
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)                    # torch.Size([100, 1024])
        
        # 特徴量の平均をパッチ数分だけ連結 （ここをなくしたい）
        #z_avg = torch.cat([z_avg]*num_patch, dim=0)
        
        
        # 特徴量の連結
        features = torch.cat(features, dim=0)
        #print("features.shape : ", features.shape)              # torch.Size([2000, 1024])
        
        
        # ラベルの作成
        labels = torch.cat([torch.arange(len(z_avg))], dim=0)
        #print("labels.shape : ", labels.shape)                  # torch.Size([100])
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #print("labels.shape : ", labels.shape)                  # torch.Size([100, 100])
        
        labels = torch.cat([labels]*num_patch, dim=0)
        labels = labels.cuda()
        #print("labels.shape : ", labels.shape)                  # torch.Size([2000, 100])
        
        """
        print("labels[0] : ", labels[0])
        print("labels[98] : ", labels[98])
        print("labels[99] : ", labels[99])
        print("labels[100] : ", labels[100])
        print("labels[101] : ", labels[101])
        print("labels[102] : ", labels[102])
        """
        
        # 特徴量の正規化
        features = F.normalize(features, dim=1)
        z_avg =F.normalize(z_avg, dim=1)
        
        
        # 類似度行列の計算
        similarity_matrix = torch.matmul(features, z_avg.T)
        #print("similarity_matrix.shape : ", similarity_matrix.shape)       # torch.Size([2000, 100])
        
        
        # 正例の計算
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        #print("positives.shape : ", positives.shape)         # torch.Size([2000, 1])
        
        
        # 負例の計算
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        #print("negatives.shape : ", negatives.shape)         # torch.Size([2000, 99])
        
        
        # logitsの計算
        logits = torch.cat([positives, negatives], dim=1)
        #print("logits.shape : ", logits.shape)              # torch.Size([2000, 100])
        
        # labelsの再作成
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        logits = logits / self.temp
        
        loss = self.criterion(logits, labels)
        return loss


