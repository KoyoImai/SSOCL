
import numpy as np


import torch
import torch.utils.data as data
from torchvision import transforms, datasets


class Cifar100(data.Dataset):

    def __init__(self, cfg, augmentation, data_folder=None, train=True):

        data.Dataset.__init__(self)

        self.data_folder = data_folder
        self.train = train

        # データセットの用意
        dataset = datasets.CIFAR100(root=self.data_folder,
                                    download=True,
                                    train=self.train)
        self.dataset = dataset

        # データ拡張の設定
        if not isinstance(augmentation, list):
            augmentation = [augmentation]
        self.augmentation = augmentation

    
    def __getitem__(self, index):

        # データとラベルをself.datasetから取り出す
        image, label = self.dataset[index]

        # データに加えるデータ拡張を設定
        i = 0
        transform = None
        if self.augmentation is not None:
            i = np.random.randint(len(self.augmentation))
            transform = self.augmentation[i]
        
        if transform is not None:
            im = transform(image)
        
        out = {
            "input": im,
            "target": label
        }

        return out
    

    def __len__(self):
        return len(self.dataset)








