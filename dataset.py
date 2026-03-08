from torch.utils.data import Dataset
import numpy as np
import torch

class BatchData(Dataset):
    def __init__(self, images, labels, input_transform=None):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if isinstance(label, int):
            label=str(label)
            label=np.array([label])
        label = label.astype(float)
        #label = torch.from_numpy(label)
        #label=label.to(torch.float)
        label = torch.tensor(label)
        label = label.long()
        image = image.astype(float)
        return torch.Tensor(image), label

    def __len__(self):
        return len(self.images)
