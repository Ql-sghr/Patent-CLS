import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # Fully connected layer
        self.fc = WALinear(300, 65)  # Note: Adjust input dimension based on attention output   #65
        # self.fc = nn.Linear(300, 65)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length, embedding_dim)

        # Apply convolution and activation functions
        conv_outputs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Max pooling
        pooled_outputs = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conv_outputs]

        # Concatenate all pooled features
        pooled_outputs = torch.cat(pooled_outputs, 1)  # (batch_size, num_filters * len(filter_sizes))

        # Dropout
        b = self.dropout(pooled_outputs)
        logits = self.fc(b)
        return logits

    def weight_align(self, step_b):
        self.fc.align_norms(step_b)


class WALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_num_classes = self.out_features // 5
        self.WA_linears = nn.ModuleList()
        self.WA_linears.extend([nn.Linear(self.in_features, self.sub_num_classes, bias=False) for i in range(5)])  # 5

    def forward(self, x):
        out1 = self.WA_linears[0](x)
        out2 = self.WA_linears[1](x)
        out3 = self.WA_linears[2](x)
        out4 = self.WA_linears[3](x)
        out5 = self.WA_linears[4](x)

        return torch.cat([out1, out2, out3, out4, out5], dim=1)

    def align_norms(self, step_b):
        # Fetch old and new layers
        new_layer = self.WA_linears[step_b]
        old_layers = self.WA_linears[:step_b]

        # Get weight of layers
        new_weight = new_layer.weight.cpu().detach().numpy()
        # for i in range(step_b):
        #    old_weight = np.concatenate([old_layers[i].weight.cpu().detach().numpy() for i in range(step_b)])
        weights_list = []
        for i in range(step_b):
            # 从每个层中获取权重，转换为CPU上的NumPy数组，并添加到列表中
            weight = old_layers[i].weight.cpu().detach().numpy()
            weights_list.append(weight)
        old_weight = np.concatenate(weights_list, axis=0)
        print("old_weight's shape is: ", old_weight.shape)
        print("new_weight's shape is: ", new_weight.shape)

        # Calculate the norm
        Norm_of_new = np.linalg.norm(new_weight, axis=1)
        Norm_of_old = np.linalg.norm(old_weight, axis=1)
        assert (len(Norm_of_new) == 13)
        assert (len(Norm_of_old) == step_b * 13)

        # Calculate the Gamma
        # gamma = np.mean(Norm_of_new) / np.mean(Norm_of_old)
        gamma = np.mean(Norm_of_old) / np.mean(Norm_of_new)
        print("Gamma = ", gamma)

        # Update new layer's weight
        self.WA_linears[step_b].weight.data = gamma * self.WA_linears[step_b].weight.data



