import pickle
import numpy as np
import os


class Cifar100:
    def __init__(self):
        with open(r'E:\third\patent\data\65.npy', 'rb', ) as f:
            self.train_data1 = np.load(f, encoding='latin1', allow_pickle=True)
        with open(r'E:\third\patent\data\label.npy', 'rb', ) as f:
            self.train_label1 = np.load(f, encoding='latin1', allow_pickle=True)

        self.train_groups, self.val_groups, self.test_groups = self.initialize()
        self.batch_num = 5 #5

    def initialize(self):
        train_groups = [[], [], [], [], [], []]
        for train_data, train_label in zip(self.train_data1, self.train_label1):

            if 0 <= int(train_label) < 13:
                train_groups[0].append((train_data, train_label))
            elif 13 <= int(train_label) < 26:
                train_groups[1].append((train_data, train_label))
            elif 26 <= int(train_label) < 39:
                train_groups[2].append((train_data, train_label))
            elif 39 <= int(train_label) < 52:
                train_groups[3].append((train_data, train_label))
            elif 52<= int(train_label) < 65:
                train_groups[4].append((train_data, train_label))


        val_groups = [[], [], [], [], [], []]
        test_groups = [[], [], [], [], [], []]
        for i, train_group in enumerate(train_groups):


            test_groups[i] = train_groups[i][500:600] + train_groups[i][1100:1200] + train_groups[i][1700:1800] + \
                             train_groups[i][2300:2400] + train_groups[i][2900:3000] + train_groups[i][3500:3600] + \
                             train_groups[i][4100:4200] + train_groups[i][4700:4800] + train_groups[i][5300:5400] + \
                             train_groups[i][5900:6000] + train_groups[i][6500:6600] + \
                             train_groups[i][7100:7200] + train_groups[i][7700:7800]

            train_groups[i] = train_groups[i][0:500] + train_groups[i][600:1100] + train_groups[i][1200:1700] + \
                              train_groups[i][1800:2300] + train_groups[i][2400:2900] + train_groups[i][3000:3500] + \
                              train_groups[i][3600:4100] + train_groups[i][4200:4700] + train_groups[i][4800:5300] + \
                              train_groups[i][5400:5900] + train_groups[i][6000:6500] + \
                              train_groups[i][6600:7100] + train_groups[i][7200:7700]



        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]


if __name__ == "__main__":
    cifar = Cifar100()
    for i in range(5):
        print(len(cifar.train_groups[i]))
        print(len(cifar.val_groups[i]))
        print(len(cifar.test_groups[i]))
    print(cifar.train_groups[4][500:])
