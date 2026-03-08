import torch
import numpy as np
from trainer import Trainer
import argparse
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--epoch', default =1, type = int)
parser.add_argument('--lr', default = 0.01, type = int)
parser.add_argument('--max_size', default =2000, type = int)
parser.add_argument('--total_cls', default =65, type = int)  #65
args = parser.parse_args()


if __name__ == "__main__":
    t1 = time.time()
    params = [(0.5, 0.5)]
    #params = [(0.3, 0.5), (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.5)]
    for a, b in params:
        trainer = Trainer(args.total_cls)
        trainer.train(args.batch_size, args.epoch, args.lr, args.max_size, a, b)
    t2 = time.time()
    print("训练时间为：", t2 - t1)
    allocated_memory = torch.cuda.memory_allocated()
    print(allocated_memory)
    print(allocated_memory / (1024 * 1024))
