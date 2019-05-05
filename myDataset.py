#encoding=utf-8

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class myMnistDataset(Dataset):
    def __init__(self, root_dir, train_test):
        fp = open(root_dir+"/"+train_test+"_labels.txt")
        labels = []
        img_dir = []
        for line in fp:
            temp = line.strip().split(",")
            labels.append(int(temp[1]))
            img_dir.append(root_dir+"/"+temp[0])
        fp.close()
        self.labels = labels
        self.img_dir = img_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_name = self.img_dir[index]
        image = cv2.imread(img_name)
        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32)
        image = image[:,:,0]
        image = np.expand_dims(image, axis=0)
        image /= 255
        label = self.labels[index]
        return image, label
