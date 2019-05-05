#encoding=utf-8

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib

cols = 10
nz = 100
ngf = 64
nc = 1
n_classes = 10
netG_path = "netG_ac.torch"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                    nn.ConvTranspose2d(nz+n_classes, ngf*8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf*8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf*4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf*2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                )
        self.label_emb = nn.Embedding(n_classes, nz)

    def forward(self, labels, noise):
        labels = labels.view(-1, 1)
        onehot = torch.zeros(len(labels), n_classes).scatter_(1, labels, 1)
        x = torch.cat((onehot, noise), 1)
        x = x.view(-1, nz+n_classes, 1, 1)
        output = self.main(x)
        return output

netG = Generator()
netG.load_state_dict(torch.load(netG_path))

noise = np.random.randn(cols ,nz)
noises = np.zeros((0, nz))
for _ in range(n_classes):
    noises = np.concatenate((noises, noise), axis=0)
noises = torch.Tensor(noises)

gen_labels = []
for x in range(n_classes):
    for col in range(cols):
        gen_labels.append(x)
gen_labels = np.array(gen_labels)
gen_labels = torch.LongTensor(gen_labels)

fake_imgs = netG(gen_labels, noises).detach().numpy()
fake_imgs *= 255
fake_imgs = np.clip(fake_imgs, 0, 255).astype(np.uint8)

i = 0
out_img = np.zeros((0, cols*64))
for classes in range(n_classes):
    temp = np.zeros((64, 0))
    for col in range(cols):
        temp = np.concatenate((temp, fake_imgs[i, 0]), axis=1)
        i += 1
    out_img = np.concatenate((out_img, temp), axis=0)
cv2.imwrite("out_img.png", out_img)
