#encoding=utf-8

import os
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from myDataset import myMnistDataset
import numpy as np

batch_size = 64
root_dir = "/home/pan/workspace/mnist_gan"
netG_path = "netG_ac.torch"
netD_path = "netD_ac.torch"

train_dataset = myMnistDataset(root_dir, "train")
test_dataset = myMnistDataset(root_dir, "test")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

nz = 100
ngf = 64
ndf = 64
nc = 1
n_classes = 10
lr = 0.0002
beta1 = 0.5
epochs = 25

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*8),
                    nn.LeakyReLU(0.2, inplace=True),
                )
        self.adv_layer = nn.Sequential(nn.Linear(512*4*4, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(512*4*4, n_classes), nn.Softmax())

    def forward(self, x):
        output = self.main(x)
        output = output.view(output.shape[0], -1)
        validity = self.adv_layer(output)
        label = self.aux_layer(output)
        return validity, label

netG = Generator()
#netG.apply(weights_init)
netG.load_state_dict(torch.load(netG_path))
print(netG)

netD = Discriminator()
#netD.apply(weights_init)
netD.load_state_dict(torch.load(netD_path))
print(netD)

adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

real_label = 1
fake_label = 0

optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(epochs):
    for current_batch, (real_imgs, classes) in enumerate(train_loader):
        real_imgs = Variable(real_imgs)
        classes = Variable(classes)
        
        netG.zero_grad()
        noise = torch.randn(len(real_imgs), nz)
        gen_labels = torch.LongTensor(len(real_imgs)).random_() % n_classes
        label = torch.full((len(real_imgs),), real_label)
        fake_imgs = netG(gen_labels, noise)
        fake_adv_label, fake_aux_label = netD(fake_imgs)
        lossG = 0.5 * (adversarial_loss(fake_adv_label, label) + auxiliary_loss(fake_aux_label, gen_labels))
        lossG.backward()
        optG.step()

        netD.zero_grad()
        label.fill_(real_label)
        real_adv_label, real_aux_label = netD(real_imgs)
        lossD_real = 0.5 * (adversarial_loss(real_adv_label, label) + auxiliary_loss(real_aux_label, classes))
        lossD_real.backward()

        label.fill_(fake_label)
        fake_adv_label, fake_aux_label = netD(fake_imgs.detach())
        lossD_fake = 0.5 * (adversarial_loss(fake_adv_label, label) + auxiliary_loss(fake_aux_label, gen_labels))
        lossD_fake.backward()
        optD.step()

        pred = np.concatenate([real_aux_label.data.numpy(), fake_aux_label.data.numpy()], axis=0)
        gt = np.concatenate([classes.data.numpy(), gen_labels.data.numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        d_real_acc = np.mean(np.argmax(real_aux_label.data.numpy(), axis=1) == classes.data.numpy())
        d_fake_acc = np.mean(np.argmax(fake_aux_label.data.numpy(), axis=1) == gen_labels.data.numpy())
        lossD = (lossD_real + lossD_fake) / 2

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%, acc_real: %d%%, acc_fake: %d%%] [G loss: %f]"
            % (epoch, epochs, current_batch, len(train_loader), lossD.item(), 100*d_acc, 100*d_real_acc, 100*d_fake_acc, lossG.item()))

        if current_batch%100 == 0:
            folder_name = "ac_epoch{}_batch{}".format(epoch, current_batch)
            os.system("mkdir result_img_torch/{}".format(folder_name))
            for j in range(len(fake_imgs)):
                out_img = fake_imgs[j,0].detach().numpy()
                out_img *= 255
                out_img = out_img.astype(np.uint8)
                cv2.imwrite("result_img_torch/{}/{}_{}.png".format(folder_name, j, gen_labels[j]), out_img)
            for j in range(len(real_imgs)):
                out_img = real_imgs[j,0].detach().numpy()
                out_img *= 255
                out_img = out_img.astype(np.uint8)
                cv2.imwrite("result_img_torch/{}/{}_{}_real.png".format(folder_name, j, np.argmax(real_aux_label.data.numpy()[j])), out_img)
            if current_batch != 0:
                torch.save(netG.state_dict(), "netG_ac.torch")
                torch.save(netD.state_dict(), "netD_ac.torch")
                print("model saved")
