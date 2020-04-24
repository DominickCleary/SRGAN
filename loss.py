import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from torch import nn
from torchvision.models.vgg import vgg16
from model import InferAesthetic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.aesthetic_loss = AestheticLoss()

    def forward(self, out_labels, out_images, target_images):
        # Aesthetic Loss

        # stardt = time.time()
        aesthetic_loss = self.aesthetic_loss(out_images, target_images)
        # end = time.time()
        # print(end - start)

        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(
            out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)

        result = image_loss + 0.001 * adversarial_loss + 0.006 * \
            perception_loss + 2e-8 * tv_loss + 0.05 * aesthetic_loss
        return result


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class AestheticLoss(nn.Module):
    def __init__(self):
        super(AestheticLoss, self).__init__()
        self.aesthetic_loss = InferAesthetic()

    def forward(self, out, target):

        fake_predict = torch.Tensor(out.size()[0]).to(DEVICE)
        target_predict = torch.Tensor(target.size()[0]).to(DEVICE)

        for x in range(0, len(out)):
            fake_predict[x] = self.aesthetic_loss(out[x])
            target_predict[x] = self.aesthetic_loss(target[x])
        fake_mean = torch.mean(fake_predict)
        target_mean = torch.mean(target_predict)
        # Get the difference between average values of the the target and fake images
        result = torch.abs(target_mean - fake_mean)
        return result


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
