import time
import torch
import numpy as numpy
import matplotlib.pyplot as plt
import torchvision

from torch import nn
from torchvision.models.vgg import vgg16
from model import InferAesthetic

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
        # torchvision.utils.save_image(target_images, "target.png")
        # torchvision.utils.save_image(out_images, "out.png")

        aesthetic_loss = self.aesthetic_loss(out_images, target_images)
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(
            out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        # print(len(x))
        # time.sleep(10)
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
    
    def forward(self, x, y):
        print("FAKE" + str(self.aesthetic_loss.predict(x[0])))
        print("REAL" + str(self.aesthetic_loss.predict(y[0])))
        # torchvision.utils.save_image(x[0], "fake.png")
        # torchvision.utils.save_image(y[0], "real.png")
        # time.sleep(5)
            

if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
