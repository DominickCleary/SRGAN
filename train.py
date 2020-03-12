import argparse
import os
import shutil
import time
from math import log10

import pandas as pd
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, LoadAVADataset
from loss import GeneratorLoss
from model import *

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int,
                    help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100,
                    type=int, help='train epoch number')
parser.add_argument('--load_checkpoint', default="",
                    type=str, help='load saved checkpoint')
parser.add_argument('--checkpoint_prefix', default="default",
                    type=str, help='choose checkpoint prefix')
parser.add_argument('--ava_location', default="/media/docleary/Storage/Documents/Datasets/AVA_dataset",
                    type=str, help='directory of the AVA dataset')
parser.add_argument('--batch_size', default=32,
                    type=int, help='batch size')


def save_ckp(state, checkpoint_name):
    f_path = "checkpoints/" + checkpoint_name
    torch.save(state, f_path)


def load_ckp(checkpoint_path, netG, netD, optimizerG, optimizerD):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['gen_state_dict'])
    netD.load_state_dict(checkpoint['dis_state_dict'])
    optimizerG.load_state_dict(checkpoint['gen_optimizer'])
    optimizerD.load_state_dict(checkpoint['dis_optimizer'])
    return netG, netD, optimizerG, optimizerD, checkpoint['epoch']


def train_sr():
    train_set = TrainDatasetFromFolder(
        'data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(
        'data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel()
                                         for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel()
                                             for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    start_epoch = 1

    if(LOAD_CHECKPOINT != ""):
        ckp_path = "checkpoints/" + LOAD_CHECKPOINT
        netG, netD, optimizerG, optimizerD, start_epoch = load_ckp(
            ckp_path, netG, netD, optimizerG, optimizerD)
        print("loading checkpoint '" + LOAD_CHECKPOINT +
              "'\ncontinuing from epoch " + str(start_epoch))

    results = {'d_loss': [], 'g_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            # Generate fake image
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] /
                running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0,
                              'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * \
                    log10(1 / (valing_results['mse'] /
                               valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / \
                    valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(
                    image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' %
                   (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' %
                   (UPSCALE_FACTOR, epoch))

        checkpoint = {
            'epoch': epoch + 1,
            'gen_state_dict': netG.state_dict(),
            'dis_state_dict': netD.state_dict(),
            'gen_optimizer': optimizerG.state_dict(),
            'dis_optimizer': optimizerD.state_dict()
        }
        save_ckp(checkpoint, CHECKPOINT_PREFIX + "_checkpoint_" +
                 str(UPSCALE_FACTOR) + "_" + str(epoch) + ".pth")
        # save loss\scores\psnr\ssim
        results['d_loss'].append(
            running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(
            running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(
            running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(
            running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        # Every 10th epoch save statistics
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) +
                              '_train_results.csv', index_label='Epoch')


def train_nima():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    train_set = LoadAVADataset(AVA_LOCATION + "/train_ava.csv",
                               AVA_LOCATION + "/images", transform=train_transform)
    val_set = LoadAVADataset(AVA_LOCATION + "/val_ava.csv",
                             AVA_LOCATION + "/images", transform=val_transform)

    # print(len(val_set))
    # time.sleep(10)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4)

    model = NIMA()

    model = model.to(DEVICE)

    conv_base_lr = 3e-7
    dense_lr = 3e-6
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
    )

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    count = 0
    init_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(1, NUM_EPOCHS):
        # lrs.send('epoch', epoch)
        batch_losses = []
        for i, data in enumerate(train_loader):
            # print("HIO")
            images = data['image'].to(DEVICE)
            labels = data['annotations'].to(DEVICE).float()
            outputs = model(images)
            outputs = outputs.view(-1, 10, 1)

            optimizer.zero_grad()

            loss = emd_loss(labels, outputs)
            batch_losses.append(loss.item())

            loss.backward()

            optimizer.step()

            # lrs.send('train_emd_loss', loss.item())

            print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1,
                                                                            NUM_EPOCHS, i + 1, len(train_set) // BATCH_SIZE + 1, loss.data[0]))

        avg_loss = sum(batch_losses) / (len(train_set) //
                                        BATCH_SIZE + 1)
        train_losses.append(avg_loss)
        print('Epoch %d averaged training EMD loss: %.4f' %
              (epoch + 1, avg_loss))

        # exponetial learning rate decay
        if (epoch + 1) % 10 == 0:
            conv_base_lr = conv_base_lr * 0.95 ** ((epoch + 1) / 10)
            dense_lr = dense_lr * 0.95 ** ((epoch + 1) / 10)
            optimizer = optim.SGD([
                {'params': model.features.parameters(), 'lr': conv_base_lr},
                {'params': model.classifier.parameters(), 'lr': dense_lr}],
                momentum=0.9
            )

            # send decay hyperparams
            # lrs.send({
            #    'lr_decay_rate': config.lr_decay_rate,
            #    'lr_decay_freq': config.lr_decay_freq,
            #    'conv_base_lr': config.conv_base_lr,
            #    'dense_lr': config.dense_lr
            #    })

        # do validation after each epoch
        batch_val_losses = []
        for data in val_loader:
            images = data['image'].to(DEVICE)
            labels = data['annotations'].to(DEVICE).float()
            with torch.no_grad():
                outputs = model(images)
            outputs = outputs.view(-1, 10, 1)
            val_loss = emd_loss(labels, outputs)
            batch_val_losses.append(val_loss.item())
        avg_val_loss = sum(batch_val_losses) / (len(val_set) // 1 + 1)
        val_losses.append(avg_val_loss)

        # lrs.send('val_emd_loss', avg_val_loss)

        print('Epoch %d completed. Averaged EMD loss on val set: %.4f.' %
              (epoch + 1, avg_val_loss))

        # Use early stopping to monitor training
        if avg_val_loss < init_val_loss:
            init_val_loss = avg_val_loss
            # save model weights if val loss decreases
            print('Saving model...')
            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            torch.save(model.state_dict(), os.path.join(
                config.ckpt_path, 'epoch-%d.pkl' % (epoch + 1)))
            print('Done.\n')
            # reset count
            count = 0
        elif avg_val_loss >= init_val_loss:
            count += 1
            if count == PATIENCE:
                print(
                    'Val EMD loss has not decreased in %d epochs. Training terminated.' % PATIENCE)
                break

    print('Training completed.')


if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    LOAD_CHECKPOINT = opt.load_checkpoint
    CHECKPOINT_PREFIX = opt.checkpoint_prefix
    AVA_LOCATION = opt.ava_location
    BATCH_SIZE = opt.batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 5

    # train_sr()
    train_nima()
