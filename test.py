import sys
import os
import requests
from functools import partial
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Resize

import dataset
import models_mae
import cv2

# define the utils
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    # plt.show()
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [12, 12]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()



if __name__ == '__main__':
    # load an image
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    img = Image.open(requests.get(img_url, stream=True).raw)
    # img = Image.open('test.jpg')

    # please add the path of your own image in 'path'
    # img = cv2.imread('test.jpg')
    # img = 255 - img
    # transformed_img = cv2.resize(img, (640, 640))
    resize_operator = Resize(size=(640, 640))
    transformed_img = resize_operator(img)
    transforms = dataset.Compose()
    transformed_img = transformed_img - imagenet_mean
    transformed_img = transformed_img / imagenet_std
    # processed_im = transforms(img)
    # normalize by ImageNet mean and std

    image = transformed_img.astype('float32')
    image = torch.tensor(np.array(image))
    # print('4.image shape after change to tensor:', image.shape)
    # plt.imshow(transformed_img)
    # plt.title("Original image", fontsize=16)
    # plt.axis('off')


    image = image.unsqueeze(0)  # CHW
    B, H, W, C = image.shape
    # x = image.expand(3, H, W)
    # x = image.permute(1, 2, 0)
    image = image.squeeze(0)

    # print(img.shape)
    # img = cv2.resize(img, (224, 224))
    # img = img / 255.
    # assert img.shape == (224, 224, 3)

    plt.rcParams['figure.figsize'] = [5, 5]
    # show_image(torch.tensor(img))

    #Load a pre-trained MAE model
    chkpt_dir = 'mae_visualize_vit_large.pth'
    # chkpt_dir = './output_dir/checkpoint-1.pth'
    # model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    model_mae = models_mae.MaskedAutoencoderViT(
        patch_size=16, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(image, model_mae)