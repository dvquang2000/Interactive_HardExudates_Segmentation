import torch
import torch.nn as nn
from .automatic_segmentation_model import UNetWithResnet50Encoder, VGGUnet, vgg16bn_unet
from PIL import Image
import os
import numpy as np
from torchvision import datasets, models, transforms
import cv2
def pil_loader(image_path, size):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.resize((size, size))
        return img.convert('RGB')
def Normalize(img,color_mean=[0.485, 0.456, 0.406], color_std=[0.229, 0.224, 0.225]):
    img1 = transforms.functional.to_tensor(img)
    out = transforms.functional.normalize(img1, color_mean, color_std)
    return out

def VGGUnet_predict(img_path, classic_seg_model):
    image_size=512
    softmax = nn.Softmax(dim=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = vgg16bn_unet(output_dim=2, pretrained=True)
    if os.path.isfile(classic_seg_model):
        checkpoint = torch.load(classic_seg_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    with torch.set_grad_enabled(False):
        image = pil_loader(img_path, size=1024)
        image = np.array(image)
        if(image.shape[2] ==3):
            image = np.transpose(image, (2,0,1))
            image = image / 255
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image,0)
        image = image.to(device=device, dtype=torch.float)
        bs, _, h, w = image.shape
        h_size = (h - 1) // image_size + 1
        w_size = (w - 1) // image_size + 1
        masks_pred = torch.zeros((1,2,1024,1024)).to(dtype=torch.float)
        for i in range(h_size):
            for j in range(w_size):
                h_max = min(h, (i + 1) * image_size)
                w_max = min(w, (j + 1) * image_size)
                inputs_part = image[:,:, i*image_size:h_max, j*image_size:w_max]
                masks_pred_single = model(inputs_part)
                masks_pred[:, :, i*image_size:h_max, j*image_size:w_max] = masks_pred_single

        mask_pred_softmax_batch = softmax(masks_pred).cpu().numpy()
        mask_soft_batch = mask_pred_softmax_batch[:, 1:, :, :]
        mask_soft_batch = np.squeeze(mask_soft_batch)
        mask_soft_batch = mask_soft_batch * 255
        new_mask = np.zeros((1024,1024,3))
        new_mask[:, :, 0] = mask_soft_batch
        new_mask[:, :, 1] = mask_soft_batch
        new_mask[:, :, 2] = mask_soft_batch

    return new_mask


   
