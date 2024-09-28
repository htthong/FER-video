import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a PIL Image."""
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0)  # Convert CHW to HWC
    tensor = (tensor * 255).byte()  # Scale to [0, 255] and convert to uint8
    return Image.fromarray(tensor.numpy())

def image_to_base64(image):
    """Convert a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def tensors_to_base64_images(tensors, titles):
    base64_images = []
    for tensor, title_list in zip(tensors, titles):
        images = []
        for i in range(tensor.size(2)):
            image = tensor_to_image(tensor[0, :, i])
            image_base64 = image_to_base64(image)
            images.append({'title': title_list[i], 'base64': image_base64})
        base64_images.append(images)
    return base64_images

def get_index(inp, x_origin, x_sample):
    org_idx = []
    spl_idx = []

    for i in range(16):
        for j in range(32):
            if torch.equal(x_origin[i], inp[j]):
                org_idx.append(j)
                break

    for i in range(16):
        for j in range(32):
            if torch.equal(x_sample[i], inp[j]):
                spl_idx.append(j)
                break
    return org_idx, spl_idx

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)


class AverageMeter_img(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()