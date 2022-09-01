# -*- coding: utf-8 -*-

"""
Generic Image Transform utillities.
"""

import cv2
import random, math
import numpy as np
from collections import Iterable

import torch.nn.functional as F
from torch.autograd import Variable


class ResizePad:
    """
    Resize and pad an image to given size.
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        resized_img = cv2.resize(img, (resized_w, resized_h))

        # if img.ndim > 2:
        if img.ndim > 2:
            new_img = np.zeros(
                (self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)
        else:
            resized_img = np.expand_dims(resized_img, -1)
            new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)
        new_img[pad_h: pad_h + resized_h,
                pad_w: pad_w + resized_w, ...] = resized_img
        return new_img


class CropResize:
    """Remove padding and resize image to its original size."""

    def __call__(self, img, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        im_h, im_w = img.data.shape[:2]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        # scale = torch.Tensor([[input_h / im_h, input_w / im_w]]).max()
        resized_h = int(np.round(im_h * scale))
        # resized_h = torch.round(im_h * scale)
        resized_w = int(np.round(im_w * scale))
        # resized_w = torch.round(im_w * scale)
        crop_h = int(np.floor(resized_h - input_h) / 2)
        # crop_h = torch.floor(resized_h - input_h) // 2
        crop_w = int(np.floor(resized_w - input_w) / 2)
        # crop_w = torch.floor(resized_w - input_w) // 2
        # resized_img = cv2.resize(img, (resized_w, resized_h))
        resized_img = F.upsample(
            img.unsqueeze(0).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear',align_corners=True)

        resized_img = resized_img.squeeze().unsqueeze(0)

        return resized_img[0, crop_h: crop_h + input_h,
                           crop_w: crop_w + input_w]


class ResizeImage:
    """Resize the largest of the sides of the image to a given size"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            Variable(img).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear',align_corners=True).squeeze().data
        return out


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            Variable(img).unsqueeze(0).unsqueeze(0),
            size=(resized_h, resized_w),
            mode='bilinear',align_corners=True).squeeze().data
        return out


class ToNumpy:
    """Transform an torch.*Tensor to an numpy ndarray."""

    def __call__(self, x):
        return x.numpy()

def letterbox(img, mask, height, color=(123.7, 116.3, 103.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        # print(top, bottom, left, right)
        # input()
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)  # padded square
        # print(mask)
    return img, mask, ratio, dw, dh

def random_affine(img, mask, targets, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5), all_bbox=None):
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    if mask is not None:
        maskw = cv2.warpPerspective(mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST,
                                  borderValue=1)  # BGR order borderValue
    else:
        maskw = None

    # Return warped points also
    if type(targets)==type([1]):
        targetlist=[]
        for bbox in targets:
            targetlist.append(wrap_points(bbox, M, height, a))
        return imw, maskw, targetlist, M
    elif all_bbox is not None:
        targets = wrap_points(targets, M, height, a)
        for ii in range(all_bbox.shape[0]):
            all_bbox[ii,:] = wrap_points(all_bbox[ii,:], M, height, a)
        return imw, maskw, targets, all_bbox, M
    elif targets is not None:   ## previous main
        targets = wrap_points(targets, M, height, a)
        return imw, maskw, targets, M
    else:
        return imw

def wrap_points(targets, M, height, a):
    # n = targets.shape[0]
    # points = targets[:, 1:5].copy()
    points = targets.copy()
    # area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
    area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, 1).T

    # reject warped points outside of image
    np.clip(xy, 0, height, out=xy)
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    ## print(targets, xy)
    ## [ 56  36 108 210] [[ 47.80464857  15.6096533  106.30993434 196.71267693]]
    # targets = targets[i]
    # targets[:, 1:5] = xy[i]
    targets = xy[0]
    return targets   