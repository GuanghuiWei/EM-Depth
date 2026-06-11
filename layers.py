from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def updown_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="nearest") #"nearest"
    
def downsample(x):
    """Downsample input tensor by a factor of 1/2
    """
    return F.interpolate(x, scale_factor=1.0/2, mode="bilinear")

def up_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="bilinear")#bilinear

def upsample(x, scale_factor=2, mode="nearest"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

def to_inv(depth, eps=1e-7):
    return (depth > 0) / (depth + eps)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv1and3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv1and3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv3x3 = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=1)
        self.conv1x1 = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.pad(x)
        out_3x3 = self.conv3x3(out)
        out_1x1 = self.conv1x1(x)
        out = out_1x1 + out_3x3

        return out

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_smooth_loss_cm(disp, img, mask_x, mask_y):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x *= mask_x
    grad_disp_y *= mask_y

    return grad_disp_x.mean() + grad_disp_y.mean()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class DepthMetrics(nn.Module):
    """ Compute depth performance
    """

    def __init__(self, img_bound, min_depth, max_depth):
        super(DepthMetrics, self).__init__()
        self.depth_metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']
        self.img_bound = img_bound
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, inputs, outputs, mask=None):
        disp_pred = outputs[('disp_scaled', 0, 0)]  # (B, 1, H, W)
        # print(disp_pred.shape)
        depth_gt = inputs['depth_gt']  # (B, dataset.max_depth_samp, 3)    # include padding
        depth_valid = inputs['depth_valid']  # (B, dataset.max_depth_samp,)      # specify original
        gt_dim = inputs['gt_dim']  # (B, 2,) - ground truth image dim
        # gt_dim = [900, 1600]  # (B, 2,) - ground truth image dim

        metrics = {metric: 0 for metric in self.depth_metric_names}
        if mask is not None:
            mask_labels = [l.item() for l in torch.unique(mask)]  # split based on mask values
            metrics.update(
                {f'{metric}_mask': {l: [0, 0] for l in mask_labels} for metric in self.depth_metric_names})

        for bi, (disp_p, depth_g, valid, dim) in enumerate(zip(disp_pred, depth_gt, depth_valid, gt_dim)):

            gt_height, gt_width = dim[0].item(), dim[1].item()
            # print(gt_height, gt_width)
            up, down = int(self.img_bound[0] * gt_height), int(self.img_bound[1] * gt_height)
            left, right = int(self.img_bound[2] * gt_width), int(self.img_bound[3] * gt_width)

            valid = torch_and(valid,
                              depth_g[:, 0] >= up,  # check for image bounds
                              depth_g[:, 0] < down,
                              valid, depth_g[:, 1] >= left,
                              valid, depth_g[:, 1] < right,
                              depth_g[:, 2] > self.min_depth,  # check for depth bounds
                              depth_g[:, 2] < self.max_depth)

            valid_ind = depth_g[:, 0][valid].long(), depth_g[:, 1][valid].long()
            depth_p = 1 / nn.functional.interpolate(disp_p[None], (gt_height, gt_width), mode='bilinear',
                                                    align_corners=False).squeeze()

            d_gt = depth_g[:, 2][valid]
            d_pd = depth_p[valid_ind]

            # median scaling and clamp
            d_pd *= torch.median(d_gt) / torch.median(d_pd)
            d_pd = torch.clamp(d_pd, self.min_depth, self.max_depth)

            depth_errors = compute_errors(d_gt, d_pd)
            for i, metric in enumerate(self.depth_metric_names):
                metrics[metric] += depth_errors[i]

            if mask is not None:
                m_valid = mask[bi][valid_ind]
                for l in mask_labels:
                    m = m_valid == l

                    dgm, dpm = d_gt[m], d_pd[m]
                    cnt = dgm.shape[0]
                    if cnt == 0:
                        continue
                    depth_errors = compute_errors(dgm, dpm)

                    for i, metric in enumerate(self.depth_metric_names):
                        metrics[f'{metric}_mask'][l][0] += depth_errors[i].item() * cnt
                        metrics[f'{metric}_mask'][l][1] += cnt

        for metric in self.depth_metric_names:
            metrics[metric] = metrics[metric] / disp_pred.size(0)

        return metrics

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def meshgrid(height, width, is_homogeneous=True):
    x = torch.ones(height).float().view(height, 1)
    # shape : (h,w)
    x = torch.matmul(x, torch.linspace(0, 1, width).view(1, width))

    y = torch.linspace(0, 1, height).view(height, 1)
    # shape : (h,w)
    y = torch.matmul(y, torch.ones(width).float().view(1, width))

    x = x * (width - 1)
    y = y * (height - 1)

    if is_homogeneous:
        ones = torch.ones(height, width).float()
        coords = torch.stack((x, y, ones), dim=2)  # shape: h,w,3
    else:
        coords = torch.stack((x, y), dim=2)  # shape: h,w,2

    # shape:  (h, w, 2 or 3)
    return coords

