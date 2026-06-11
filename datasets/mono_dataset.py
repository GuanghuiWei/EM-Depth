from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy

import torchvision
from PIL import Image

import torch
import torch.utils.data as data

import cv2
import albumentations as A
from albumentations.pytorch import ToTensor
import albumentations.augmentations.transforms as transforms


def cv2_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        height_pose
        width_pose
        frame_idxs
        num_scales
        is_train
        img_ext
        is_cutmix
        is_local_crop
        is_cityscapes
        is_patch_reshuffle
        load_pseudo_depth
        is_use_grid
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 height_pose,
                 width_pose,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png',
                 is_cutmix=False,
                 is_local_crop=False,
                 is_cityscapes=False, 
                 is_patch_reshuffle=False,
                 load_pseudo_depth=False,
                 is_use_grid=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.height_pose = height_pose
        self.width_pose = width_pose
        self.num_scales = num_scales

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.is_cutmix = is_cutmix
        self.local_crop = is_local_crop
        self.patch_reshuffle = is_patch_reshuffle
        self.use_grid = is_use_grid

        self.loader = cv2_loader
        self.to_tensor = ToTensor
        self.interp = cv2.INTER_LANCZOS4  # Image.ANTIALIAS
        self.is_cityscapes = is_cityscapes
        
        self.load_pseudo_depth = load_pseudo_depth

        self.cam_name = "FRONT"
        self.img_type = "downsample"
        if self.is_train:
            self.load_depth = False
            self.load_mask = False
        else:
            self.load_depth = True
            self.load_mask = False
        self.max_lidar_num = 25000  #used to pad for batching

        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        self.resize = {}
        self.resize_cityscapes = {}

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = A.Resize(self.height // s, self.width // s, interpolation=cv2.INTER_LANCZOS4)
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize_cityscapes[i] = torchvision.transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)
        self.resize_pose = A.Resize(self.height_pose, self.width_pose, interpolation=cv2.INTER_LANCZOS4)
        self.resize_pseudo_depth = A.Resize(self.height, self.width, interpolation=cv2.INTER_LANCZOS4)

        self.load_depth = self.check_depth()
        #self.load_depth = False

    def preprocess(self, inputs, color_aug, color_aug_cm, do_color_aug, is_cityscapes=False):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        if self.local_crop:
            self.resize_ratio_upper, self.resize_ratio_lower = 2.0, 1.2
            resize_ratio = (self.resize_ratio_upper - self.resize_ratio_lower) * random.random() + self.resize_ratio_lower
            height_re = int(self.height * resize_ratio)
            width_re = int(self.width * resize_ratio)
            w0 = int((width_re - self.width) * random.random())
            h0 = int((height_re - self.height) * random.random())
            self.resize_local = A.Resize(height_re, width_re, interpolation=cv2.INTER_LANCZOS4)
            box = (w0, h0, w0 + self.width, h0 + self.height)
            gridx, gridy = np.meshgrid(np.linspace(-1, 1, width_re), np.linspace(-1, 1, height_re))
            gridx = torch.from_numpy(gridx)
            gridy = torch.from_numpy(gridy)
            grid = torch.stack([gridx, gridy], dim=0)
            inputs[("grid_local")] = grid[:, h0: h0 + self.height, w0: w0 + self.width].clone()
            inputs[("ratio_local")] = torch.tensor([resize_ratio])

        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    if is_cityscapes:
                        inputs[(n, im, i)] = self.resize_cityscapes[i](inputs[(n, im, i - 1)])
                    else:
                        if self.local_crop:
                            if i == 0:
                                color_local = self.resize_local(image=inputs[(n, im, -1)])['image']
                                inputs[(n + "_local", im, i)] = A.crop(color_local, w0, h0, w0 + self.width, h0 + self.height)
                        inputs[(n, im, i)] = self.resize[i](image=inputs[(n, im, i - 1)])['image']

        if self.local_crop:
            for k in list(inputs):
                frame = inputs[k]
                if "color_local" in k:
                    n, im, i = k
                    for i in range(1, self.num_scales):
                        inputs[(n, im, i)] = self.resize[i](image=inputs[(n, im, i - 1)])['image']

            for k in list(inputs):
                f = inputs[k]
                if "color_local" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor()(image=f)['image']

        for k in list(inputs):
            f = inputs[k]
            if is_cityscapes:
                f = np.array(f)
            if "color" in k:
                n, im, i = k
                if i == 0 and im == 0 and self.is_cutmix:
                    color = np.copy(f)
                    aug_color = color_aug_cm(image=np.copy(color))['image']

                elif i == 0 and im == -1 and self.is_cutmix:
                    color_f_f = np.copy(f)
                    color_f = color_aug(image=color_f_f)['image']

                elif i == 0 and im == 1 and self.is_cutmix:
                    color_l_l = np.copy(f)
                    color_l = color_aug(image=color_l_l)['image']

                inputs[(n, im, i)] = self.to_tensor()(image=f)['image']
                if do_color_aug:
                    aug_img = color_aug(image=f)['image']
                else:
                    aug_img = color_aug(f)

                if i == 0 and im == 0 and self.is_cutmix and self.patch_reshuffle:
                    sp_color = np.copy(aug_img)

                inputs[(n + "_aug", im, i)] = self.to_tensor()(image=aug_img)['image']

                # For high resolution input, e.g. 1024 × 320 and 1280 × 384,
                # the posenet takes a lower resolution 640x192 as input for memory savings.
                if i == 0 and self.height_pose != 0 and self.width_pose != 0:
                    if self.height_pose == self.height and self.width_pose == self.width:
                        aug_img_pose = aug_img
                    else:
                        aug_img_pose = self.resize_pose(image=aug_img)['image']
                    inputs[(n + "_aug_pose", im, 0)] = self.to_tensor()(image=aug_img_pose)['image']

        if self.is_cutmix:
        
            if self.patch_reshuffle:
                self.split_ratio_lower = 0.1
                self.split_ratio_upper = 0.9
                ## Split-Permute as depicted in paper (vertical + horizontal)
                img_aug = Image.fromarray(sp_color)
                newimg_aug = img_aug.copy()
                ratio_x = random.random() * (
                            self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                ratio_y = random.random() * (
                            self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower

                w_i = int(self.width * ratio_x)
                patch1_aug = img_aug.crop((0, 0, w_i, self.height)).copy()  # (left, upper, right, lower)
                patch2_aug = img_aug.crop((w_i, 0, self.width, self.height)).copy()
                newimg_aug.paste(patch2_aug, (0, 0))
                newimg_aug.paste(patch1_aug, (self.width - w_i, 0))

                h_i = int(self.height * ratio_y)
                patch1_aug = newimg_aug.crop((0, 0, self.width, h_i)).copy()
                patch2_aug = newimg_aug.crop((0, h_i, self.width, self.height)).copy()
                newimg_aug.paste(patch2_aug, (0, 0))
                newimg_aug.paste(patch1_aug, (0, self.height - h_i))

                newimg_aug = np.array(newimg_aug)
                inputs[("color_aug_reshuffle", 0, 0)] = self.to_tensor()(image=newimg_aug)['image']
                inputs[("split_xy")] = torch.tensor([self.width - w_i, self.height - h_i])

            else:

                if len(self.frame_idxs) == 2:
                    color_f, color_l = np.copy(color), np.copy(color)

                if self.use_grid:
                    grid = inputs["grid"]
                    color_cm, color_cm_f, color_cm_l, color_cm_aug, grid_cm, crop_index = \
                        self.cut_mix_color_and_grid(color, color_f, color_l, aug_color, grid)
                else:
                    color_cm, color_cm_f, color_cm_l, color_cm_aug, crop_index = \
                        self.cut_mix_color(color, color_f, color_l, aug_color)
    
                if self.use_grid:
                    inputs["grid_scm"] = grid_cm

                inputs[("color_aug_scm", 0, 0)] = self.to_tensor()(image=color_cm_aug)['image']
                inputs[("color_scm", 0, 0)] = self.to_tensor()(image=color_cm)['image']
                inputs[("color_aug_scm", -1, 0)] = self.to_tensor()(image=color_cm_f)['image']
                inputs[("color_aug_scm", 1, 0)] = self.to_tensor()(image=color_cm_l)['image']
                inputs[("color_sindex", 0)] = torch.tensor(crop_index)

                for i in range(1, 4):
                    color_res = self.resize[i](image=color_cm)['image']
                    color_res_f = self.resize[i](image=color_cm_f)['image']
                    color_res_l = self.resize[i](image=color_cm_l)['image']
                    color_res_aug = self.resize[i](image=color_cm_aug)['image']
                    inputs[("color_scm", 0, i)] = self.to_tensor()(image=color_res)['image']
                    inputs[("color_scm", -1, i)] = self.to_tensor()(image=color_res_f)['image']
                    inputs[("color_scm", 1, i)] = self.to_tensor()(image=color_res_l)['image']
                    inputs[("color_aug_scm", 0, i)] = self.to_tensor()(image=color_res_aug)['image']

    def __len__(self):
        return len(self.filenames)

    # def load_intrinsics(self, folder, frame_index):
    # return self.K.copy()

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
            folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
            inputs.update(self.get_colors(folder, frame_index, side, do_flip))

        elif type(self).__name__ in ["nuScenesDataset"]:

            line = self.filenames[index].split()
            folder = line[0]
            frame_index = int(line[1])
            if len(line) == 3:
                side = line[2]
            else:
                side = 'l'
            for i in self.frame_idxs:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        else:
            line = self.filenames[index].split()
            folder = line[0]

            if len(line) == 3:
                frame_index = int(line[1])
            else:
                frame_index = 0

            if len(line) == 3:
                side = line[2]
            else:
                side = None

            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                else:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
                K = self.load_intrinsics(folder, frame_index)

            elif type(self).__name__ in ["nuScenesDataset"]:
                self.K_nus = self.get_intrinsic(folder).copy()
                K = self.K_nus.copy()

            else:
                K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(brightness=self.brightness,
                                               contrast=self.contrast,
                                               saturation=self.saturation,
                                               hue=self.hue,
                                               always_apply=False, p=1)
            color_aug_cm = A.NoOp(p=1)     #color_aug_cm = (lambda x: x)

        else:

            color_aug = A.NoOp(p=1)   #color_aug = (lambda x: x)

            color_aug_cm = transforms.ColorJitter(brightness=self.brightness,
                                                  contrast=self.contrast,
                                                  saturation=self.saturation,
                                                  hue=self.hue,
                                                  always_apply=False, p=1)
        if self.use_grid:
            x = torch.linspace(-1, 1, self.width)
            y = torch.linspace(-1, 1, self.height)
            grid_y, grid_x = torch.meshgrid(y, x)
            grid = torch.stack([grid_x, grid_y], dim=0)
            inputs["grid"] = grid
            
        self.preprocess(inputs, color_aug, color_aug_cm, do_color_aug, self.is_cityscapes)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_pseudo_depth:
            pseudo_depth = self.get_depth(folder, frame_index, side, do_flip)
            pseudo_depth = self.resize_pseudo_depth(image=pseudo_depth)['image']
            inputs["pseudo_depth"] = self.to_tensor()(image=pseudo_depth)['image']
            
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def cut_mix_color(self, color, color_f, color_l, color_aug):

        crop_index = []
        min_space = 4

        crop_height = random.randint(self.height // 6, self.height)
        crop_width = random.randint(self.width // 6, self.width // 2)

        if crop_height > self.height - 2 * min_space:
            list_wh = [0, self.height - crop_height]
            lift_start_height = right_start_height = random.choice(list_wh)
        else:
            lift_start_height_list = [0] + list(range(min_space, self.height - crop_height - min_space)) + [
                self.height - crop_height]
            lift_start_height = random.choice(lift_start_height_list)
            if lift_start_height == 0 or lift_start_height == self.height - crop_height:
                right_start_height = lift_start_height
            else:
                adv_weight = list(range(-self.height // 12, self.height // 12, min_space))
                adv_lift_start_height = max(min_space, int(lift_start_height + random.choice(adv_weight)))
                right_start_height = max(min_space, min(adv_lift_start_height, self.height - crop_height - min_space))

        if random.random() < 0.5:
            lift_start_height = lift_start_height
            right_start_height = right_start_height
        else:
            y = right_start_height
            right_start_height = lift_start_height
            lift_start_height = y

        lift_start_width_list = [0] + list(range(min_space, self.width - 2 * crop_width - min_space)) + [
            self.width - 2 * crop_width]
        lift_start_width = random.choice(lift_start_width_list)
        if lift_start_width > self.width - 2 * crop_width - 2 * min_space:
            right_start_width = lift_start_width + crop_width
        else:
            right_start_width_list = [lift_start_width + crop_width] + list(
                range(lift_start_width + crop_width + min_space, self.width - crop_width - min_space)) + [
                                         self.width - crop_width]
            right_start_width = random.choice(right_start_width_list)

        if random.random() < 0.5:
            lift_start_width = lift_start_width
            right_start_width = right_start_width
        else:
            x = lift_start_width
            lift_start_width = right_start_width
            right_start_width = x

        crop_index.append(crop_height)
        crop_index.append(crop_width)
        crop_index.append(lift_start_height)
        crop_index.append(lift_start_width)
        crop_index.append(right_start_height)
        crop_index.append(right_start_width)

        lift_crop_color = np.copy(
            color[lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width])

        right_crop_color = np.copy(
            color[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color
        color[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color

        lift_crop_color_f = np.copy(
            color_f[lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width])

        right_crop_color_f = np.copy(
            color_f[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color_f[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color_f
        color_f[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color_f

        lift_crop_color_l = np.copy(
            color_l[lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width])

        right_crop_color_l = np.copy(
            color_l[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color_l[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color_l
        color_l[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color_l

        lift_crop_color_aug = np.copy(
            color_aug[lift_start_height:lift_start_height + crop_height,
            lift_start_width:lift_start_width + crop_width])

        right_crop_color_aug = np.copy(
            color_aug[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color_aug[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color_aug
        color_aug[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color_aug

        return color, color_f, color_l, color_aug, crop_index


    def cut_mix_color_and_grid(self, color, color_f, color_l, color_aug, grid):

        crop_index = []
        min_space = 4

        crop_height = random.randint(self.height // 6, self.height)
        crop_width = random.randint(self.width // 6, self.width // 2)

        if crop_height > self.height - 2 * min_space:
            list_wh = [0, self.height - crop_height]
            lift_start_height = right_start_height = random.choice(list_wh)
        else:
            lift_start_height_list = [0] + list(range(min_space, self.height - crop_height - min_space)) + [
                self.height - crop_height]
            lift_start_height = random.choice(lift_start_height_list)
            if lift_start_height == 0 or lift_start_height == self.height - crop_height:
                right_start_height = lift_start_height
            else:
                adv_weight = list(range(-self.height // 12, self.height // 12, min_space))
                adv_lift_start_height = max(min_space, int(lift_start_height + random.choice(adv_weight)))
                right_start_height = max(min_space, min(adv_lift_start_height, self.height - crop_height - min_space))

        if random.random() < 0.5:
            lift_start_height = lift_start_height
            right_start_height = right_start_height
        else:
            y = right_start_height
            right_start_height = lift_start_height
            lift_start_height = y

        lift_start_width_list = [0] + list(range(min_space, self.width - 2 * crop_width - min_space)) + [
            self.width - 2 * crop_width]
        lift_start_width = random.choice(lift_start_width_list)
        if lift_start_width > self.width - 2 * crop_width - 2 * min_space:
            right_start_width = lift_start_width + crop_width
        else:
            right_start_width_list = [lift_start_width + crop_width] + list(
                range(lift_start_width + crop_width + min_space, self.width - crop_width - min_space)) + [
                                         self.width - crop_width]
            right_start_width = random.choice(right_start_width_list)

        if random.random() < 0.5:
            lift_start_width = lift_start_width
            right_start_width = right_start_width
        else:
            x = lift_start_width
            lift_start_width = right_start_width
            right_start_width = x

        crop_index.append(crop_height)
        crop_index.append(crop_width)
        crop_index.append(lift_start_height)
        crop_index.append(lift_start_width)
        crop_index.append(right_start_height)
        crop_index.append(right_start_width)

        lift_crop_color = np.copy(
            color[lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width])

        right_crop_color = np.copy(
            color[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color
        color[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color

        lift_crop_color_f = np.copy(
            color_f[lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width])

        right_crop_color_f = np.copy(
            color_f[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color_f[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color_f
        color_f[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color_f

        lift_crop_color_l = np.copy(
            color_l[lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width])

        right_crop_color_l = np.copy(
            color_l[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color_l[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color_l
        color_l[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color_l

        lift_crop_color_aug = np.copy(
            color_aug[lift_start_height:lift_start_height + crop_height,
            lift_start_width:lift_start_width + crop_width])

        right_crop_color_aug = np.copy(
            color_aug[right_start_height:right_start_height + crop_height,
            right_start_width:right_start_width + crop_width])

        color_aug[lift_start_height:lift_start_height + crop_height,
        lift_start_width:lift_start_width + crop_width] = right_crop_color_aug
        color_aug[right_start_height:right_start_height + crop_height,
        right_start_width:right_start_width + crop_width] = lift_crop_color_aug

        grid_cm = grid.clone()

        grid_cm[:, lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width] = \
            grid[:, right_start_height:right_start_height + crop_height, right_start_width:right_start_width + crop_width]

        grid_cm[:, right_start_height:right_start_height + crop_height, right_start_width:right_start_width + crop_width] = \
            grid[:, lift_start_height:lift_start_height + crop_height, lift_start_width:lift_start_width + crop_width]

        return color, color_f, color_l, color_aug, grid_cm, crop_index