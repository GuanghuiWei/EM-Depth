from __future__ import absolute_import, division, print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import time

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from networks.depth_hrnet import DepthEncoder
from networks.depth_decoder_My2S import DepthDecoder_My2S


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.profile = self.opt.profile

        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.pretrain_youtube_weights is not None:
            print("The pretrain weights are use pretrain_youtube_weights")

            encoder_path = os.path.join(self.opt.pretrain_youtube_weights, "encoder.pth")
            decoder_path = os.path.join(self.opt.pretrain_youtube_weights, "depth.pth")

            encoder_weight_dict = torch.load(encoder_path)
            decoder_weight_dict = torch.load(decoder_path)

            self.models["encoder"] = DepthEncoder(
                self.opt.num_layers, pretrained=False)
            encoder_dict = self.models["encoder"].state_dict()
            self.models["encoder"].load_state_dict({k: v for k, v in encoder_weight_dict.items() if k in encoder_dict})
            encoder_matched_keys = {k: v for k, v in encoder_weight_dict.items() if k in encoder_dict}
            print('encoder_dict_numbers : {}'.format(len(encoder_dict)))
            print('encoder_weight_dict_numbers : {}'.format(len(encoder_weight_dict)))
            print('encoder_matched_keys_numbers : {}'.format(len(encoder_matched_keys)))
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

            self.models["depth"] = DepthDecoder_My2S(self.models["encoder"].num_ch_enc, self.opt.scales)
            decoder_dict = self.models["depth"].state_dict()
            self.models["depth"].load_state_dict({k: v for k, v in decoder_weight_dict.items() if k in decoder_dict})
            decoder_matched_keys = {k: v for k, v in decoder_weight_dict.items() if k in decoder_dict}
            print('decoder_dict_numbers : {}'.format(len(decoder_dict)))
            print('decoder_weight_dict_numbers : {}'.format(len(decoder_weight_dict)))
            print('decoder_matched_keys_numbers : {}'.format(len(decoder_matched_keys)))
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        else:
            print("The pretrain weights are use HRNet-Image-Classification")
            self.models["encoder"] = DepthEncoder(self.opt.num_layers, pretrained=True)
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

            self.models["depth"] = DepthDecoder_My2S(self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3],
                                                    weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),
            T_mul=1,
            eta_min=self.opt.lr[1],
            last_epoch=-1,
            max_lr=self.opt.lr[0],
            warmup_steps=0,
            gamma=0.9
        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIPseudoDepthDataset, #datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "nuscenes": datasets.nuScenesDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, self.opt.pose_height, self.opt.pose_width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, is_cutmix=True, is_local_crop=self.opt.use_local_crop, load_pseudo_depth=self.opt.use_pseudo_depth) # if train on cityscapes or nuscenes, please use is_local_crop=False, load_pseudo_depth=False.
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, multiprocessing_context="spawn")
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width, self.opt.pose_height, self.opt.pose_width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, is_cutmix=True, is_local_crop=self.opt.use_local_crop, load_pseudo_depth=self.opt.use_pseudo_depth) # if train on cityscapes or nuscenes, please use is_local_crop=False, load_pseudo_depth=False.
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, multiprocessing_context="spawn")
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                #self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder

            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

            features_scm = self.models["encoder"](inputs["color_aug_scm", 0, 0])
            outputs_scm = self.models["depth"](features_scm)
            outputs[("disp_scm", 0)] = outputs_scm[("disp", 0)]

            if self.opt.use_local_crop:
                features_local = self.models["encoder"](inputs["color_local", 0, 0])
                outputs_local = self.models["depth"](features_local)
                outputs[("disp_local", 0)] = outputs_local[("disp", 0)]
                disp_local = outputs_local[("disp", 0)]

        res_disp = self.restore_depth(inputs, outputs)
        disp = outputs[("disp", 0)]

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            outputs[("disp", scale)] = F.interpolate(
                disp, [h, w], mode="bilinear", align_corners=False)
            outputs[("disp_res", scale)] = F.interpolate(
                res_disp, [h, w], mode="bilinear", align_corners=False)
            if self.opt.use_local_crop:
                outputs[("disp_local", scale)] = F.interpolate(
                    disp_local, [h, w], mode="bilinear", align_corners=False)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug_pose", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            #self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp_scm = outputs[("disp_res", scale)]

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            disp_scale, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            disp_res_scale, depth_scm = disp_to_depth(disp_scm, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            outputs[("depth_scm", 0, scale)] = depth_scm

            outputs[("disp_scale", scale)] = disp_scale
            outputs[("disp_res_scale", scale)] = disp_res_scale

            if self.opt.use_local_crop:
                disp_local = outputs[("disp_local", scale)]
                disp_local_scale, depth_local = disp_to_depth(disp_local, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_local", 0, scale)] = depth_local
                outputs[("disp_local_scale", scale)] = disp_local_scale

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.use_local_crop:
                    Rt_Rc = torch.zeros_like(T).to(self.device)
                    gx0 = (inputs[("grid_local")][:, 0, 0, -1] + inputs[("grid_local")][:, 0, 0, 0]) / 2.
                    gy0 = (inputs[("grid_local")][:, 1, -1, 0] + inputs[("grid_local")][:, 1, 0, 0]) / 2.
                    f = (inputs[("grid_local")][:, 0, 0, -1] - inputs[("grid_local")][:, 0, 0, 0]) / 2.
                    fx = inputs[("K", 0)][0, 0, 0] / self.opt.width
                    fy = inputs[("K", 0)][0, 1, 1] / self.opt.height
                    Rc_v = torch.stack([-gx0 / (2 * fx), -gy0 / (2 * fy), f], dim=1)
                    Rc = torch.eye(3).to(self.device)
                    Rc = Rc[None, :, :].repeat(Rc_v.shape[0], 1, 1)
                    Rc[:, :, 2] = Rc_v
                    # outputs[("Rc", f_i)] = Rc
                    Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(T[:, :3, :3], torch.inverse(Rc)))
                    Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, T[:, :3, 3:4])
                    T_rc = Rt_Rc

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[scale](
                    depth, inputs[("inv_K", scale)])
                cam_points_scm = self.backproject_depth[scale](
                    depth_scm, inputs[("inv_K", scale)])

                pix_coords = self.project_3d[scale](
                    cam_points, inputs[("K", scale)], T)
                pix_coords_scm = self.project_3d[scale](
                    cam_points_scm, inputs[("K", scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("sample_scm", frame_id, scale)] = pix_coords_scm

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                outputs[("color_scm", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, scale)],
                    outputs[("sample_scm", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if self.opt.use_local_crop:
                    cam_points_local = self.backproject_depth[scale](
                        depth_local, inputs[("inv_K", scale)])
                    pix_coords_local = self.project_3d[scale](
                        cam_points_local, inputs[("K", scale)], T_rc)
                    outputs[("sample_local", frame_id, scale)] = pix_coords_local
                    outputs[("color_local", frame_id, scale)] = F.grid_sample(
                        inputs[("color_local", frame_id, scale)],
                        outputs[("sample_local", frame_id, scale)],
                        padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def normalize_tensor(self, tensor):
        B, C, H, W = tensor.shape

        flattened_tensor = tensor.reshape(B, -1)

        min_vals = torch.min(flattened_tensor, dim=1, keepdim=True)[0]
        max_vals = torch.max(flattened_tensor, dim=1, keepdim=True)[0]

        min_vals = min_vals.reshape(B, 1, 1, 1)
        max_vals = max_vals.reshape(B, 1, 1, 1)

        eps = 1e-6

        ranges = max_vals - min_vals + eps

        normalized_images = (tensor - min_vals) / ranges

        return normalized_images

    def median_tensor(self, tensor):
        reshaped_tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1)
        medians = torch.median(reshaped_tensor, dim=2, keepdim=True).values
        median_tensor = medians.reshape(tensor.size(0), tensor.size(1), 1, 1)
        return median_tensor

    def ssi_loss(self, pred, gt, eps=1e-7):
        normalized_pred = self.normalize_tensor(pred)
        normalized_gt = self.normalize_tensor(gt)

        t_d_pred = self.median_tensor(normalized_pred)
        t_d_gt = self.median_tensor(normalized_gt)

        s_d_pred = torch.mean(torch.abs(normalized_pred - t_d_pred), dim=(2, 3), keepdim=True)
        s_d_gt = torch.mean(torch.abs(normalized_gt - t_d_gt), dim=(2, 3), keepdim=True)

        scaled_shifted_pred = (normalized_pred - t_d_pred) / s_d_pred
        scaled_shifted_gt = (normalized_gt - t_d_gt) / s_d_gt

        loss = torch.mean(torch.log(torch.abs(scaled_shifted_pred - scaled_shifted_gt) + 1), dim=(2, 3), keepdim=True)

        return loss.mean()

    def compute_SI_log_depth_sc_loss(self, pred, target, mask=None, lamda=0.5):#, mask=None, lamda=0.5

        if mask is None:
            mask = torch.ones_like(pred).to(self.device)
        mask = mask[:, 0]
        log_pred = torch.log(pred[:, 0] + 1e-8) * mask
        log_tgt = torch.log(target[:, 0] + 1e-8) * mask

        log_diff = log_pred - log_tgt
        valid_num = mask.sum(1).sum(1) + 1e-8
        log_diff_squre_sum = (log_diff ** 2).sum(1).sum(1)
        log_diff_sum_squre = (log_diff.sum(1).sum(1)) ** 2
        loss = log_diff_squre_sum / valid_num - lamda * log_diff_sum_squre / (valid_num ** 2)

        grad_pred_x = log_pred[:, :, :-1] - log_pred[:, :, 1:]
        grad_pred_y = log_pred[:, :-1, :] - log_pred[:, 1:, :]

        grad_target_x = log_tgt[:, :, :-1] - log_tgt[:, :, 1:]
        grad_target_y = log_tgt[:, :-1, :] - log_tgt[:, 1:, :]

        grad_y_diff = (grad_pred_x - grad_target_x) ** 2
        grad_x_diff = (grad_pred_y - grad_target_y) ** 2
        return loss.mean() + 0.5 * grad_x_diff.mean() + 0.5 * grad_y_diff.mean()

    def compute_SI_log_depth_loss(self, pred, target, mask=None, lamda=0.5):#, mask=None, lamda=0.5

        if mask is None:
            mask = torch.ones_like(pred).to(self.device)
        mask = mask[:, 0]
        log_pred = torch.log(pred[:, 0] + 1e-8) * mask
        log_tgt = torch.log(target[:, 0] + 1e-8) * mask

        log_diff = log_pred - log_tgt
        valid_num = mask.sum(1).sum(1) + 1e-8
        log_diff_squre_sum = (log_diff ** 2).sum(1).sum(1)
        log_diff_sum_squre = (log_diff.sum(1).sum(1)) ** 2
        loss = log_diff_squre_sum / valid_num - lamda * log_diff_sum_squre / (valid_num ** 2)

        return loss.mean()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            reprojection_losses_scm = []
            reprojection_losses_local = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            # if scale == 0:   # if use smoothness
            #     disp = outputs[("disp", scale)]
            #     disp_scm = outputs[("disp_res", scale)]
            #     disp_local = outputs[("disp_local", scale)]
            #     # color_cm = inputs[("color_cm", 0, scale)]
            #     color_local = inputs[("color_local", 0, scale)]
            #     color = inputs[("color", 0, scale)]

            target = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                outputs[("reprojection_loss", frame_id, scale)] = self.compute_reprojection_loss(pred, target)
                reprojection_losses.append(outputs[("reprojection_loss", frame_id, scale)])
            reprojection_losses = torch.cat(reprojection_losses, 1)
            outputs[("min_loss", scale)], idxs_rep = torch.min(reprojection_losses, dim=1, keepdim=True)
            outputs[("mask_l", scale)] = idxs_rep == 1

            for frame_id in self.opt.frame_ids[1:]:
                pred_scm = outputs[("color_scm", frame_id, scale)]
                outputs[("reprojection_loss_scm", frame_id, scale)] = self.compute_reprojection_loss(pred_scm, target)
                reprojection_losses_scm.append(outputs[("reprojection_loss_scm", frame_id, scale)])
            reprojection_losses_scm = torch.cat(reprojection_losses_scm, 1)
            outputs[("min_loss_scm", scale)], idxs_rep_scm = torch.min(reprojection_losses_scm, dim=1, keepdim=True)
            outputs[("mask_scm_l", scale)] = idxs_rep_scm == 1

            if self.opt.use_local_crop:
                target_local = inputs[("color_local", 0, scale)]
                for frame_id in self.opt.frame_ids[1:]:
                    pred_local = outputs[("color_local", frame_id, scale)]
                    outputs[("reprojection_loss_local", frame_id, scale)] = self.compute_reprojection_loss(pred_local, target_local)
                    reprojection_losses_local.append(outputs[("reprojection_loss_local", frame_id, scale)])
                reprojection_losses_local = torch.cat(reprojection_losses_local, 1)
                outputs[("min_loss_local", scale)], idxs_rep_local = torch.min(reprojection_losses_local, dim=1, keepdim=True)
                outputs[("mask_local_l", scale)] = idxs_rep_local == 1

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                identity_reprojection_losses_local = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.use_local_crop:
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color_local", frame_id, scale)]
                        identity_reprojection_losses_local.append(
                            self.compute_reprojection_loss(pred, target_local))
                    identity_reprojection_losses_local = torch.cat(identity_reprojection_losses_local, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                combined_scm = torch.cat((identity_reprojection_loss, reprojection_losses_scm), dim=1)
                if self.opt.use_local_crop:
                    combined_local = torch.cat((identity_reprojection_losses_local, reprojection_losses_local), dim=1)
            else:
                combined = reprojection_loss
                combined_scm = reprojection_loss
                combined_local = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                to_optimise_scm, idxs_scm = torch.min(combined_scm, dim=1)
                outputs[("mask", scale)] = (idxs < 2).unsqueeze(1)
                outputs[("mask_scm", scale)] = (idxs_scm < 2).unsqueeze(1)

                if self.opt.use_local_crop:
                    to_optimise_local, idxs_local = torch.min(combined_local, dim=1)
                    outputs[("mask_local", scale)] = (idxs_local < 2).unsqueeze(1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            loss += to_optimise_scm.mean()
            if self.opt.use_local_crop:
                loss += to_optimise_local.mean()

            #if scale == 0:
                #mean_disp = disp.mean(2, True).mean(3, True)
                #norm_disp = disp / (mean_disp + 1e-7)
                #smooth_loss = get_smooth_loss(norm_disp, color)

                #mean_disp_scm = disp_scm.mean(2, True).mean(3, True)
                #norm_disp_scm = disp_scm / (mean_disp_scm + 1e-7)
                #smooth_loss_scm = get_smooth_loss(norm_disp_scm, color)

                #mean_disp_local = disp_local.mean(2, True).mean(3, True)
                #norm_disp_local = disp_local / (mean_disp_local + 1e-7)
                #smooth_loss_local = get_smooth_loss(norm_disp_local, color_local)

                #loss += self.opt.disparity_smoothness * smooth_loss_local / (2 ** scale)
                #loss += self.opt.disparity_smoothness * smooth_loss_scm / (2 ** scale)
                #loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss * (0.05 ** scale)
            losses["loss/{}".format(scale)] = loss

        with torch.no_grad():
            for frame_id in self.opt.frame_ids[1:]:
                for scale in self.opt.scales:
                    reprojection_loss = outputs[("reprojection_loss", frame_id, scale)]
                    reprojection_loss_scm = outputs[("reprojection_loss_scm", frame_id, scale)]

                    sample_best = outputs[("sample", frame_id, scale)].permute(0, 3, 1, 2)
                    sample_scm = outputs[("sample_scm", frame_id, scale)].permute(0, 3, 1, 2)

                    sample_teacher = torch.where((reprojection_loss_scm < reprojection_loss).repeat(1, 2, 1, 1),
                                                 sample_scm, sample_best)
                    outputs[("sample_best", frame_id, scale)] = sample_teacher

        dis_loss = 0
        for scale in self.opt.scales:
            mask = outputs[("mask", scale)].repeat(1, 2, 1, 1)
            mask_scm = outputs[("mask_scm", scale)].repeat(1, 2, 1, 1)
            mask_l = outputs[("mask_l", scale)].repeat(1, 2, 1, 1)
            mask_cm_l = outputs[("mask_scm_l", scale)].repeat(1, 2, 1, 1)
            depth_label_f = outputs[("sample_best", -1, scale)]
            depth_label_l = outputs[("sample_best", 1, scale)]

            dis_loss_depth_f = torch.log(
                torch.abs(depth_label_f - outputs[("sample", -1, scale)].permute(0, 3, 1, 2)) + 1)
            dis_loss_depth_scm_f = torch.log(
                torch.abs(depth_label_f - outputs[("sample_scm", -1, scale)].permute(0, 3, 1, 2)) + 1)

            dis_loss_depth_l = torch.log(
                torch.abs(depth_label_l - outputs[("sample", 1, scale)].permute(0, 3, 1, 2)) + 1)
            dis_loss_depth_scm_l = torch.log(
                torch.abs(depth_label_l - outputs[("sample_scm", 1, scale)].permute(0, 3, 1, 2)) + 1)

            dis_loss_depth = torch.where(mask_l, dis_loss_depth_l, dis_loss_depth_f)
            dis_loss_depth_scm = torch.where(mask_cm_l, dis_loss_depth_scm_l, dis_loss_depth_scm_f)

            dis_loss_depth = torch.where(mask, dis_loss_depth.detach(), dis_loss_depth)
            dis_loss_depth_scm = torch.where(mask_scm, dis_loss_depth_scm.detach(), dis_loss_depth_scm)

            dis_loss += (dis_loss_depth.mean() + dis_loss_depth_scm.mean()) * (0.05 ** scale)

        loss_dc_local = 0
        if self.opt.use_local_crop:
            for scale in self.opt.scales:
                h0 = self.opt.height // (2 ** scale)
                w0 = self.opt.width // (2 ** scale)
                disp = outputs[("disp", scale)]
                disp = F.interpolate(disp, [h0, w0], mode="bilinear", align_corners=False)
                loss_dc_i = 0
                for b in range(self.opt.batch_size):
                    disp_local = outputs[("disp_local", scale)][b].clone()
                    x0 = round(w0 * (inputs[("grid_local")][b, 0, 0, 0].item() - (-1)) / 2.)
                    y0 = round(h0 * (inputs[("grid_local")][b, 1, 0, 0].item() - (-1)) / 2.)
                    w = round(w0 / inputs[("ratio_local")][b, 0].item())
                    h = round(h0 / inputs[("ratio_local")][b, 0].item())
                    disp_local = F.interpolate(disp_local.unsqueeze(0), [h, w], mode="bilinear", align_corners=False)
                    _, depth_local = disp_to_depth(disp_local, self.opt.min_depth, self.opt.max_depth)
                    depth_local *= inputs[("ratio_local")][b, 0]
                    _, depth_from_ori = disp_to_depth(disp[b, :, y0:y0 + h, x0:x0 + w].clone().unsqueeze(0),
                                                      self.opt.min_depth, self.opt.max_depth)

                    loss_dc_i += self.compute_SI_log_depth_sc_loss(depth_local, depth_from_ori)
                loss_dc_i /= self.opt.batch_size
                loss_dc_local += (0.05 ** scale) * loss_dc_i

        loss_dc = 0
        for scale in self.opt.scales:
            depth = outputs[("depth", 0, scale)]
            depth_scm = outputs[("depth_scm", 0, scale)]
            # loss_dc += (0.05 ** scale) * self.compute_SI_log_depth_loss(depth_cm, depth)
            loss_dc += (0.05 ** scale) * self.compute_SI_log_depth_sc_loss(depth_scm, depth)

        pseudo_loss = 0
        if self.opt.use_pseudo_depth:
            pseudo_depths = inputs["pseudo_depth"].unsqueeze(1).permute(0, 1, 3, 2)
            for scale in self.opt.scales:
                H, W = self.opt.height // (2 ** scale), self.opt.width // (2 ** scale)
                pred_depth = outputs[("disp_scale", scale)]
                pred_depth_res = outputs[("disp_res_scale", scale)]
                pseudo_depth_scale = F.interpolate(pseudo_depths, [H, W], mode="bilinear", align_corners=False)
                # pseudo_loss += self.compute_SI_log_depth_sc_loss(pred_depth, pseudo_depth_scale) / (2 ** scale)
                # print(self.compute_SI_log_depth_sc_loss(pred_depth, pseudo_depth_scale) / (2 ** scale), scale)
                pseudo_loss += self.ssi_loss(pred_depth, pseudo_depth_scale) / (2 ** scale)
                pseudo_loss += self.ssi_loss(pred_depth_res, pseudo_depth_scale) / (2 ** scale)

        pseudo_loss_local = 0
        if self.opt.use_local_crop and self.opt.use_pseudo_depth:
            for scale in self.opt.scales:
                h0 = self.opt.height // (2 ** scale)
                w0 = self.opt.width // (2 ** scale)
                pseudo_depths = inputs["pseudo_depth"].unsqueeze(1).permute(0, 1, 3, 2)
                pseudo_depths = F.interpolate(pseudo_depths, [h0, w0], mode="bilinear", align_corners=False)
                pseudo_loss_local_i = 0
                for b in range(self.opt.batch_size):
                    disp_local = outputs[("disp_local_scale", scale)][b].clone()
                    x0 = round(w0 * (inputs[("grid_local")][b, 0, 0, 0].item() - (-1)) / 2.)
                    y0 = round(h0 * (inputs[("grid_local")][b, 1, 0, 0].item() - (-1)) / 2.)
                    w = round(w0 / inputs[("ratio_local")][b, 0].item())
                    h = round(h0 / inputs[("ratio_local")][b, 0].item())
                    disp_local = F.interpolate(disp_local.unsqueeze(0), [h, w], mode="bilinear", align_corners=False)
                    disp_local_sc, _ = disp_to_depth(disp_local, self.opt.min_depth, self.opt.max_depth)
                    depth_local = disp_local_sc / inputs[("ratio_local")][b, 0]
                    depth_from_ori = pseudo_depths[b, :, y0:y0 + h, x0:x0 + w].clone().unsqueeze(0)

                    pseudo_loss_local_i += self.ssi_loss(depth_local, depth_from_ori)
                pseudo_loss_local_i /= self.opt.batch_size
                pseudo_loss_local += (0.05 ** scale) * pseudo_loss_local_i

        if self.epoch < 5:
            weight_pseudo = 0.1
            
        else:
            weight_pseudo = 0.05

        if self.opt.use_local_crop:
            total_loss = total_loss + dis_loss + 0.15 * loss_dc + 0.15 * loss_dc_local

            if self.opt.use_pseudo_depth:
                total_loss =  total_loss + weight_pseudo * (pseudo_loss + pseudo_loss_local)
        else:
            total_loss = total_loss + dis_loss + 0.15 * loss_dc

        losses["loss"] = total_loss
        return losses

    def restore_depth(self, inputs, outputs):
        crop_index = inputs[("color_sindex", 0)]
        res_depth = []
        crop_height, crop_width = crop_index[:, 0], crop_index[:, 1]
        top_start_height, top_start_width = crop_index[:, 2], crop_index[:, 3]
        bottom_start_height, bottom_start_width = crop_index[:, 4], crop_index[:, 5]

        depth_cm = outputs[("disp_scm", 0)].clone()
        for i in range(self.opt.batch_size):
            minbat_crop_height = crop_height[i].item()
            minbat_crop_width = crop_width[i].item()
            minbat_top_start_height = top_start_height[i].item()
            minbat_top_start_width = top_start_width[i].item()
            minbat_bottom_start_height = bottom_start_height[i].item()
            minbat_bottom_start_width = bottom_start_width[i].item()
            minbat_depth_cm = depth_cm[i, :, :, :].unsqueeze(0)

            minbat_top_depth = minbat_depth_cm[:, :, minbat_top_start_height:minbat_top_start_height + minbat_crop_height,
                               minbat_top_start_width:minbat_top_start_width + minbat_crop_width].clone()
            minbat_boottom_depth = minbat_depth_cm[:, :, minbat_bottom_start_height:minbat_bottom_start_height + minbat_crop_height,
                                   minbat_bottom_start_width:minbat_bottom_start_width + minbat_crop_width].clone()

            minbat_depth_cm[:, :, minbat_bottom_start_height:minbat_bottom_start_height + minbat_crop_height,
            minbat_bottom_start_width:minbat_bottom_start_width + minbat_crop_width] = minbat_top_depth
            minbat_depth_cm[:, :, minbat_top_start_height:minbat_top_start_height + minbat_crop_height,
            minbat_top_start_width:minbat_top_start_width + minbat_crop_width] = minbat_boottom_depth

            res_depth.append(minbat_depth_cm)
        return torch.cat(res_depth, dim=0)

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

