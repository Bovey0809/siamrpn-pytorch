from __future__ import absolute_import, division
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from got10k.trackers import Tracker

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm


fig_dict = {}
patch_dict = {}


def save_frame(image, boxes=None, fig_n=1, linewidth=3, cmap=None, colors=None, legends=None, save_path='result'):
    """Save an image with rectangle(s) drawn on it."""
    # 确保保存目录存在
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 如果是PIL图像，转换为numpy数组
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 确保图像是BGR格式
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # 创建新的图形
    plt.figure(fig_n, figsize=(10, 10))
    plt.axis('off')
    plt.tight_layout()
    
    # 显示图像
    plt.imshow(image_rgb)

    # 绘制边界框
    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]
        
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + \
                list(mcolors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        # 为每个边界框创建一个矩形补丁
        for i, box in enumerate(boxes):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2], box[3],
                linewidth=linewidth,
                edgecolor=colors[i % len(colors)],
                facecolor='none',
                alpha=0.7 if len(boxes) > 1 else 1.0)
            plt.gca().add_patch(rect)
        
        if legends is not None:
            plt.legend(legends, loc=1, prop={'size': 8}, 
                      fancybox=True, framealpha=0.5)

    # 保存图像并关闭
    plt.savefig(f'{save_path}/frame_{fig_n:04d}.png', 
                bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        return self.inference(x, **self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)
        
        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls


class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)
        self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamRPN()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, 
                map_location=lambda storage, loc: storage,
                weights_only=True))
        self.net = self.net.to(self.device)

        # 新增参数
        self.top_k = 5  # 保存前K个候选框
        self.response_threshold = 0.5  # 响应阈值
        self.search_window = 1.5  # 搜索窗口扩展系数
        self.history_candidates = []  # 历史候选框列表

    def parse_args(self, **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = self._create_anchors(self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            self.cfg.exemplar_sz, self.avg_color)

        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)

    def update(self, image):
        image = np.asarray(image)
        
        # 原有的搜索图像处理代码
        instance_image = self._crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, self.avg_color)

        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls)
        
        # 计算所有候选框的响应值和位置
        offsets = out_reg.permute(1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]
        
        # 获取响应值
        response = F.softmax(out_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        
        # 计算惩罚项
        penalty = self._create_penalty(self.target_sz, offsets)
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        
        # 获取 top-K 个候选框
        top_k_ids = np.argsort(response)[-self.top_k:][::-1]
        top_k_responses = response[top_k_ids]
        top_k_offsets = offsets[:, top_k_ids]
        
        # 如果最佳响应值低于阈值，在历史候选框附近搜索
        best_id = top_k_ids[0]
        if response[best_id] < self.response_threshold and len(self.history_candidates) > 0:
            # 在历史候选框附近扩大搜索范围
            search_result = self._search_near_history(image)
            if search_result is not None:
                best_offset, best_response = search_result
                if best_response > response[best_id]:
                    offset = best_offset
                else:
                    offset = top_k_offsets[:, 0] * self.z_sz / self.cfg.exemplar_sz
            else:
                offset = top_k_offsets[:, 0] * self.z_sz / self.cfg.exemplar_sz
        else:
            offset = top_k_offsets[:, 0] * self.z_sz / self.cfg.exemplar_sz
        
        # 更新中心位置
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])
        
        # 更新尺度
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])
        
        # 更新搜索区域大小
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # 保存当前候选框到历史记录
        current_box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        self._update_history_candidates(current_box, response[best_id])
    
        return current_box
    
    def _update_history_candidates(self, box, response):
        """更新历史候选框列表"""
        self.history_candidates.append({
            'box': box.copy(),
            'response': response,
            'frame': len(self.history_candidates)
        })
        # 只保留最近的N帧
        if len(self.history_candidates) > 30:  # 可以调整历史帧数
            self.history_candidates.pop(0)

    def _search_near_history(self, image):
        """在历史候选框附近搜索"""
        best_response = -float('inf')
        best_offset = None
        
        # 遍历最近的历史候选框
        for candidate in reversed(self.history_candidates[-5:]):  # 只查找最近的5帧
            # 在候选框附近扩大搜索范围
            search_center = np.array([
                candidate['box'][1] + candidate['box'][3]/2,
                candidate['box'][0] + candidate['box'][2]/2
            ])
            
            # 使用更大的搜索窗口
            search_size = self.x_sz * self.search_window
            instance_image = self._crop_and_resize(
                image, search_center, search_size,
                self.cfg.instance_sz, self.avg_color)
            
            # 执行目标检测
            instance_image = torch.from_numpy(instance_image).to(
                self.device).permute(2, 0, 1).unsqueeze(0).float()
            with torch.set_grad_enabled(False):
                out_reg, out_cls = self.net.inference(
                    instance_image, self.kernel_reg, self.kernel_cls)
            
            # 计算响应值
            response = F.softmax(out_cls.permute(
                1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
            
            # 如果找到更好的响应，更新结果
            max_response = np.max(response)
            if max_response > best_response:
                best_response = max_response
                offsets = out_reg.permute(
                    1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
                best_id = np.argmax(response)
                best_offset = offsets[:, best_id]
        
        return (best_offset, best_response) if best_offset is not None else None

    def _create_anchors(self, response_sz):
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in self.cfg.scales:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)
        
        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
            self.cfg.penalty_k)

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def track(self, imgs, box, visualize=False):
        frame_num = len(imgs)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        logging.info(f"Tracking {frame_num} frames")
        progress_bar = tqdm(total=len(imgs), desc="Tracking Progress")
        for f, img in enumerate(imgs):
            start_time = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - start_time

            if visualize:
                save_frame(img, boxes[f, :], fig_n=f+1)
            
            progress_bar.update(1)
        progress_bar.close()

        return boxes, times