import torch
import torch.nn as nn

from model.BaseModel import BaseModel
from utils.data import get_from_dict
from utils.config import cfg_has
from .IntrinsicsNet import IntrinsicsNet
from .MonoDepthNet import MonoDepthNet
from .PoseNet import PoseNet

class zzyNet(BaseModel):
    """
    zzyNet: 用于自监督相机内参校正、深度估计和位姿估计

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super(zzyNet, self).__init__(cfg)

        self.intrinsics_net = IntrinsicsNet(cfg.intrinsics)
        self.depth_net = MonoDepthNet(cfg.depth)
        self.pose_net = PoseNet(cfg.pose)

        self.set_attr(cfg.model, 'use_gt_pose', False)
        self.set_attr(cfg.model, 'use_gt_intrinsics', True)

    def forward(self, batch, epoch=0):
        """
        batch: 训练/推理时输入的数据字典
        epoch: (可选)当前训练 epoch，若不需要可以删除
        """
        tgt = (0, 0)
        rgb = batch['rgb']

        # 确保 tgt 可访问
        if tgt not in rgb:
            raise ValueError(f"Target frame {tgt} not found in batch['rgb']")

        tgt_rgb = rgb[tgt]

        # 选择 GT intrinsics 或者预测 intrinsics
        if self.use_gt_intrinsics:
            intrinsics = get_from_dict(batch, 'intrinsics')
            if intrinsics is None:
                raise ValueError("GT intrinsics requested, but missing in batch")
        else:
            intrinsics = self.intrinsics_net(rgb=tgt_rgb)

        valid_mask = get_from_dict(batch, 'mask')

        # 预测深度
        depth_output = self.depth_net(rgb=tgt_rgb, intrinsics=intrinsics[tgt] if self.use_gt_intrinsics else None)
        pred_depth = depth_output.get('depths', depth_output.get('depth'))
        if pred_depth is None:
            raise ValueError("MonoDepthNet output missing 'depth' or 'depths'")

        predictions = {'depth': pred_depth}

        # 记录不确定度 logvar（如果有）
        pred_logvar = depth_output.get('logvar')
        if pred_logvar is not None:
            predictions['logvar'] = pred_logvar

        # 推理模式下直接返回
        if not self.training:
            return {'predictions': predictions}

        # 选择 GT 还是预测 pose
        if self.use_gt_pose:
            pose = batch.get('pose')
            if pose is None:
                raise ValueError("GT pose requested, but missing in batch")
        else:
            pose_output = self.pose_net(rgb)
            pose = pose_output.get('transformation')
            if pose is None:
                raise ValueError("PoseNet output missing 'transformation'")

        predictions['pose'] = pose

        return {'predictions': predictions}
