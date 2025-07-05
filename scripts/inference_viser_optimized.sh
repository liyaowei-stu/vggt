#!/bin/bash

export XDG_CACHE=/group/40034/share/zhaoyangzhang/PretrainedCache
export TORCH_HOME=/group/40034/share/zhaoyangzhang/PretrainedCache
export HF_HOME=/group/40034/share/zhaoyangzhang/PretrainedCache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ENV_VENUS_PROXY=http://zzachzhang:rmdRjCXJAhvOXxhE@vproxy.woa.com:31289
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcloudapi.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
export NCCL_TOPO_FILE=/tmp/topo.txt

export CUDA_VISIBLE_DEVICES=0

# 性能优化参数
# --max_points: 限制点云数量，减少渲染负担
# --point_size: 调整点大小，提高可见性
# --update_throttle: 节流更新频率，减少卡顿
# --conf_threshold: 提高初始置信度阈值，减少低质量点

python demo_viser.py \
    --image_folder /group/40034/yaoweili/code/image_generation/ic_custom/vggt/data/navi/navi_v1.5/soldier_wood_showpiece/multiview-04-pixel_6pro/masked_images/ \
    --max_points 500000 \
    --point_size 0.002 \
    --conf_threshold 20.0 \
    --use_point_map