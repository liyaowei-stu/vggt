
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

python /group/40034/yaoweili/code/image_generation/ic_custom/vggt/inference_pts_to_rendering.py \
    --image_path /group/40034/yaoweili/code/image_generation/ic_custom/vggt/data/test_data_soldier/images \
    --mask_path /group/40034/yaoweili/code/image_generation/ic_custom/vggt/data/navi/navi_v1.5/soldier_wood_showpiece/multiview-01-pixel_4a/masks \
    --model facebook/VGGT-1B \
    --max_points 1000000 \
    --conf_threshold 85.0


# if [ $? != 0 ]; then
#    echo "Fail! Exit with 1"
#    cd /group/40005/yaoweili/code/
#    python multi_occupy.py
#    exit 1
# else
#    echo "Success! Exit with 0"
#    cd /group/40005/yaoweili/code/
#    python multi_occupy.py
#    exit 0
# fi