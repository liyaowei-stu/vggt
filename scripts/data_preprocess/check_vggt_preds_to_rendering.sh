
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

export PYTHONPATH=.:$PYTHONPATH

python ic_custom_data_prepare/check_vggt_preds_to_rendering.py \
    --predictions_path output/test_data_soldier_predictions_20250704_205059_all.npz \
    --max_points 450000 \
    --conf_threshold 50.0 \
    --radius 0.01 \
    --points_per_pixel 100 \
    --bin_size 0


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