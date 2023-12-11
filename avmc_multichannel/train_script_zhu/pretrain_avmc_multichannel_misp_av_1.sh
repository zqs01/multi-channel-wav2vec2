
#/bin/bash
# pip install tensorboardX
# pip install soundfile
# pip install editdistance
# pip install opencv-python==4.6.0.66
# cd /apdcephfs/private_qiushizhu/avmc/fairseq && pip install -e .

export WORLD_SIZE=$1
export HOST_GPU_NUM=8
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export RANK=$INDEX
export MASTER_ADDR=$CHIEF_IP
export MASTER_PORT=10000
export HYDRA_FULL_ERROR=1


export cuda_devices=0,1,2,3,4,5,6,7
world_size=`echo $cuda_devices | sed -e 's/,/\n/g' | wc -l`

nnodes=4
all_world_size=`echo "${world_size}*$nnodes"|bc`
export WORLD_SIZE=${all_world_size}
NNODES=$nnodes

distributed_rank=`echo "${INDEX}*$world_size"|bc`



updatefreq=$2
model_path=/apdcephfs/private_qiushizhu/avmc_results/pretrain_avmc_multichannel_misp_av_1_ngpu${all_world_size}_updatefreq${updatefreq}

cd /apdcephfs/private_qiushizhu/avmc/avmc_multichannel

i=$3
echo $(($distributed_rank+$i))
echo $i

CUDA_VISIBLE_DEVICES=$cuda_devices python3 /apdcephfs/private_qiushizhu/avmc/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /apdcephfs/private_qiushizhu/avmc/avmc_multichannel/config/ \
       --config-name avmc_base_misp \
       common.user_dir=`pwd` \
       checkpoint.save_dir=${model_path} \
       hydra.run.dir=${model_path} \
       task.data=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/multichannel_tsv/multichannel_pretrain_tsv \
       distributed_training.distributed_world_size=${all_world_size} \
       distributed_training.distributed_rank=$(($distributed_rank + $i)) \
       distributed_training.device_id=$i \
       optimization.update_freq=[${updatefreq}] \
       optimization.lr=[5e-4]  \
       dataset.train_subset="dev" \
       dataset.valid_subset="dev" \
       dataset.max_tokens=1000000 \
       task.min_sample_size=32000 \
       task.max_sample_size=280000 \
       dataset.num_workers=8 \
       +task.enable_padding=True \
       +model.modalities=["audio","video"] \
       +model.modality_dropout=0.0 \
       +model.audio_dropout=0.0
       # +model.conv_feature_layers="\[(512, 10, 5) + (512, 3, 2) * 4 + (512,2,2) + (512,2,2)\]" \

# sleep 500h

