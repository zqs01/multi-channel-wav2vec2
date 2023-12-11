#!/bin/bash
# unset WORLD_SIZE
# pip install numpy==1.19.5
# pip install scipy==1.5.4
# pip install opencv-python==4.6.0.66
# pip install sentencepiece
# cd /apdcephfs/private_qiushizhu/av_hubert/fairseq  && pip install -e .

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


i=$3
echo $(($distributed_rank+$i))
echo $i

cd /apdcephfs/private_qiushizhu/av_hubert/avhubert


CUDA_VISIBLE_DEVICES=$cuda_devices python3 /apdcephfs/private_qiushizhu/av_hubert/fairseq/fairseq_cli/hydra_train.py \
       --config-dir conf/pretrain --config-name base_vox_iter4.yaml \
       task.data=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/pretrain_hubert_iter3_tsv \
       task.label_dir=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/pretrain_hubert_iter3_tsv \
       hydra.run.dir=/apdcephfs/private_qiushizhu/av_results/pretrain_avhubert_iter4_${all_world_size}ngpu_${updatefreq}updatefreq \
       common.user_dir=`pwd` \
       distributed_training.distributed_world_size=${all_world_size} \
       distributed_training.distributed_rank=$(($distributed_rank + $i)) \
       distributed_training.device_id=$i \
       optimization.update_freq=[${updatefreq}] \
       dataset.max_tokens=1000  \
       model.label_rate=25  \
       dataset.train_subset="train" \
       dataset.valid_subset="valid"

