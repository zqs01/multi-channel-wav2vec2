
# /bin/bash
# pip install numpy==1.19.5
# pip install scipy==1.5.4
# pip install opencv-python==4.6.0.66
# pip install sentencepiece
# pip install editdistance
# cd /apdcephfs/private_qiushizhu/av_hubert/fairseq2  && pip install -e .

export WORLD_SIZE=$1
# export HOST_NUM=$3
export HOST_GPU_NUM=$1
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export RANK=$INDEX
export MASTER_ADDR=$CHIEF_IP
export MASTER_PORT=10000
export HYDRA_FULL_ERROR=1

ngpu=$1
updatefreq=$2


cd /apdcephfs/private_qiushizhu/av_hubert/avhubert_ctc

# -m torch.distributed.launch --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM  --node_rank=$INDEX --master_addr=$CHIEF_IP --master_port=$MASTER_PORT
python3  /apdcephfs/private_qiushizhu/av_hubert/fairseq2/fairseq_cli/hydra_train.py \
       --config-dir /apdcephfs/private_qiushizhu/av_hubert/avhubert_ctc/conf/av-finetune \
       --config-name base_misp_ctc.yaml \
       task.data=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/audio_far_visual_middle \
       task.label_dir=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/audio_far_visual_middle \
       model.w2v_path=/apdcephfs/private_qiushizhu/av_results/pretrain_avhubert_iter3_32ngpu_1updatefreq/checkpoints/checkpoint_last.pt \
       hydra.run.dir=/apdcephfs/private_qiushizhu/av_results/finetune_misp_audio_far_visual_middle_self_iter3_ctc_1_${ngpu}ngpu_${updatefreq}updatefreq \
       common.user_dir=`pwd` \
       distributed_training.distributed_world_size=${ngpu} \
       distributed_training.nprocs_per_node=8 \
       distributed_training.ddp_backend="legacy_ddp" \
       optimization.update_freq=[${updatefreq}] \
       dataset.max_tokens=2000  \
       task.modalities=["audio","video"] \
       model.freeze_finetune_updates=0 \


# sleep 100h



# sleep 100h