
#/bin/bash
pip install tensorboardX
pip install soundfile
pip install editdistance
pip install opencv-python==4.6.0.66
cd /apdcephfs/private_qiushizhu/avmc/fairseq && pip install -e .

export WORLD_SIZE=$1
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
model_path=/apdcephfs/share_1316500/qiushizhu/avmc_results/finetune_avmc_misp_near_a_2_ngpu${ngpu}_updatefreq${updatefreq}

cd /apdcephfs/private_qiushizhu/avmc/avmc

python3 /apdcephfs/private_qiushizhu/avmc/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /apdcephfs/private_qiushizhu/avmc/avmc/config/ \
       --config-name base_100h \
       common.user_dir=`pwd` \
       checkpoint.save_dir=${model_path} \
       hydra.run.dir=${model_path} \
       task.data=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/audio_near_visual_middle \
       model.w2v_path=/apdcephfs/share_1316500/qiushizhu/avmc_results/pretrain_avmc_misp_a_1_ngpu4_updatefreq4/checkpoint_last.pt \
       distributed_training.distributed_world_size=${ngpu} \
       optimization.update_freq=[${updatefreq}] \
       optimization.max_update=35000 \
       optimization.lr=[1e-4]  \
       task.normalize=True \
       model.feature_grad_mult=1.0 \
       model.mask_prob=0.65 \
       lr_scheduler.phase_ratio=[0.6,0.2,0.2] \
       dataset.max_tokens=400000 \
       dataset.valid_subset="dev" \
       task.normalize=false \
       +model.modalities=["audio"]

# sleep 500h

