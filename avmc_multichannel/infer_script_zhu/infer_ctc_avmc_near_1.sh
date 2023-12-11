#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/apdcephfs/private_qiushizhu/avmc/fairseq3


subset=evalc4
path=/apdcephfs/share_1316500/qiushizhu/avmc_results/finetune_avmc_multichannel_misp_audio_middle_1_ngpu8_updatefreq4
model=checkpoint_best.pt

mkdir -p ${path}/${subset}_${model}
cd  /apdcephfs/private_qiushizhu/avmc/avmc_multichannel

python3 /apdcephfs/private_qiushizhu/avmc/avmc_multichannel/new/infer.py --config-dir /apdcephfs/private_qiushizhu/avmc/avmc_multichannel/new/conf \
       --config-name infer \
       common.user_dir=`pwd` \
       task=audio_finetuning \
       task.data=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/multichannel_tsv/audio_middle_visual_far \
       task.labels=ltr \
       decoding.type=viterbi \
       dataset.gen_subset=${subset} \
       common_eval.path=${path}/${model} \
       distributed_training.distributed_world_size=1 \
       common_eval.results_path=${path}/${subset}_${model} \
       decoding.results_path=${path}/${subset}_${model} \



# sclite -r ref.word.txt -h hypo.word.txt -i rm -o all stdout > wer.log