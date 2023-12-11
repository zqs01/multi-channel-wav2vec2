#!/bin/bash

decode_path=/apdcephfs/share_1316500/qiushizhu/av_results/finetune_misp_visual_far_iter5_ctc_1_8ngpu_1updatefreq
finetuned_model=checkpoint_best.pt
beam=1
decode_dataset=dev

python3 -B infer_ctc.py --config-dir conf/ --config-name ctc_decode.yaml \
  dataset.gen_subset=${decode_dataset} \
  dataset.max_tokens=1000 \
  common_eval.path=${decode_path}/checkpoints/${finetuned_model} \
  common_eval.results_path=${decode_path}/${finetuned_model}_${decode_dataset}_beam${beam} \
  override.modalities=['video'] \
  common.user_dir=`pwd` \
  override.data=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/audio_far_visual_far \
  override.label_dir=/apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/audio_far_visual_far \
  generation.beam=${beam} \
  # generation.max_len_a=1.0 \
  # generation.max_len_b=0 \
  # generation.lenpen=-1.0 \

# dataset.max_tokens=1000 \
# sclite -r new_ref.txt -h new_hyp.txt -i rm -o all stdout > cer.log