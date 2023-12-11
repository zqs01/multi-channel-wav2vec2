nshrad=$1
rank=$2
python3 dump_hubert_feature.py /apdcephfs/share_1316500/qiushizhu/misp_related/misp_tsv_new/visual_audio/pretrain_hubert_tsv valid /apdcephfs/private_qiushizhu/av_results/pretrain_fbank_iter1_16ngpu_1updatefreq/checkpoints/checkpoint_last.pt 9  ${nshrad} ${rank} /apdcephfs/share_1316500/qiushizhu/av_results/dump_features/pretrain_hubert_iter1  --user_dir `pwd`/../
# python3 dump_hubert_feature.py /data/misp_related/misp_tsv_new/visual_audio/audio_near_visual_middle train /apdcephfs/private_qiushizhu/av_results/pretrain_fbank_iter1_16ngpu_1updatefreq/checkpoints/checkpoint105.pt 9  ${nshrad} ${rank} /data/misp_related/avhubert_feature/pretrain_hubert_iter1  --user_dir `pwd`/../
