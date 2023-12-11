nshrad=$1
rank=$2
python3 dump_hubert_feature.py /data/misp_related/misp_tsv_new/visual_audio/pretrain_hubert_tsv dev /apdcephfs/private_qiushizhu/av_results/pretrain_avhubert_iter4_32ngpu_1updatefreq/checkpoints/checkpoint_last.pt 12  ${nshrad} ${rank} /data/misp_related/avhubert_feature/pretrain_hubert_iter4  --user_dir `pwd`/../
# python3 dump_hubert_feature.py /data/misp_related/misp_tsv_new/visual_audio/pretrain_hubert_tsv train /apdcephfs/private_qiushizhu/av_results/pretrain_avhubert_iter4_32ngpu_1updatefreq/checkpoints/checkpoint_last.pt 12  ${nshrad} ${rank} /data/misp_related/avhubert_feature/pretrain_hubert_iter4  --user_dir `pwd`/../
