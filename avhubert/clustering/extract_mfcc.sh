nshrad=$1
rank=$2
# python3 dump_mfcc_feature.py /data/misp_related/misp_tsv_new/visual_audio/audio_near_visual_middle dev ${nshrad} ${rank} /data/misp_related/avhubert_feature
python3 dump_mfcc_feature.py /data/misp_related/misp_tsv_new/visual_audio/pretrain_hubert_tsv train ${nshrad} ${rank} /data/misp_related/avhubert_feature/pretrain_hubert