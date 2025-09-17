DATASETS = {
    "for": {
        "indir": "/ds-slt/audio/for-norm/for-norm/",
        "metadata": "./processed_metadata/for_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_for.npy"
    },
    "timit": {
        "indir": "/ds-slt/audio/TIMIT/TIMIT-TTS/CLEAN",
        "metadata": "./processed_metadata/timit_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_timit.npy"
    },
    "asv19_eval": {
        "indir": "/ds-slt/audio/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac",
        "metadata": "./processed_metadata/asv19_eval_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_asv19_eval.npy",
        "flac": True
    },
    "asv19_train": {
        "indir": "/ds-slt/audio/ASVspoof2019/LA/ASVspoof2019_LA_train/flac",
        "metadata": "./processed_metadata/asv19_train_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_asv19_train.npy",
        "flac": True
    },
    "asv19_dev": {
        "indir": "/ds-slt/audio/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac",
        "metadata": "./processed_metadata/asv19_dev_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_asv19_dev.npy",
        "flac": True
    },
    "asv21": {
        "indir": "/ds-slt/audio/ASVspoof2021/DF/ASVspoof2021_DF_eval/flac",
        "metadata": "./processed_metadata/asv21_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_asv21.npy",
        "flac": True
    },
    "asv5_train": {
        "indir": "/ds-slt/audio/ASVSpoof2024/flac_T/",
        "metadata": "./processed_metadata/asv5_train_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_asv5_train.npy",
        "flac": True
    },
    "asv5_dev": {
        "indir": "/ds-slt/audio/ASVSpoof2024/flac_D/",
        "metadata": "./processed_metadata/asv5_dev_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_asv5_dev.npy",
        "flac": True
    },
    "mailabs": {
        "indir": "/ds-slt/audio/MLAAD/v5/Processed_sr_16000_PCM16/real/",
        "metadata": "./processed_metadata/mailabs_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_m-ailabs.npy"
    },
    "mlaad_v5": {
        "indir": "/ds-slt/audio/MLAAD/v5/Processed_sr_16000_PCM16/",
        "metadata": "./processed_metadata/mlaad_v5_xls_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_mlaad_v5.npy"
    },
    "odss": {
        "indir": "/ds-slt/audio/ODSS/",
        "metadata": "./processed_metadata/odss_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_odss.npy"
    },
    "itw": {
        "indir": "/ds-slt/audio/release_in_the_wild/",
        "metadata": "./processed_metadata/itw_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_itw.npy"
    },
    "ai4trust": {
        "indir": "/",
        "metadata": "./processed_metadata/ai4trust_segm_systems.csv",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_ai4trust.npy"
    }, 
    "ADD22_track1":{
        "indir": "/ds-slt/audio/yelkheir/ADD22/track1test2/track1test/",
        "metadata": "./processed_metadata/ADD22_track1.txt",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_ADD22_track1.npy",
        "wav": True
    },
    "ADD22_track3":{
        "indir": "/ds-slt/audio/yelkheir/ADD22/track32test/",
        "metadata": "./processed_metadata/ADD22_track3.txt",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_ADD22_track3.npy",
        "wav": True
    },
    "ADD2023_round1":{
        "indir": "/ds/audio/ADD23_track_1.2/Track1.2/testR1/wav/",
        "metadata": "./processed_metadata/ADD2023_round1.txt",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_ADD2023_round1.npy",
        "wav": True
    },
    "ADD2023_round2":{
        "indir": "/ds/audio/ADD23_track_1.2/Track1.2/testR2/wav/",
        "metadata": "./processed_metadata/ADD2023_round2.txt",
        "outfile": "./feats/wav2vec2-xls-r-2b/wav2vec2-xls-r-2b_Layer9_ADD2023_round2.npy",
        "wav": True
    },
}

# # define the train datasets groups
train_groups = {
    "asv19_train": [1],
    "asv19_dev": [2],
    "asv19_eval": [3],
    "asv21": [4],
    "asv5": [5, 6],
    "for": [7],
    "mlaad": [9, 10],
    "odss": [11],
    "timit": [12],
}
## define the eval datasets groups
eval_groups = {
    "itw": [8],
    "ai4trust": [0],
}
## directory where all metadatas will be
meta_dir = "./processed_metadata/"
## metadata file names, the order must match the indexes from the train and eval groups !!
metadata = [
    "ai4trust_segm_systems.csv",
    "asv19_train_systems.csv",
    "asv19_dev_systems.csv",
    "asv19_eval_systems.csv",
    "asv21_systems.csv",
    "asv5_train_systems.csv",
    "asv5_dev_systems.csv",
    "for_systems.csv",
    "itw_systems.csv",
    "mailabs_systems.csv",
    "mlaad_v5_xls_systems.csv",
    "odss_systems.csv",
    "timit_systems.csv",
]
## directory where all features will be saved
feats_dir = "./feats/wav2vec2-xls-r-2b/"
## list of best performing layer features for all datasets
feats = [
    f"wav2vec2-xls-r-2b_Layer9_ai4trust.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv19_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv19_dev.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv19_eval.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv21.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv5_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv5_dev.npy",
    f"wav2vec2-xls-r-2b_Layer9_for.npy",
    f"wav2vec2-xls-r-2b_Layer9_itw.npy",
    f"wav2vec2-xls-r-2b_Layer9_m-ailabs.npy",
    f"wav2vec2-xls-r-2b_Layer9_mlaad_v5.npy",
    f"wav2vec2-xls-r-2b_Layer9_odss.npy",
    f"wav2vec2-xls-r-2b_Layer9_timit.npy",
]
## the augmented features for asv19 train+dev to reproduce the baseline deepfake detector
asv19_augm = [
    "wav2vec2-xls-r-2b_asv19_train_augm_rb_Layer9.npy",
    "wav2vec2-xls-r-2b_asv19_dev_augm_rb_Layer9.npy",
    "wav2vec2-xls-r-2b_asv19_train_augm_codecs_Layer9.npy",
    "wav2vec2-xls-r-2b_asv19_dev_augm_codecs_Layer9.npy",
]
## modify with what margin pruning will save
metadata_augm = [
    "metadata_marginPruned_XLS_fromALL_margin_both_135.txt",
    "metadata_marginPruned_XLS_fromALL_margin_both_135.txt",
    "itw_systems.csv",
    "ai4trust_segm_systems.csv",
]
## the augmented features that the margin pruning selected above (the ones that are saved in the metadata above)
feats_augm = [
    "wav2vec2-xls-r-2b_augm_codecs_Layer9.npy",
    "wav2vec2-xls-r-2b_augm_rb_Layer9.npy",
    "wav2vec2-xls-r-2b_Layer9_itw.npy",
    "wav2vec2-xls-r-2b_Layer9_ai4trust.npy",
]
