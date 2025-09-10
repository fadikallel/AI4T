## define the train datasets groups
train_groups = {
    "asv19_train": [1],
    "asv19_dev": [2],
    "asv19_eval": [3],
    "asv19_all" : [1,2,3],
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
