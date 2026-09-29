# # define the train datasets groups
train_groups = {
    "asv19_train": [0],
    "asv19_dev": [1],
    "asv5": [2, 3],
    "for": [4,5],
    "codecfake": [6,7],
    "add22": [8,9],
    "add23": [10,11],
    "dfadd": [12,13],}

## define the eval datasets groups
eval_groups = {
    "itw": [14],
}
# parquets: 
parquets = [
    "/netscratch/fkallel/universal_DF/parquets/ASV19/asvspoof2019_la_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ASV19/asvspoof2019_la_dev.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ASV5/asvspoof5_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ASV5/asvspoof5_val_full.parquet",
    "/netscratch/fkallel/universal_DF/parquets/FoR/for_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/FoR/for_dev.parquet",
    "/netscratch/fkallel/universal_DF/parquets/CodecFake/codecfake_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/CodecFake/codecfake_dev.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ADD22/add22_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ADD22/add22_val.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ADD23/add23_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/ADD23/add23_val.parquet",
    "/netscratch/fkallel/universal_DF/parquets/DFADD/dfadd_train.parquet",
    "/netscratch/fkallel/universal_DF/parquets/DFADD/dfadd_dev.parquet",
    "/netscratch/fkallel/universal_DF/parquets/InTheWild/itw_eval.parquet"
]
## directory where all features will be saved
feats_dir = "./feats/wav2vec2-xls-r-2b/"
## list of best performing layer features for all datasets
feats = [
    f"wav2vec2-xls-r-2b_Layer9_asv19_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv19_dev.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv5_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_asv5_dev.npy",
    f"wav2vec2-xls-r-2b_Layer9_for_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_for_dev.npy",
    f"wav2vec2-xls-r-2b_Layer9_codecfake_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_codecfake_dev.npy",
    f"wav2vec2-xls-r-2b_Layer9_add22_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_add22_val.npy",
    f"wav2vec2-xls-r-2b_Layer9_add23_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_add23_val.npy",
    f"wav2vec2-xls-r-2b_Layer9_dfadd_train.npy",
    f"wav2vec2-xls-r-2b_Layer9_dfadd_dev.npy",
    "wav2vec2-xls-r-2b_Layer9_itw.npy"
]
