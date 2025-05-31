meta_dir = './processed_metadata/'
metadata = [
          "asv19_train_systems.csv",
          "asv19_dev_systems.csv",
          "asv19_eval_systems.csv",
          "for_systems.csv",
          "asv21_systems.csv",
          "asv5_train_systems.csv",
          "asv5_dev_systems.csv",
          "odss_systems.csv",
          "timit_systems.csv",
          "itw_systems.csv",
          "ai4trust_segm_systems.csv",
          "mlaad_v5_xls_systems.csv",
          "mailabs_systems.csv",
]
feats_dir = "./feats/wav2vec2-xls-r-2b/"
feats = [
        f"wav2vec2-xls-r-2b_Layer10_asv19_train.npy",
        f"wav2vec2-xls-r-2b_Layer10_asv19_dev.npy",
        f"wav2vec2-xls-r-2b_Layer10_asv19_eval.npy",
        f"wav2vec2-xls-r-2b_Layer10_for.npy",
        f"wav2vec2-xls-r-2b_Layer10_asv21.npy",
        f"wav2vec2-xls-r-2b_Layer10_asv5_train.npy",
        f"wav2vec2-xls-r-2b_Layer10_asv5_dev.npy",
        f"wav2vec2-xls-r-2b_Layer10_odss_PITCH_CROP4.npy",
        f"wav2vec2-xls-r-2b_Layer10_timit_tts_clean_PITCH_CROP4.npy",
        f"wav2vec2-xls-r-2b_Layer10_itw.npy",
        f"wav2vec2-xls-r-2b_Layer10_ai4trust.npy",
        f"wav2vec2-xls-r-2b_Layer10_mlaad_v5.npy",
        f"wav2vec2-xls-r-2b_Layer10_m-ailabs_PITCH_CROP4.npy",
    ]
metadata_augm = [
            "metadata_marginPruned_XLS_fromALL_margin_both_135.txt",
            "metadata_marginPruned_XLS_fromALL_margin_both_135.txt",
            "itw_systems.csv",
            "ai4trust_segm_systems.csv",
]
feats_augm = [f"sel_from_ALL_margin_both_135_codecs_wav2vec2_xls-r-2b_embeddings/wav2vec2-xls-r-2b_augm_codecs_Layer9.npy",
         "sel_from_ALL_margin_both_135_rawboost_wav2vec2_xls-r-2b_embeddings/wav2vec2-xls-r-2b_Layer9.npy",
         f"wav2vec2-xls-r-2b_Layer9_itw.npy",
         f"wav2vec2-xls-r-2b_Layer9_ai4trust.npy"]
