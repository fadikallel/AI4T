# Unmasking real-world audio deepfakes: A data-centric approach

This is the official repository for the paper:

> David Combei, Adriana Stan, Dan Oneata, Nicolas Müller, Horia Cucu,  
> "Unmasking real-world audio deepfakes: A data-centric approach",  
> accepted at Interspeech 2025, Rotterdam, Netherlands.

---

## Citation

```bibtex
@inproceedings{combei_interspeech25,
  author={Combei, David and Stan, Adriana and Oneata, Dan and Müller, Nicolas and Cucu, Horia},
  booktitle={Proc. of Interspeech},
  title={{Unmasking real-world audio deepfakes: A data-centric approach}},
  year={2025}
}
```
## Disclaimer

The dataset referenced in this paper is shared via publicly available links. We do not own the copyrights to redistribute the original samples directly.
The links included in this repository were valid as of February 2025. Any content removed from social media platforms before this date is not included.
Please note that some links may become unavailable over time due to platform policies regarding AI-generated or manipulated content or just by owners removing the content from the platform.

## SETUP

1. Datasets
   Download the datasets from:
     - [ASV19](https://datashare.ed.ac.uk/handle/10283/3336)
     - [ASV21](https://www.asvspoof.org/index2021.html)
     - [ASV5](https://zenodo.org/records/14498691)
     - [MLAAD](https://deepfake-total.com/mlaad)
     - [TIMIT](https://zenodo.org/records/6560159)
     - [ODSS](https://zenodo.org/records/8370668)
     - [FoR](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data)
     - [ITW](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa)

   AI4T links are available in links directory. NOTE: we segmented each audio file into 10s segments in our experiments.

2. Feature Extraction
   In our implementation we used the pretrained and frozen SSL model [wav2vec2-xls-r-2b](https://huggingface.co/facebook/wav2vec2-xls-r-2b).
   You can extract the averaged pool representation from each layer using the following script:
   ```
   wav2vec2-xls-r-2b_all-layers_extractor.py
   ```
   You can also extract the RawBoost and codec augmented features from layer 9 using these scripts:
   ```
   rawboost_wav2vec-xls-r-2b_extractor.py
   ```
   and
   ```
   xls-r-2b_codec-augm_extractor.py
   ```

## EXPERIMENTS

   1. Baseline deepfake detector: We extracted the features from all datasets and all layers using the script above and found out that layer 9 yield the best result. You can check the extended results for this analysis [here](https://docs.google.com/spreadsheets/d/1B3PGSqAgrYepOi66SEj0wHZy84aXPwot4GqWbehtwvM/edit?usp=sharing). We also augmentated the ASV19 train+dev for the baseline experiments.
   2. Dataset mixing: We train the logistic regression classifier and evaluate each combination on ITW and AI4T datasets. Run:
      ```
      train_logReg_iterative.py
      ```
      script for this experiment. You can check the extended results of Table 3 [here](https://docs.google.com/spreadsheets/d/1B3PGSqAgrYepOi66SEj0wHZy84aXPwot4GqWbehtwvM/edit?gid=0#gid=0).
   3. Data Pruning: After finding the best dataset mixing combination, we move to data pruning strategies for ALL data and best dataset combination:
      For random selection run:
      ```
      random_selection.py
      ```
      For cluster based pruning run:
      ```
      cluster_based_pruning.py
      ```
      For margin pruning run:
      ```
      margin_pruning.py
      ```
      NOTE: for random selection, the seed is not set. We reported the average of 3 random seeds. Results might not look the same in your case.
   4. Post pruning data augmentation:
      After saving the file names of the selected data during pruning, you can augment only these samples if you did not augment all the data in previous steps. After building the final training set using the selected and augmented data, you can train the final      classifier that yields the best results, as in Table4 using
      ```
      run_logReg_deepfake_detection_WAugm_margin_pruning.py
      ```
      script. Please note that this process has some randomness due to the augmentation process, so results might are likely to be slightly different.

## ACKNOWLEDGEMENTS

This work was co-funded by EU Horizon project AI4TRUST (No. 101070190), and by the Romanian
Ministry of Research, Innovation and Digitization project DLT-
AI SECSPP (id: PN-IV-P6-6.3-SOL-2024-2-0312), and by the
Free State of Bavaria under the DSgenAI project (Grant Nr.:
RMF-SG20-3410-2-18-4)
      

