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

1. Datasets and dependencies

   Download the datasets from their original repositories:
     - [ASV19](https://datashare.ed.ac.uk/handle/10283/3336)
     - [ASV21](https://www.asvspoof.org/index2021.html)
     - [ASV5](https://zenodo.org/records/14498691)
     - [MLAAD](https://deepfake-total.com/mlaad)
     - [TIMIT](https://zenodo.org/records/6560159)
     - [ODSS](https://zenodo.org/records/8370668)
     - [FoR](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data)
     - [ITW](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa)

   Links for the AI4T data are available in `AI4T dataset` directory. NOTE: Each audio file was segmented into 10 seconds chunks for our experiments.

   For the dependencies, run:

   ```
   pip install -r requirements.txt
   ```

3. Feature Extraction

   In our implementation we used the pretrained and frozen SSL model [wav2vec2-xls-r-2b](https://huggingface.co/facebook/wav2vec2-xls-r-2b).
   You can extract the averaged pool representation from each layer using the following script:
   Please make sure to change the paths to the wav files and use the metadata files from `processed_metadata` directory
    ```
    wav2vec2-xls-r-2b_all-layers_extractor.py
    ```
    The result of this script will be 49 `.npy` files, one from each layer, layer 0 being the output for convolution block and 1-48 are the transformers. Each sample will be saved in `1x1920` shape. In our experiments, the 9th transformer layer was performing best (you can check the extended results [here](https://github.com/davidcombei/AI4T/blob/main/Layers_eval.pdf).
   For data augmentation, we used the first 2 algorithms together in series described by Tak et al. in [RawBoost](https://arxiv.org/abs/2111.04433) paper and the codec, where we use AAC and Opus with a chance of 50/50.
    For the RawBoost and codec augmented features:
   ```
   rawboost_wav2vec-xls-r-2b_extractor.py
   ```
   and
   ```
   xls-r-2b_codec-augm_extractor.py
   ```
These scripts will be extracting the features from layer 9 only. We truncated the SSL model from 48 transformer layers to only 9, speeding up the inference time.

3. Config
   
   In ``` config.py ``` file you have all the directory and file paths necessary to run the experiments. Please modify it accordingly to your paths and names. 
## EXPERIMENTS

   ### 1. Baseline deepfake detector: 
   
   For the baseline deepfake detector, the training data used is ASV19 train+dev partitions and evaluate on all the datasets. We compare the last hidden state (L) and layer 9(B:9) with non-augmented features, and all 3 combinations of augmented features (+RB), (+C) and (+RB+C) using layer 9  as shown in the table below:
   
   ![image](https://github.com/user-attachments/assets/948ea6cd-de00-412d-ac3c-80a7b95f0d13)

   We can see that data augmentation improves the results for scientific datasets, in contrast with the behaviour while evaluating on real-world datasets.
   ### 2. Dataset mixing
   
   Having 7 scientific datasets, we try dataset mixing to find to most relevant datasets for generalization on real-world datasets. For this, we train the Logistic Regression classifier using all datasets combinations (127) using the non-augmented features. For this experiment, run:
      ```
      train_logReg_iterative.py
      ```
  This script will save in a log file named `results.txt` all the combinations and EER for both ITW and AI4T. 
  You can check the extended results for this experiment [here](https://github.com/davidcombei/AI4T/blob/main/Dataset%20mixing.pdf).
   
   ### 3. Data Pruning
   
  After finding the best dataset mixing combination, we move to data pruning strategies for all 7 datasets combined (ALL) and best dataset combination (FoR+ODSS+MLAAD):

  For random selection run:
      ```
      random_selection.py
      ```
  This script will randomly select samples from all seven training sets combined, using selection percentages ranging from 10% to 90%. For each percentage, the sampling is repeated 3 times with different random seeds.
  NOTE: We reported the average results of 3 random seeds. Results might not look the same in your case.
  
  For cluster based pruning run:
      ```
      cluster_based_pruning.py
      ```
  This approach computes a centroid using `NearestCentroid` and ranks the samples based on Euclidean distance. We apply this strategy independently on both real and fake samples from each of the N datasets, and obtain 2 * N sets of samples selected, which we ensemble in the final pruned dataset. It selects either the `cluster-closest` or `cluster-furthest` based on the `order` parameter.
  
  For margin pruning run:
      ```
      margin_pruning.py
      ```
  In this script, we use the logistic regression's margins over the samples. We remove the closest or the closest and furthest samples with respect to the decision hyperplane, no matter the dataset. 
  For removing the closest samples use set the `strategy` parameter to `noisy` and in order to remove both closest and furthest, set the parameter to `both`.
  Below you can see the results for the data selection experiments for ALL datasets and FoR+ODSS+MLAAD:

  
  ![image](https://github.com/user-attachments/assets/34a16acb-cf50-4d01-9dd5-54a8a140bfe8)


  
  ### 4. Post pruning data augmentation:
  The margin pruning will save a `.txt` file with the selected samples. You can use it as a metadata to augment the data for this step if you did not augment it before. When you have the selected and augmented features, run:
      ```
      run_logReg_deepfake_detection_WAugm_margin_pruning.py
      ```
  This script will train the logistic regression classifier from scratch using the most relevant data and their augmentations, resulting in the best version in our paper, as in the table below.
  NOTE: this process has some randomness due to the augmentation process, so results are likely to be slightly different. Below you can see the results with the data augmentation post pruning:

  ![image](https://github.com/user-attachments/assets/1a59428f-1257-46c1-8556-d4ad75e51f87)


## ACKNOWLEDGEMENTS

This work was co-funded by EU Horizon project AI4TRUST (No. 101070190), and by the Romanian
Ministry of Research, Innovation and Digitization project DLT-
AI SECSPP (id: PN-IV-P6-6.3-SOL-2024-2-0312), and by the
Free State of Bavaria under the DSgenAI project (Grant Nr.:
RMF-SG20-3410-2-18-4)
      

