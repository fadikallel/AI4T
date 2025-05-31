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
   
   

