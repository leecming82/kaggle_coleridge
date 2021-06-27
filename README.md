## Introduction
Pytorch code + Kaggle notebook used by the 2nd place winner for the [2021 Coleridge Initiative Kaggle competition](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data)
My solution is detailed in this [Kaggle forum post](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248296).

### Recipe: 
1. Run the [training code](label_classifier.py) in a Docker container with configuration specified by the linked [Dockerfile](Dockerfile) using the training corpus of 5K samples [here](roberta-annotate-abbr.csv) to generate the prerequisite HuggingFace Transformer model/tokenizer
2. Run the [inference notebook](2nd-place-coleridge-inference-code.ipynb) in a Kaggle GPU instance linking to the HuggingFace model trained from Step 1.
Note: I did not fix a seed for the training code and have observed variances in the resulting model/performance on the LB so you can use the actual model/tokenizer I used linked [here](https://www.kaggle.com/leecming/robertalabelclassifierrawipcc)

### Misc:
1. Training took <5 minutes (5 epochs, Batch-size 32, max-sequence-length:128) on local hardware: Threadripper 3960X + 64GB RAM + Nvidia A6000 GPU
2. Test-set inference took <10 minutes on a Kaggle GPU instance
