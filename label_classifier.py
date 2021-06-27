"""
Baseline PyTorch binary classifier using a pretrained HuggingFace Transformer model
"""
import os
import time
from random import shuffle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from apex import amp
from tqdm import trange
from apex.optimizers import FusedAdam

USE_AMP = True  # Use mixed-precision training
PRETRAINED_MODEL = 'roberta-base'
TRAIN_CSV_PATH = 'data/roberta-annotate-abbr.csv'
SAVE_MODEL = True
MODEL_PATH = 'models/label-classification/roberta-base-rawabbr'
MAX_CORES = 24  # limit MP calls to use this # cores at most; for tokenizing
MAX_SEQ_LEN = 128  # max sequence length for input strings: gets padded/truncated
# Num. epochs to train against (if validation data exists, the model will switch to training against the validation
# data in the 2nd half of epochs
NUM_EPOCHS = 5
# Gradient Accumulation: updates every ACCUM_FOR steps so that effective BS = BATCH_SIZE * ACCUM_FOR
BATCH_SIZE = 32
ACCUM_FOR = 1
LR = 1e-5  # Learning rate - constant value


def train(model, train_tuple, opt, curr_epoch):
    """
    Trains against the train_tuple features for a single epoch
    """
    # Shuffle train indices for current epoch, batching
    all_strings, all_labels = train_tuple
    train_indices = list(range(len(all_labels)))

    shuffle(train_indices)
    train_strings = all_strings[train_indices]
    train_labels = all_labels[train_indices]

    model.train()
    iter = 0
    running_total_loss = 0  # Display running average of loss across epoch
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            iter += 1
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

            current_batch = list(train_strings[batch_idx_start:batch_idx_end])
            batch_features = tokenizer(current_batch,
                                       truncation=True,
                                       max_length=64,
                                       padding='max_length',
                                       add_special_tokens=True,
                                       return_tensors='pt')
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).long().cuda()
            batch_features = {k: v.cuda() for k, v in batch_features.items()}

            model_outputs = model(**batch_features,
                                  labels=batch_labels,
                                  return_dict=True)
            loss = model_outputs['loss']
            loss = loss / ACCUM_FOR  # Normalize if we're doing GA

            if USE_AMP:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            running_total_loss += loss.detach().cpu().numpy()
            t.set_postfix(loss=running_total_loss / iter)

            if iter % ACCUM_FOR == 0:
                opt.step()
                opt.zero_grad()


def main_driver(train_tuple):
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   num_labels=2)
    classifier = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    opt = FusedAdam(classifier.parameters(), lr=LR)

    if USE_AMP:
        classifier, opt = amp.initialize(classifier, opt, opt_level='O2', verbosity=0)

    for curr_epoch in range(NUM_EPOCHS):
        train(classifier, train_tuple, opt, curr_epoch)

        if SAVE_MODEL:
            curr_epoch_dir = os.path.join(MODEL_PATH, str(curr_epoch))
            if not os.path.exists(curr_epoch_dir):
                os.makedirs(curr_epoch_dir)
            classifier.save_pretrained(curr_epoch_dir)
            tokenizer.save_pretrained(curr_epoch_dir)


if __name__ == '__main__':
    start_time = time.time()

    # Load train, validation, and pseudo-label data
    input_df = pd.read_csv(TRAIN_CSV_PATH)
    train_strings = input_df['long'].values
    train_labels = input_df['is_dataset'].values

    # use MP to batch encode the raw feature strings into Bert token IDs
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    main_driver([train_strings, train_labels])

    print('Elapsed time: {}'.format(time.time() - start_time))
