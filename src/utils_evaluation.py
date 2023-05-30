import numpy as np
import pandas as pd
import json
from extract_explanations import read_blimp_dataset, read_sva_dataset, read_ioi_dataset


def read_subset(subset):
    # Read datasets from csv as DataFrames
    if 'sva' in subset:
        # Expected 'sva_[number]'
        df = read_sva_dataset()
        num_attractors = int(subset[-1])
        df = df[df['num_attractors']==num_attractors].reset_index(drop=True)
        num_examples = 200
    elif subset == 'ioi':
        df = read_ioi_dataset()
        num_examples = len(df)
    else:
        df = read_blimp_dataset(subset)
        num_examples = len(df)
    df['target'].replace(np.nan, 'NaN', inplace=True)
    if 'target_phrase' in df.columns:
        df['target_phrase'].replace(np.nan, 'NaN', inplace=True)
    return df, num_examples

def read_attribution_scores(name_path, dataset, subset, method):
    # Read json files with attribtuion scores (obtained with extract_explanations.py)
    if 'sva' in subset:
        num_attractors = subset[-1]
        f = open(f'./results/{subset}/{subset}_{name_path}_{method}_{str(num_attractors)}.json')
    else:
        print(f'./results/{dataset}/{subset}_{name_path}_{method}.json')
        f = open(f'./results/{dataset}/{subset}_{name_path}_{method}.json')
    explanations_dict = json.load(f)
    return explanations_dict
    
def read_sentence_target_foil(df, subset, idx):
    # Get sentence, target word and foil word
    if 'sva' in subset:
        text = df['one_prefix_prefix'][idx][:df['one_prefix_prefix'][idx].index('***mask***')-1]
    else:
        text = df['one_prefix_prefix'][idx]
    # Acceptable prediction
    target = df['one_prefix_word_good'][idx]
    # Unacceptable prediction (foil)
    foil = df['one_prefix_word_bad'][idx]

    return text, target, foil

def tokenize_target_foil(name_path, tokenizer, target, foil):
    # Tokenize target and foil
    if 'facebook-opt' in name_path:
        # OPT tokenizer adds a BOS token at pos 0
        CORRECT_ID = tokenizer(" "+ target)['input_ids'][1]
        FOIL_ID = tokenizer(" "+ foil)['input_ids'][1]
        min_length = 2
    else:
        CORRECT_ID = tokenizer(" "+ target)['input_ids'][0]
        FOIL_ID = tokenizer(" "+ foil)['input_ids'][0]
        min_length = 1
    return CORRECT_ID, FOIL_ID, min_length
    

def get_ground_truth_tokens(subset, text, target, tokenizer, df, idx):
    # Consider target_phrase or target as evidence of linguistic phenomena
    if subset is not 'distractor_agreement_relational_noun' and subset is not 'irregular_plural_subject_verb_agreement_1' and subset is not 'regular_plural_subject_verb_agreement_1' and 'sva' not in subset:
        if 'target_phrase' in df.columns:
            gt_word = df['target_phrase'][idx]
        else:
            gt_word = df['target'][idx]
    else:
        gt_word = df['target'][idx]

    if gt_word == text + ' ' + target:
        return None, None
    
    if gt_word == 'NaN':
        # Target word not found (NA)
        return None, None

    gt_token_first_word_tokens = tokenizer(gt_word)['input_ids']
    # pt_batch = tokenizer(text, return_tensors="pt").to(device)
    # tokenized_text = tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])
    # list_text_tokens = list(tokenizer.convert_tokens_to_ids(tokenized_text))
    list_text_tokens = tokenized_text = tokenizer(text)['input_ids']

    if 'facebook/opt' in tokenizer.name_or_path:
        gt_token_not_first_word_tokens = tokenizer(" "+ gt_word)['input_ids'][1:] # skip </s>
    else:
        gt_token_not_first_word_tokens = tokenizer(" "+ gt_word)['input_ids']
        
    # Obtain the position of the ground truth (evidence) tokens (gt_token_pos)
    # e.g. [3,4,5] means tokens at position 3, 4, and 5 of the prefix
    # are the evidence of linguistic phenomena
    gt_token_pos = []
    if gt_token_first_word_tokens == list_text_tokens[:(len(gt_token_first_word_tokens))]:
        for token in gt_token_first_word_tokens:
            gt_token_pos.append(list_text_tokens.index(token))
    elif gt_token_not_first_word_tokens[0] in list_text_tokens:
        for token in gt_token_not_first_word_tokens:
            gt_token_pos.append(list_text_tokens.index(token))
    else:
        print('Not found')
        return None, None

    # Create ground-truth binary vector
    # Following prev example: [0,0,0,1,1,1]
    ground_truth_binary_vector = np.zeros(len(tokenized_text))
    ground_truth_binary_vector[gt_token_pos] = 1
    return ground_truth_binary_vector, gt_token_pos
