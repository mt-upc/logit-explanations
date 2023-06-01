import os
import argparse
import json
from tqdm import tqdm
import pandas as pd
import csv
from collections import defaultdict
import json
import pickle
import torch
import src.utils_contributions as utils_contributions
from src.contributions import ModelWrapper

from lm_saliency import *

device = "cuda" if torch.cuda.is_available() else "cpu"

our_methods = ['logit_aff_x_j', 'logit_aff_x_j_alti']


def read_sva_dataset():
    df = pd.read_csv('./data/sva_with_targets/lgd_dataset.csv', sep=',', index_col=0)
    return df

def read_blimp_dataset(blimp_subset):
    '''Read blimp_subset dataset as list of lists'''
    blimp_dir = 'data/blimp'
    df = pd.read_csv(f'{blimp_dir}_with_targets/{blimp_subset}.csv', index_col=0)

    return df

def read_ioi_dataset():
    '''Read blimp_subset dataset as list of lists'''
    df = pd.read_csv(f'./data/ioi_with_targets/ioi_dataset.csv', index_col=0)

    return df

def save_logits_preds(info_sentence, dataset, name_path):
    os.makedirs(f'./results/{name_path}', exist_ok=True)
    info_sentence_results_file = f'./results/{name_path}/{dataset}_info_sentence.csv'
    with open(info_sentence_results_file, 'w', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f, delimiter ='|')
        writer.writerow(('sentence', 'logit_diff', 'correct_id', 'foil_id'))
        writer.writerows(info_sentence)

def tokens2words(tokenized_text, bos=False):
    words = []
    tokens_in_words = []
    if bos==True:
        init_sentence = 1
    else:
        init_sentence = 0
 
    for counter, tok in enumerate(tokenized_text):
        if bos==True and counter==0:
            words.append(tokenized_text[counter])
            tokens_in_words.append([counter])

        elif tok.startswith('Ġ') or counter==init_sentence:# or tok in punctuation:
            if tok.startswith('Ġ'):
                tok = tok[1:]
            words.append(tok)
            tokens_in_words.append([counter])
        else:
            words[-1] += tok
            tokens_in_words[-1].append(counter)
    return tokens_in_words, words

def track2input_tokens(logit_trans_vect_dict, methods, contributions_mix_alti, token_list):
    '''Gets layer-wise Attn logits contributions and tracks them down to the input.
            logit_trans_vect_dict: dictionary 'logit_attn_simp' and 'logit_attn_full'
            contributions_mix_alti: layerwise ALTI contributions
    '''
    results_dict = defaultdict(list)
    for token in range(len(token_list)):
        for method in methods:
            layerwise_contributions = logit_trans_vect_dict[method]
            # Assume no token mixing across layers
            results_dict[f'logit_{method}'].append(layerwise_contributions[:, -1, token].cpu().detach())

            for layer in range(0,layerwise_contributions.shape[0]):
                # Track contributions to the input via ALTI (contributions_mix_alti), M matrix in the paper
                if layer == 0:
                    alti_logit_layer_token = layerwise_contributions[layer, -1, token].cpu().detach().unsqueeze(0)
                else:
                    # Multiply attn decomposition by ALTI_{l-1}
                    token_layer_contribs = torch.matmul(layerwise_contributions[layer, -1, token].cpu().detach(),contributions_mix_alti[layer-1])
                    alti_logit_layer_token = torch.cat([alti_logit_layer_token, token_layer_contribs.unsqueeze(0)], dim=0)

            results_dict[f'logit_{method}_alti'].append(alti_logit_layer_token)

    return results_dict

def main(args):
    name_path = args.name_path
    model, tokenizer = utils_contributions.load_model_tokenizer(name_path)
    explanation_type = args.explanation_type

    model_wrapped = ModelWrapper(model)
    
    # Create dictionary where attibution scores are stored
    explanations_dict = defaultdict(list)

    # Load dataset as DataFrame
    dataset = args.dataset
    if 'sva' in dataset:
        
        df = read_sva_dataset()
        print('\ndataset', dataset)
        print(dataset[-1])
        num_attractors = int(dataset[-1])
        # Filter by num_attractors
        if num_attractors != -1:
            df = df[df['num_attractors']==num_attractors].reset_index(drop=True)
        num_examples = 200

    elif dataset == 'ioi':
        df = read_ioi_dataset()
        num_examples = len(df)
    else:
        df = read_blimp_dataset(dataset)
        num_examples = len(df)
    
    logits_modules_list = []
    info_sentence = []
    
    for idx in tqdm(range(num_examples)):

        model.zero_grad()

        # Load text from DataFrame
        if 'sva' in dataset:
            text = df['one_prefix_prefix'][idx][:df['one_prefix_prefix'][idx].index('***mask***')-1]
        else:
            text = df['one_prefix_prefix'][idx]
        input = text
        target = df['one_prefix_word_good'][idx]
        foil = df['one_prefix_word_bad'][idx]

        # Tokenize target and foil
        if 'facebook/opt' in tokenizer.name_or_path:
            # OPT tokenizer adds a BOS token at pos 0 when 
            # tokenizing, so we pick second position
            CORRECT_ID = tokenizer(" " + target)['input_ids'][1]
            FOIL_ID = tokenizer(" " + foil)['input_ids'][1]
            min_length = 2
        else:
            CORRECT_ID = tokenizer(" " + target)['input_ids'][0]
            FOIL_ID = tokenizer(" " + foil)['input_ids'][0]
            min_length = 1

        # Get number of layers in model
        try:
            num_layers = model.config.n_layers
        except:
            num_layers = model.config.num_hidden_layers

        # Tokenize sentence
        pt_batch = tokenizer(text, return_tensors="pt").to(device)
        tokenized_text = tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])

        # If sentence contains just one token skip sentence
        if CORRECT_ID == FOIL_ID or len(tokenized_text) == min_length:
            # Add zeros explanation not to affect the order when evaluating
            contra_explanation = np.zeros(len(tokenized_text))
            if 'ours' in explanation_type:
                for method in our_methods:
                    explanations_dict[method].append(contra_explanation.tolist())
                    #if method == 'logit_aff_x_j' or method == 'logit_aff_x_j_alti':
                    for layer in range(num_layers):
                        explanations_dict[f'{method}_layer_{str(layer)}'].append(contra_explanation.tolist())
                
                logits_modules_list.append('NA')
                info_sentence.append([text, 0, CORRECT_ID, FOIL_ID])
            elif 'grad' in explanation_type:
                explanations_dict['grad_norm'].append(contra_explanation.tolist())
                explanations_dict['grad_norm_2'].append(contra_explanation.tolist())
                explanations_dict['grad_inp'].append(contra_explanation.tolist())
                explanations_dict['grad_inp_2'].append(contra_explanation.tolist())
            elif explanation_type == 'erasure':
                explanations_dict['erasure'].append(contra_explanation.tolist())
            continue

        if 'ours' in explanation_type:
            # Run inference
            logits, hidden_states, attentions = model_wrapped(pt_batch)
            correct_id_logit = logits[0, -1, CORRECT_ID].item()
            foil_id_logit = logits[0, -1, FOIL_ID].item()
            info_sentence.append([text, correct_id_logit - foil_id_logit, CORRECT_ID, FOIL_ID])

            # Our Approach (layerwise logits contributions)
            # Contrastive explanation
            token = [CORRECT_ID, FOIL_ID]
            logit_trans_vect_dict, logits_modules, layer_alti_data = model_wrapped.get_logit_contributions(hidden_states, attentions, token)
            # ALTI results
            contributions_mix_alti = utils_contributions.compute_alti(layer_alti_data)
            methods_decomp = ['aff_x_j'] # Logits Affine part of layer-wise decomposition
            # Track layer-wise Attn and MLPs contributions to input
            alti_lg_dict = track2input_tokens(logit_trans_vect_dict, methods_decomp, contributions_mix_alti, token)

            for method in our_methods:
                # Get logit difference between tokens and sum across layers
                contrastive_contributions = (alti_lg_dict[method][0] - alti_lg_dict[method][1]).sum(0)
                explanations_dict[method].append(contrastive_contributions.tolist())
                #if method == 'logit_attn_full' or method == 'logit_attn_full_alti':
                for layer in range(num_layers):
                    contrastive_contributions = (alti_lg_dict[method][0][layer] - alti_lg_dict[method][1][layer])
                    explanations_dict[f'{method}_layer_{str(layer)}'].append(contrastive_contributions.tolist())
                
            # Add difference logits
            logits_modules['correct_id_logit'] = correct_id_logit
            logits_modules['foil_id_logit'] = foil_id_logit
            logits_modules_list.append(logits_modules)

        else:
            # Kayo Yin results
            input = input.strip() + " "
            input_tokens = tokenizer(input)['input_ids']
            attention_ids = tokenizer(input)['attention_mask']

            if explanation_type == 'erasure':
                contra_explanation = erasure_scores(model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True)
                explanations_dict['erasure'].append(contra_explanation.tolist())

            elif 'grad' in explanation_type:
                saliency_matrix, embd_matrix = saliency(model, input_tokens, attention_ids, foil=FOIL_ID)
                contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)
                explanations_dict['grad_norm'].append(contra_explanation.tolist())
                contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)
                explanations_dict['grad_inp'].append(contra_explanation.tolist())

                model.zero_grad()
                saliency_matrix, embd_matrix = saliency(model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID)
                contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)
                explanations_dict['grad_norm_2'].append(contra_explanation.tolist())
                contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)
                explanations_dict['grad_inp_2'].append(contra_explanation.tolist())

    name_path = name_path.replace('/','-')
    save_logits_preds(info_sentence, dataset, name_path)
    print('explanations_dict', explanations_dict.keys())
    
    if 'sva' in dataset:
        os.makedirs(f'./results/{dataset}', exist_ok = True)
        save_dir = f'./results/{dataset}/{dataset}_{name_path}_{explanation_type}_{str(num_attractors)}.json'
    elif dataset == 'ioi':
        os.makedirs(f'./results/ioi', exist_ok = True)
        save_dir = f'./results/ioi/{dataset}_{name_path}_{explanation_type}.json'
    else:
        os.makedirs(f'./results/blimp', exist_ok = True)
        save_dir = f'./results/blimp/{dataset}_{name_path}_{explanation_type}.json'
    with open(save_dir, 'w') as fp:
        json.dump(explanations_dict, fp)
    if 'ours' in explanation_type:
        # Save logits info
        os.makedirs(f'./results/logits', exist_ok = True)
        with open(f'./results/logits/{dataset}_{name_path}.pickle', 'wb') as handle:
            pickle.dump(logits_modules_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_path', help="path/name of model", type= str, default='gpt2-large')
    parser.add_argument('--dataset', help="linguistic_phenomena", type= str)
    parser.add_argument('--explanation_type', help="type of explanation: ours/erasure/grad_norm/grad_inp", type= str, default='ours')
    
    args=parser.parse_args()

    main(args)