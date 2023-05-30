#@title Utilities


import numpy as np
import itertools
import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
import random
random.seed(10)

#from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from transformers.modeling_utils import PreTrainedModel
from ipywidgets import IntProgress
import torch.nn.functional as F

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.contributions import ModelWrapper
import matplotlib
#matplotlib.use('Agg')

from captum.attr import visualization

try:
    from IPython.core.display import HTML, display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

rc={'font.size': 20, 'axes.labelsize': 20, 'legend.fontsize': 20,
    'axes.titlesize': 24, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.linewidth': 1, 'figure.figsize': (12,12)}
plt.rcParams.update(**rc)

color_name = 'color{}'
define_color = '\definecolor{{{}}}{{HTML}}{{{}}}'
box = '\\mybox{{{}}}{{\strut{{{}}}}}'

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM

def load_model_tokenizer(name_path, only_tokenizer=False):
    '''Load model and tokenizer.'''

    if name_path == 'facebook/opt-125m':
        # use_fast = False as indicated in https://huggingface.co/docs/transformers/model_doc/opt
        tokenizer = AutoTokenizer.from_pretrained(name_path, use_fast=False)
        if only_tokenizer == False:
            model = AutoModelForCausalLM.from_pretrained(name_path)
    elif name_path == 'gpt2' or name_path == 'gpt2-large' or name_path == 'gpt2-xl':
        tokenizer = GPT2Tokenizer.from_pretrained(name_path)
        if only_tokenizer == False:
            model = GPT2LMHeadModel.from_pretrained(name_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name_path)
        if only_tokenizer == False:
            model = AutoModelForCausalLM.from_pretrained(name_path)

    if only_tokenizer == False:
        model.to(device)
        model.zero_grad()
    else:
        model = None

    return model, tokenizer

def normalize_attribution_visualization(attributions):
    """Applies min-max normalization for visualization purposes."""

    min_importance_matrix = attributions.min(0, keepdim=True)[0]
    max_importance_matrix = attributions.max(0, keepdim=True)[0]
    attributions = (attributions - min_importance_matrix) / (max_importance_matrix - min_importance_matrix)
    return attributions
@torch.no_grad()
def normalize_contributions(model_contributions,scaling='minmax',resultant_norm=None):
    """Normalization of the matrix of contributions/weights extracted from the model."""

    normalized_model_contributions = torch.zeros(model_contributions.size())
    for l in range(0,model_contributions.size(0)):

        if scaling == 'min_max':
            ## Min-max normalization
            min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
            max_importance_matrix = model_contributions[l].max(-1, keepdim=True)[0]
            normalized_model_contributions[l] = (model_contributions[l] - min_importance_matrix) / (max_importance_matrix - min_importance_matrix)
            normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)

        elif scaling == 'sum_one':
            normalized_model_contributions[l] = model_contributions[l] / model_contributions[l].sum(dim=-1,keepdim=True)
            #normalized_model_contributions[l] = normalized_model_contributions[l].clamp(min=0)

        # For l1 distance between resultant and transformer vectors we apply min_sum
        elif scaling == 'min_sum':
            if resultant_norm == None:
                min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(min_importance_matrix)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
            else:
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(resultant_norm[l].unsqueeze(1))
                normalized_model_contributions[l] = torch.clip(normalized_model_contributions[l],min=0)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
        elif scaling == 'softmax':
            normalized_model_contributions[l] = F.softmax(model_contributions[l], dim=-1)
        elif scaling == 't':
            model_contributions[l] = 1/(1 + model_contributions[l])
            normalized_model_contributions[l] =  model_contributions[l]/ model_contributions[l].sum(dim=-1,keepdim=True)
        else:
            print('No normalization selected!')
    return normalized_model_contributions

def plot_histogram(input_tensor,text):
    """Helper function to make bar plots."""

    input_tensor = input_tensor.cpu().detach().numpy()
    # Creating plot
    fig = plt.figure(figsize =(20,4))
    ax = fig.add_subplot(111)
    ax.bar(range(0,input_tensor.size),input_tensor)
    plt.xticks(ticks = range(0,input_tensor.size) ,labels = text, rotation = 45)
    return ax

def get_raw_att_relevance(full_att_mat, layer=-1,token_pos=0):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    return att_sum_heads[layer][token_pos,:]

def spearmanr(x, y):
    """Compute Spearman rank's correlation bertween two attribution vectors.
        https://github.com/samiraabnar/attention_flow/blob/master/compute_corel_distilbert_sst.py"""

    x = pd.Series(x)
    y = pd.Series(y)
    assert x.shape == y.shape
    rx = x.rank(method='dense')
    ry = y.rank(method='dense')
    d = rx - ry
    dsq = np.sum(np.square(d))
    n = x.shape[0]
    coef = 1. - (6. * dsq) / (n * (n**2 - 1.))
    return [coef]

def get_normalized_rank(x):
    """Compute normalized [0,1] word importance ranks. The higher the value, the higher the rank."""
    
    length_tok_sentence = x.shape
    x = pd.Series(x)
    rank = x.rank(method='dense')
    rank_normalized = rank/length_tok_sentence
    return rank_normalized

def get_rank(x):
    """Compute word importance ranks. The higher the value, the higher the rank."""
    length_tok_sentence = x.shape
    x = pd.Series(x)
    rank = x.rank(method='dense')
    return rank

def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):    
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens[:len(attributions)],
                            delta))
                        

def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token

def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 207
        sat = 75
        lig = 100 - int(40 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)

def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def latex_colorize(text, weights):
    """https://github.com/ucinlp/facade/blob/facade/util/generate_colorize.py"""

    s = ''
    for w, x in zip(text, weights):
        w = w.replace('#','\#')
        color = np.digitize(x, np.arange(0, 1, 0.001))# - 1
        s += ' ' + box.format(color_name.format(color), w)
    return s

def prepare_colorize():
    """ Define scale of colors for macros.tex."""

    with open('latex_saliency/colorize.tex', 'w') as f:
        cmap = plt.cm.get_cmap('seismic')
        for i, x in enumerate(np.arange(0, 1.001, 0.001)):
            rgb = matplotlib.colors.rgb2hex(cmap(x)[:3])
            # convert to upper to circumvent xcolor bug
            rgb = rgb[1:].upper() if x > 0 else 'FFFFFF'
            f.write(define_color.format(color_name.format(i), rgb))
            f.write('\n')
        f.write('''\\newcommand*{\mybox}[2]{\\tikz[anchor=base,baseline=0pt,rounded corners=0pt, inner sep=0.2mm] \\node[fill=#1!60!white] (X) {#2};}''')
        f.write('\n')
        f.write('''\\newcommand*{\mybbox}[2]{\\tikz[anchor=base,baseline=0pt,inner sep=0.2mm,] \\node[draw=black,thick,fill=#1!60!white] (X) {#2};}''')


def figure_saliency(attributions_list, tokenized_text, methods_list, methods_dict):
    """ Creates rows of paper's tables by adding the corresponding color to the text."""
    words_weights = []
    for i in methods_list:
        attr = attributions_list[i]
        # We need to replace RoBERTa's special character
        words_weights.append((tokenized_text[i].replace('\u0120',''), element.item()) for i, element in enumerate(attr))
    
    with open('latex_saliency/figure.tex', 'w') as f:
        
        for i, ww in enumerate(words_weights):
            method_latex = methods_dict[methods_list[i]]
            f.write('''\multicolumn{1}{l}{''' + method_latex + '''}\\\ \n''')
            words, weights = list(map(list, zip(*ww)))
            f.write(latex_colorize(words, weights)+'\\\\\n')
            if i == len(words_weights)-1:
                f.write('''\\bottomrule''')
            else:
                f.write('''\\addlinespace\n''')

def get_word_attributions(subword_attributions, subwords_words, pos=0):
    '''
    From subwords tokens contributions to words contribution
    '''
    word_attributions = []
    for i, word in enumerate(subwords_words):
        word_attributions.append(subword_attributions[word].sum())

    return np.array(word_attributions)

def convert_subwords_word(splited_subword_sent, splited_word_sent):
    """
    Given a sentence made of BPE subwords and words (as a string), 
    returns a list of lists where each sub-list contains BPE subwords
    for the correponding word.
    """
    word_to_bpe = [[] for _ in range(len(splited_word_sent))]
    word_i = 0
    for bpe_i, token in enumerate(splited_subword_sent):
        if bpe_i == 0:
            word_to_bpe[word_i].append(bpe_i)
        # if bpe_i == 0:
        #     # First token may use ▁
        #     word_to_bpe[word_i].append(bpe_i)
        else:
            if not token.startswith("##"):
                word_i += 1
            word_to_bpe[word_i].append(bpe_i)
        
    for word in word_to_bpe:
        assert len(word) != 0
    
    #word_to_bpe.append([len(splited_subword_sent)])
    return word_to_bpe

def compute_joint_attention(att_mat):
    """ Compute attention rollout given contributions."""

    joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)
    layers = joint_attentions.shape[0]
    joint_attentions = att_mat[0].unsqueeze(0)

    for i in range(1,layers):
        C_roll_new = torch.matmul(att_mat[i],joint_attentions[i-1])
        joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)
        
    return joint_attentions

from extract_explanations import tokens2words

def compute_alti(layer_alti_data):
    resultant_norm = torch.norm(torch.squeeze(layer_alti_data['attn_res_outputs']),p=1,dim=-1)
    normalized_contributions_alti = normalize_contributions(layer_alti_data['layerwise_contributions'],scaling='min_sum',resultant_norm=resultant_norm)
    contributions_mix_alti = compute_joint_attention(normalized_contributions_alti)
    return contributions_mix_alti

def normalize_contributions_for_visualization(explanations_list, tokenized_text, mean=False, max=False, min=0, word_level_contribs=True, add_pred=True, bos=False):
    normalized_explanations = []
    for explanation in explanations_list:

        if word_level_contribs:
            tokens_in_words, words = tokens2words(tokenized_text, bos)
            # Combine contributions by summing contributions of tokens of the same word
            word_contribs = []
            for word_pos in tokens_in_words:
                word_contrib = 0
                for token_pos in word_pos:
                    word_contrib += explanation[token_pos]
                word_contribs.append(word_contrib)
            # contribs_input_tok now has word-level contributions
            explanation = word_contribs

        # Normalize for visualization
        if mean == False:
            #normalized_explanation = ((np.array(explanation)-1)/(1-(-1))/2)+ 0.5
            normalized_explanation =  np.array(explanation) + 0.5
            #normalized_explanation = np.array(explanation) - np.array(explanation).mean() + 0.5
        else:
            if max != False:
                normalized_explanation = ((np.array(explanation)-min)/(max-min)/2)+ 0.5#- mean 
        if add_pred:
            normalized_explanation = np.append(normalized_explanation, [0.5], axis=0)
        normalized_explanations.append(normalized_explanation)

        if word_level_contribs:
            text_explanations = words
        else:
            text_explanations = tokenized_text

    return normalized_explanations, text_explanations

def figure_saliency_layers(attributions_list, tokenized_text, methods_list, methods_dict):
    """ Creates rows of paper's tables by adding the corresponding color to the text."""
    words_weights = []
    for i in methods_list:
        attr = attributions_list[i]
        # We need to replace RoBERTa's special character
        words_weights.append((tokenized_text[i].replace('\u0120',''), element.item()) for i, element in enumerate(attr))
    
    with open('latex_saliency/figure.tex', 'w') as f:        
        for i, ww in enumerate(words_weights):
            method_latex = methods_dict[methods_list[i]]
            if i > (len(words_weights)-10):
                f.write(method_latex + '''\,\,\,\;|''')
            else:
                f.write(method_latex + '''\;|''')
            words, weights = list(map(list, zip(*ww)))
            f.write(latex_colorize(words, weights)+'\\\\\n')
            if i == len(words_weights)-1:
                f.write('''\\bottomrule''')
            else:
                f.write('''\\addlinespace\n''')
