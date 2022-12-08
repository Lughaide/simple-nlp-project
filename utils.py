import numpy as np
import pandas as pd
from collections import defaultdict

def remove_val_from_list(data_list, val):
    return [value for value in data_list if value != val]

def read_file(filename):
    with open(filename, "r+", encoding="utf-8") as f:
        file_content = f.readlines()
    return file_content

def make_vocab(vocab_list):
    vocab = {}
    for count, word in enumerate(vocab_list):
        vocab[word] = count
    return vocab

def make_pred_set(dataset):
    pred_set = []
    for count, word in enumerate(dataset):
        if len(word.split()):
            pred_set.append(word.split()[0])
        else:
            pred_set.append(word)
    return pred_set

def count_sentences(dataset):
    sen_count = 0
    for word in dataset:
        if not len(word.split()):
            sen_count += 1
    return sen_count

def create_sentences(dataset):
    sentence_list = []
    sentence = []
    for word in dataset:
        if word == "\n":
            sentence_list.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    return sentence_list

def create_dicts(train_data):
    # Create dictionaries
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    # Initialize starting state
    prev_tag = "-S-"

    for word_tag in train_data:
        if len(word_tag.split()) < 1:
            prev_tag = "-S-"
            tag_counts[(prev_tag)] += 1
            continue
        else:
            word, tag = word_tag.split()
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            tag_counts[(tag)] += 1
            prev_tag = tag
    
    return emission_counts, transition_counts, tag_counts