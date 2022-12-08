import pandas as pd
from collections import defaultdict

import math
import numpy as np

from utils_hmm import *

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

def predict_pos(predict_set, true_set, emission_set, tag_set):
    num_correct = 0
    total = 0
    emission_entries = emission_set.keys()

    # Traverse through prediction word and true word in pred and true set
    for word_pred, word_true in zip(predict_set, true_set):
        max_count = 0
        pos_final = ''

        # If entry is not an endline or null, start predicting
        if len(word_pred.split()):
            for pos in tag_set:
                word_pos_set = (pos, word_pred) # Create a (pos, word) set for comparison

                if word_pos_set in emission_entries:
                    curr_count = emission_set.get(word_pos_set)

                    # Assign tag to higher count
                    if curr_count > max_count:
                        max_count = curr_count
                        pos_final = pos
            
            if pos_final == word_true.split()[-1]:
                num_correct += 1
        total += 1

    accuracy = num_correct / total
    return accuracy

def initialize(states, tag_counts, A, B, corpus, vocab):
    num_tags = len(tag_counts) - 1
    
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=np.int32)

    s_idx = states.index('-S-')
    for i in range(num_tags):
        best_probs[i, 0] = A[s_idx, i] * B[i, vocab[corpus[0]]]

    return best_probs, best_paths

def compute_accuracy(pred, test):
    num_correct = 0
    total = 0

    for pred_w, true_w, in zip(pred, test):
        if len(true_w.split()):
            if pred_w == true_w.split()[-1]:
                num_correct += 1
            total += 1
    
    return [num_correct, total]

if __name__ == "__main__":
    train_set = read_file("train_set.txt")
    test_set = read_file("test_set.txt")
    vocab_list = read_file("vocab.txt")

    for count, word in enumerate(vocab_list):
        vocab_list[count] = word.split("\n")[0]

    print("Length of train set, test set and vocabulary: ", end="")
    print(len(train_set), len(test_set), len(vocab_list))
    vocab = make_vocab(vocab_list)

    emission_counts, transition_counts, tag_counts = create_dicts(train_set)
    pred_set = make_pred_set(test_set)

    alpha = 0.001

    # Most frequent class
    print(f"Accuracy: {predict_pos(pred_set, test_set, emission_counts, tag_counts) * 100:.04f}%")

    A = create_transition_matrix(alpha, transition_counts, tag_counts)
    states = sorted(tag_counts.keys())
    print(tag_counts)
    A_sub = pd.DataFrame(A, index=states, columns=states[1:])
    print(A_sub)

    B = create_emission_matrix(alpha, emission_counts, tag_counts, list(vocab)[:-1])
    B_sub = pd.DataFrame(B, index=states[1:], columns=list(vocab)[:-1])
    print(B_sub)

    pred_set = create_sentences(pred_set)
    true_set = create_sentences(test_set)

    num_corr = 0
    total = 0
    for pred_batch, true_batch in zip(pred_set, true_set):
        best_probs, best_paths = initialize(states, tag_counts, A, B, pred_batch, vocab)

        best_subs_1 = pd.DataFrame(best_probs, index=states[1:], columns=pred_batch)
        best_subs_2 = pd.DataFrame(best_paths, index=states[1:], columns=pred_batch)

        best_probs, best_paths = viterbi_forward(A, B, pred_batch, best_probs, best_paths, vocab)

        pred = viterbi_backward(best_probs, best_paths, states[1:])
        print(pred, pred_batch)
        print(true_batch)

        x, y = compute_accuracy(pred, true_batch)
        num_corr += x
        total += y
    print(f"HMM Accuracy: {(num_corr/total)*100:.04f}%")