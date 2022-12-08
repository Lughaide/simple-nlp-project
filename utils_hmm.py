__all__ = ["create_emission_matrix", "create_transition_matrix", "initialize", "viterbi_forward", "viterbi_backward"]

from utils import np, remove_val_from_list

def create_transition_matrix(alpha, transition_counts, tag_counts):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags, num_tags))
    trans_key = set(transition_counts.keys())

    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i], all_tags[j])

            if key in trans_key:
                count = transition_counts.get(key)
            
            count_prev = tag_counts.get(all_tags[i])
            A[i, j] = (count + alpha) / (count_prev + alpha * num_tags)
    return A[:, 1:]

def create_emission_matrix(alpha, emission_counts, tag_counts, vocab):
    all_tags = sorted(tag_counts.keys())[1:]
    num_tags = len(tag_counts) - 1
    num_words = len(vocab)

    B = np.zeros((num_tags, num_words))
    emis_key = set(list(emission_counts.keys()))
    
    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key = (all_tags[i], vocab[j])
            if key in emis_key:
                count = emission_counts.get(key)
            
            count_tag = tag_counts.get(all_tags[i])
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return B

def initialize(states, tag_counts, A, B, corpus, vocab):
    num_tags = len(tag_counts) - 1
    
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=np.int32)

    s_idx = states.index('-S-')
    for i in range(num_tags):
        best_probs[i, 0] = A[s_idx, i] * B[i, vocab[corpus[0]]]

    return best_probs, best_paths

def viterbi_forward(A, B, c_corpus, best_probs, best_paths, vocab):
    num_tags = best_probs.shape[0]

    for i in range(1, len(c_corpus)):
        for j in range(num_tags):
            best_probs_init = float("-inf")
            best_path_init = None
            for k in range(num_tags):
                prob = best_probs[k, i-1] * A[k, j] * B[j, vocab[c_corpus[i]]]
                if prob > best_probs_init:
                    best_probs_init = prob
                    best_path_init = k
            best_probs[j, i] = best_probs_init
            best_paths[j, i] = best_path_init
    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, states):
    num_tags = best_probs.shape[0]
    m = best_paths.shape[1]
    z = [None] * m
    pred = [None] * m
    best_prob_last = float("-inf")

    for k in range(num_tags):
        if best_probs[k, m-1] > best_prob_last:
            best_prob_last = best_probs[k, m - 1]
            z[m - 1] = k
    
    pred[m - 1] = states[z[m-1]]

    for i in range(m-1, -1, -1):
        pos_tag_for_word_i = z[i]
        z[i - 1] = best_paths[pos_tag_for_word_i, i]
        pred[i - 1] = states[z[i - 1]]

    return pred