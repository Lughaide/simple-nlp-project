from utils_hmm import *
from utils import *

def predict_highest_pos(predict_set, true_set, emission_set, tag_set):
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
    # Read data from file
    train_set = read_file("train_set.txt")
    test_set = read_file("test_set.txt")
    vocab_list = read_file("vocab.txt")

    # Create vocabulary
    for count, word in enumerate(vocab_list):
        vocab_list[count] = word.split("\n")[0]

    train_count = remove_val_from_list(train_set, '\n')
    test_count = remove_val_from_list(test_set, '\n')
    print(f"> Number of words in train set: {len(train_count)}")
    print(f"> Number of words in test set: {len(test_count)}")
    print(f"> Vocabulary: {len(vocab_list)}")
    
    vocab = make_vocab(vocab_list)

    emission_counts, transition_counts, tag_counts = create_dicts(train_set)
    pred_set = make_pred_set(test_set)

    # Most frequent class
    print(f"Most frequent class Accuracy: {predict_highest_pos(pred_set, test_set, emission_counts, tag_counts) * 100:.04f}%")

    # Hidden Markov Model
    alpha = 0.001
    states = sorted(tag_counts.keys())
    A = create_transition_matrix(alpha, transition_counts, tag_counts)
    B = create_emission_matrix(alpha, emission_counts, tag_counts, list(vocab)[:-1])
    
    A_sub = pd.DataFrame(A, index=states, columns=states[1:])
    print(A_sub)

    B_sub = pd.DataFrame(B, index=states[1:], columns=list(vocab)[:-1])
    print(B_sub)

    # Create list of sentences to feed through HMM
    pred_set = create_sentences(pred_set)
    true_set = create_sentences(test_set)

    num_corr = 0
    total = 0

    comparison = False

    for pred_batch, true_batch in zip(pred_set, true_set):
        best_probs, best_paths = initialize(states, tag_counts, A, B, pred_batch, vocab)

        best_subs_1 = pd.DataFrame(best_probs, index=states[1:], columns=pred_batch)
        best_subs_2 = pd.DataFrame(best_paths, index=states[1:], columns=pred_batch)

        best_probs, best_paths = viterbi_forward(A, B, pred_batch, best_probs, best_paths, vocab)

        pred = viterbi_backward(best_probs, best_paths, states[1:])
            
        if comparison:
            print("Pred result:", end=" ")
            for pos, word in zip(pred, pred_batch):
                print(f"{word} {pos}", end=" ")
            print()
            
            print("True result:", end=" ")
            for word in true_batch:
                print(word.strip("\n"), end=" ")
            print()

        x, y = compute_accuracy(pred, true_batch)
        num_corr += x
        total += y
    print(f"HMM Accuracy: {num_corr}/{total} = {(num_corr/total)*100:.04f}%")