from collections import Counter, defaultdict


def tag_unknown_words(emission_counts, prev_word):
    max_cnt = 0
    prob_tag = ""
    for tup,cnt in emission_counts.items():
        if prev_word == tup[1] and cnt > max_cnt:
            prob_tag = tup[0]
            max_cnt = cnt
    return prob_tag

def apply_max_transition(transition_counts, tag):
    max_cnt = 0
    prob_tag = ""
    
    for tup,cnt in transition_counts.items():
        if '--s--' in tup:
            continue
        elif tag == tup[0] and cnt > max_cnt:
            prob_tag = tup[1]
            max_cnt = cnt
    return prob_tag

def label_unk_word(tag):
    tag_list = "N V A P M D R E C I O Z X".split()
    id_list = "-n- -v- -a- -p- -m- -d- -r- -e- -c- -i- -o- -z- -x-".split()
    for count, ex in enumerate(tag_list):
        if tag == ex:
            return id_list[count]
    return "-U-"

with open("test_words.txt", "r") as f:
    temp = f.readlines()

print(Counter(temp))

# TODO:
# - Finish HMM Viterbi
# - Implement Brill Tagger
# - Compare with VNcoreNLP
# - Make presentations
# NOTE