import py_vncorenlp
import os
from utils import *

def compare_list(a, b):
    equal_count = 0
    total_count = len(b)
    for first, second in zip(a, b):
        if first == second:
            equal_count += 1
    if equal_count > total_count:
        equal_count = total_count
    return [equal_count, total_count]


save_path = './vncorenlp-data'
if os.path.exists(save_path):
    print("Prerequisities are already met.")
    os.chdir(save_path)
else:
    py_vncorenlp.download_model(save_dir='./vncorenlp-data')

print("Loading VNCoreNLP model: ")
model = py_vncorenlp.VnCoreNLP(annotators=["pos"], save_dir='./')
os.chdir("..")

test_set = read_file("test_set.txt")
pred_set = make_pred_set(test_set)

pred_set = create_sentences(pred_set)
true_set = create_sentences(test_set)

num_corr = 0
total = 0
comparison = False
for pred_batch, true_batch in zip(pred_set, true_set):
    input_data = " ".join(pred_batch)
    input_data = input_data.replace("_", " ")
    sentences = model.annotate_text(input_data)
    sent = sentences[0]

    pred_pos = []
    true_pos = []
    for word in sent:
        if comparison: print(f"{word['wordForm']} {word['posTag'][0]}", end=" ")
        pred_pos.append(word['posTag'][0])
    if comparison: print()

    for word in true_batch:
        if comparison: print(word.strip("\n"), end=" ")
        true_pos.append(word.split()[-1])
    if comparison: print()
    
    x, y = compare_list(pred_pos, true_pos)
    num_corr += x
    total += y
print(f"VNCoreNLP Accuracy: {(num_corr/total)*100:.04f}%")