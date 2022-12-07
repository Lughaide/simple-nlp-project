import random
from collections import Counter

random.seed(42) # Picked a seed for fixed results

def make_dataset(sentences, filename, tagged = True):
    with open(filename, "w") as f:
        for line in sentences:
            for word in line:
                if tagged:
                    print(" ".join(word.split("/")), file=f)
                else:
                    print(word.split("/")[0], file=f)
            print(file=f)

# Read from dataset
with open("Dataset_processed.txt", "r", encoding="utf-8") as f:
    dataset = f.readlines()

# Create sentences with starting states
sentence_list = []
sentence = ["-S-/S"]
for word in dataset:
    if word == "\n":
        sentence_list.append(sentence)
        sentence = ["-S-/S"]
    else:
        sentence.append(f"{word.split()[0]}/{word.split()[1]}")

# Create random test set
test_set = []
print("Random chain: ", end="")
for i in range(10):
    rand_idx = random.randint(0, len(sentence_list) - 1)
    print(rand_idx, end=" ")
    test_set.append(sentence_list.pop(rand_idx))
print()

print(len(test_set))
print(len(sentence_list))

make_dataset(test_set, "test_set.txt")
make_dataset(test_set, "test_words.txt", tagged=False)
make_dataset(sentence_list, "train_set.txt")

# Create custom vocabulary
word_list = []
for line in sentence_list:
    for word in line:
        word_list.append(word.split("/")[0])

with open("vocab.txt", "w") as f:
    for word in sorted(list(Counter(word_list).keys())):
        print(word, file=f)
    print(file=f)

