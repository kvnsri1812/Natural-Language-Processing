"""
Bigram Language Model (MLE)

Description:
This program trains a simple Bigram Language Model from a small tokenized corpus.
It counts unigrams and bigrams, computes bigram probabilities using Maximum Likelihood
Estimation (MLE), and then calculates the probability of two test sentences by
multiplying their bigram probabilities. Finally, it reports which test sentence
is more likely under the trained model.
"""

from collections import defaultdict


# -------------------------------
# Step 1: Define Training Corpus
# -------------------------------

corpus = [
    ["<s>", "I", "enjoy", "NLP", "</s>"],
    ["<s>", "I", "enjoy", "machine", "learning", "</s>"],
    ["<s>", "machine", "learning", "is", "useful", "</s>"]
]


# ---------------------------------
# Step 2: Compute Unigram Counts
# ---------------------------------

unigram_counts = defaultdict(int)

for sentence in corpus:
    for word in sentence:
        unigram_counts[word] += 1


# ---------------------------------
# Step 3: Compute Bigram Counts
# ---------------------------------

bigram_counts = defaultdict(int)

for sentence in corpus:
    for i in range(1, len(sentence)):
        prev_word = sentence[i - 1]
        current_word = sentence[i]
        bigram_counts[(prev_word, current_word)] += 1


# ---------------------------------
# Step 4: Bigram Probability (MLE)
# ---------------------------------

def bigram_probability(w1, w2):
    """
    Returns P(w2 | w1) using MLE:
        P(w2 | w1) = Count(w1, w2) / Count(w1)
    """
    if unigram_counts[w1] == 0:
        return 0.0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


# ---------------------------------
# Step 5: Sentence Probability
# ---------------------------------

def sentence_probability(sentence):
    """
    Computes sentence probability by multiplying bigram probabilities.
    """
    probability = 1.0
    for i in range(1, len(sentence)):
        probability *= bigram_probability(sentence[i - 1], sentence[i])
    return probability


# ---------------------------------
# Step 6: Test Sentences
# ---------------------------------

sentence1 = ["<s>", "I", "enjoy", "NLP", "</s>"]
sentence2 = ["<s>", "I", "enjoy", "machine", "learning", "</s>"]

prob1 = sentence_probability(sentence1)
prob2 = sentence_probability(sentence2)


# ---------------------------------
# Step 7: Print Results
# ---------------------------------

print("Test Sentence 1:", " ".join(sentence1))
print("P(Test Sentence 1) =", prob1)
print()

print("Test Sentence 2:", " ".join(sentence2))
print("P(Test Sentence 2) =", prob2)
print()

if prob1 > prob2:
    print("Result: Test Sentence 1 is more probable under the bigram model.")
elif prob2 > prob1:
    print("Result: Test Sentence 2 is more probable under the bigram model.")
else:
    print("Result: Both test sentences have the same probability under the bigram model.")
