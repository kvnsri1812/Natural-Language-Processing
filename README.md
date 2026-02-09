# NLP - Homework 1


---

## Student Information
- **Name:** Komatlapalli Venkata Naga Sri
- **Student ID:** 700773763

---

## Repository Structure

### `Q1.Regex`
**Explanation:**  
Contains solutions for all **regular expression** tasks in Q1. Each regex is listed with a short explanation of what it matches (ZIP codes, non-capital words, rich numbers, email variants, `gooo` interjection, and line-ending `?` patterns).

---

### `Q2.BPE`
**Explanation:**  
Shows the **manual Byte Pair Encoding (BPE)** process for the toy corpus in Q2.1:
- Adds end-of-word marker `_`
- Lists the initial vocabulary
- Counts frequent symbol pairs
- Performs merge steps and shows how vocabulary changes  
Q2.3 work using a **custom paragraph**:
- Trains BPE with `_`
- Learns at least **30 merges**
- Reports:
  - **Top 5 most frequent merges**
  - **Top 5 longest subword tokens**
  - Segmentation of **5 words** (includes one rare word and one derived/inflected word)
- Includes a brief reflection (5–8 sentences) about learned subwords and pros/cons

---

### `Q3.Bayes Rule`
**Explanation:**  
Q3 written explanation of **Bayes Rule for text classification**:
- Defines \(P(c)\), \(P(d|c)\), \(P(c|d)\)
- Explains clearly why \(P(d)\) can be ignored when comparing classes (constant for a fixed document)

---

### `Smoothing`
**Explanation:**  
Q4 calculations and explanation of **Add-1 (Laplace) smoothing**:
- Computes the smoothed denominator \(N_c + |V|\)
- Computes example probabilities like \(P(\text{predictable}\mid -)\) and \(P(\text{fun}\mid -)\)
- Explains why smoothing prevents zero probabilities

---

### `Tokenization.py`
**Explanation:**  
Python program for Q5 that demonstrates:
- Naïve whitespace tokenization (`split()`)
- Manually corrected tokens (punctuation + clitics)
- NLTK `word_tokenize`
- Prints differences between manual vs NLTK tokenization

---

### `Q5.Tokenization`
**Explanation:**  
Written part of Q5:
- Paragraph used
- Naïve tokens and manual tokens
- Tool comparison summary
- **3 MWEs** with explanation
- Reflection (5–6 sentences)

---

## How to Run the Code

```bash
python Tokenization.py
