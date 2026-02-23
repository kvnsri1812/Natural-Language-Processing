## Bigram Language Model Implementation

### Student Information

Name: Komatlapalli Venkata Naga Sri

ID: 700773763 

Course: CS5760 – Natural Language Processing  

Semester: Spring 2026  

---

## 📌 Project Description

This project implements a Bigram Language Model using Maximum Likelihood Estimation (MLE).  
The model is trained on a small corpus:
```bash
<s> I love NLP </s>  
<s> I love deep learning </s>  
<s> deep learning is fun </s>  
```
The program:

- Computes unigram counts
- Computes bigram counts
- Estimates bigram probabilities using MLE
- Calculates sentence probabilities
- Determines which sentence is more probable

---

## 🧠 Mathematical Formulation

Bigram probability is computed using:

P(w₂ | w₁) = Count(w₁, w₂) / Count(w₁)

Sentence probability is:

P(sentence) = ∏ P(wᵢ | wᵢ₋₁)

---

## 📂 Files Included

- `Q5. Programming_Bigram_Language_Model.py` → Main implementation
- `README.md` → Project documentation

---

## ▶️ How to Run

1. Clone the repository

2. Navigate to the project directory


3. Run the program:
```bash
python3 Q5. Programming_Bigram_Language_Model.py
```


---

## Expected Output

The program prints:

- Probability of sentence 1
- Probability of sentence 2
- Which sentence the model prefers
- Explanation based on higher probability

---

## Key Concepts Used

- N-gram Language Models
- Maximum Likelihood Estimation (MLE)
- Unigram & Bigram counts
- Sentence probability computation

---

## Conclusion

The model prefers the sentence with the higher product of bigram probabilities.  
This demonstrates how statistical language models rank sentences based on learned corpus statistics.

---
