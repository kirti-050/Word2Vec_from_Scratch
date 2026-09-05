# Word2Vec From Scratch (NumPy)

A from-scratch implementation of the Word2Vec skip-gram model, built using only NumPy — no deep learning frameworks. Built as a follow-up to my [MNIST Neural Network from Scratch](https://github.com/kirti-050/MNIST-Neural-Network) project, continuing a "from scratch" learning series.

## What this project covers

- Building a vocabulary and word ↔ ID mappings from raw text
- Generating (center word, context word) training pairs using a sliding window
- Two embedding matrices (center and context vectors) per word
- Forward pass using dot products + softmax to compute prediction probabilities
- Cross-entropy loss and manually derived gradients
- A full training loop (gradient descent, no autograd)
- **Negative sampling** — a more scalable training objective using sigmoid instead of full-vocabulary softmax
- Verifying the model actually learned something, using before/after cosine similarity checks on word pairs
- Training on a real corpus (a slice of [text8](http://mattmahoney.net/dc/text8.zip), a cleaned Wikipedia dump) instead of just a toy sentence

## Why negative sampling

Full-vocabulary softmax computes a score against every word in the vocabulary for every training pair — this gets slow as vocabulary grows. Negative sampling instead checks the real context word against just a handful of randomly sampled "negative" words, keeping the cost per pair constant regardless of vocabulary size. It's noisier on a tiny vocabulary but converges cleanly at real-corpus scale.

## Results

| Setup | Vocabulary size | Loss behavior |
|---|---|---|
| Toy sentence, full softmax | 7 words | Smooth decrease, 72 → 33 over 200 epochs |
| Toy sentence, negative sampling | 7 words | Noisy oscillation (45–51) — expected, since negative sampling has little to save at this scale |
| text8 slice, negative sampling | 3,000 words | Smooth decrease, ~44,000 → ~13,000 over 10 epochs |

**Cosine similarity, before vs. after training (text8 slice):**

| Word pair | Before | After |
|---|---|---|
| anarchism / anarchist | -0.16 | 0.38 |
| anarchism / communism | -0.40 | 0.36 |
| state / kropotkin | 0.14 | 0.59 |

Related words moved measurably closer together purely from shared context in the training text — direct evidence the training mechanism works as intended.

## Key learnings

- The "guess → check → nudge" intuition behind word embeddings, and how it maps directly onto softmax (guess), cross-entropy (check), and gradient descent (nudge)
- Why negative sampling trades exact-but-expensive for approximate-but-cheap, and why that trade-off only pays off at real vocabulary scale
- Gradient derivation by hand for both the softmax and negative-sampling objectives

## Tech stack

Python, NumPy, Matplotlib

## Author

Kirti — [LinkedIn](https://www.linkedin.com/in/kirti-srivastava-16a7a3290/) · [GitHub](https://github.com/kirti-050)
