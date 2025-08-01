# Semi-Supervised Learning Toolkit

An easy-to-use modular toolkit for training and evaluating **Semi-Supervised Learning (SSL)** algorithms across **images**, **text**, and **tabular** inputs.  
Built to support real-world applications where labeled data is limited, but unlabeled data is abundant.

---

## Algorithms Implemented

| Algorithm        | Supported Inputs       | Description |
|------------------|-------------------------|-------------|
| [Mean Teacher](./algorithms/mean_teacher/)     | Images, Text, Tabular | Uses a student-teacher model with Exponential Moving Average (EMA) for consistent predictions. |
| [Pseudo-Labeling](./algorithms/pseudo_label/)  | Tabular Only          | Assigns pseudo-labels to unlabeled data when the model is confident enough, then retrains. |

---

## General Pipeline (Coming Soon)

A visual diagram of the shared SSL pipeline across input types will be added here to illustrate how tokenization, dataloading, model building, and training connect across methods.

---

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## How to Use

This repository contains two SSL algorithm implementations, each with its own training logic and configuration setup.

- 📄 [Mean Teacher](./algorithms/mean_teacher/) - Works with image, text, and tabular inputs (classification and regression). Implements student-teacher training using EMA updates.
- 📄 [Pseudo-Labeling](./algorithms/pseudo_label/) - Designed for tabular classification. Trains with labeled data, then generates and trains on pseudo-labels.

---

## Reuse

Feel free to fork and adapt the pipelines to your own data or domain.
