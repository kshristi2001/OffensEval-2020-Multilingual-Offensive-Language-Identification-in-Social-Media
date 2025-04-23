# OffensEval 2020: Multilingual Offensive Language Identification in Social Media

This repository contains our submission for [SemEval-2020 Task 12](https://aclanthology.org/2020.semeval-1.188/). The task involves detecting and categorizing offensive content in social media posts across multiple languages.

## 📄 Project Overview

The challenge includes three subtasks:
- **A**: Offensive language detection
- **B**: Categorization of offensive content
- **C**: Target identification for offensive language

We explored classical ML and deep learning models tailored to each task.

## 📁 Structure

```
├── Datasets/
├── Images/
├── subtaska_lstm.py
├── subtaskb_bert.py
├── subtaskc_svm.py
├── AIT_526_NLP_Project_SemEval_2020_Task_12.pdf
└── README.md
```

## 🧠 Models Used

- **Subtask A**: LSTM with GloVe
- **Subtask B**: Fine-tuned multilingual BERT
- **Subtask C**: SVM with TF-IDF vectors

## 🚀 Getting Started

### Requirements
- Python 3.6+
- Install using:
```bash
pip install -r requirements.txt
```

### Run Scripts
```bash
python subtaska_lstm.py
python subtaskb_bert.py
python subtaskc_svm.py
```

## 📊 Results

| Subtask | Model | F1-Score (Macro) | Accuracy |
|--------|--------|------------------|----------|
| A      | LSTM   | 0.88             | 89%      |
| B      | BERT   | 0.83             | 85%      |
| C      | SVM    | 0.80             | 82%      |

## 📚 Reference

- [SemEval 2020 OffensEval Task Paper](https://aclanthology.org/2020.semeval-1.188/)

