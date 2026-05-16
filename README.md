# Comment-Category-Prediction

A competition solution for classifying online comments into categories using NLP and machine learning. The pipeline combines TF-IDF text features with engineered numerical features, evaluated across three models — with **LightGBM** emerging as the best performer.

---

## Problem Statement

Given a dataset of online comments with metadata (votes, timestamps, sensitivity flags), predict the **category/label** of each comment. This is a multi-class text classification problem.

**Evaluation Metric:** Macro F1-Score

---

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | Labelled comments used for model training |
| `test.csv` | Unlabelled comments for final prediction |
| `sample.csv` | Submission format template |


**Key Columns:**
- `comment` — raw text of the comment
- `upvote`, `downvote` — community engagement signals
- `race`, `religion`, `gender`, `disability` — sensitivity flags
- `created_date` — timestamp of the comment
- `label` — target variable (train only)

---

## Approach

### 1. Exploratory Data Analysis
- Class distribution of the target variable
- Comment length and word count distributions
- Average upvotes/downvotes per label
- Sensitivity flag rates (race, religion, gender) by label
- Correlation heatmap of numerical features
- Hour-of-day patterns (time-dependent toxicity hypothesis)

### 2. Data Preprocessing
- Dropped the single row with a missing comment
- Converted `race`, `religion`, `gender` NaN values → binary flags (1 if mentioned, 0 if not)
- Cast `disability` boolean to integer
- Extracted `hour` from `created_date`

### 3. Text Cleaning
- Lowercasing
- URL removal
- Digit removal
- Punctuation stripping

### 4. Feature Engineering

| Feature | Description |
|---------|-------------|
| `total_votes` | upvote + downvote (total engagement) |
| `vote_ratio` | upvote / (total_votes + 1) |
| `vote_diff` | upvote − downvote (net sentiment) |
| `comment_length` | Character count |
| `word_count` | Word count |
| `hour` | Hour of posting |

### 5. TF-IDF Vectorization

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 15,000 | Cap vocabulary to avoid noise |
| `ngram_range` | (1, 2) | Unigrams + bigrams |
| `stop_words` | english | Remove common non-informative words |
| `min_df` | 3 | Ignore very rare terms |
| `max_df` | 0.9 | Ignore overly common terms |

### 6. Feature Combination
- TF-IDF sparse matrix (15,000 text features) + StandardScaler-normalised numerical features (17 columns) combined using `scipy.sparse.hstack`

---

## Models & Results

All models evaluated using **Stratified 5-Fold Cross-Validation** with Macro F1-Score.

| Model | CV Macro F1 |
|-------|-------------|
| Logistic Regression | 0.78 |
| Multinomial Naive Bayes | 0.491 |
| **LightGBM** | **0.791** |

> LightGBM was selected as the final model and further tuned with RandomizedSearchCV.

### LightGBM Hyperparameter Tuning

| Parameter | Tuned Values |
|-----------|-------------|
| `n_estimators` | 300, 500 |
| `learning_rate` | 0.03, 0.05 |
| `num_leaves` | 64, 128 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.5 |
| `reg_lambda` | 0.5 |

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── notebook.ipynb          # Main project notebook
```

---


## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
scipy
jupyter
```

---

## Key Takeaways

- Combining TF-IDF text features with engineered numerical features (votes, length, time) improved model performance over text-only approaches
- LightGBM outperformed both Logistic Regression and Naive Bayes on this task
- Sensitivity flags (race, religion, gender) showed meaningful correlation with certain comment categories
- Hour of posting was a useful signal, supporting the hypothesis that comment tone varies by time of day

---
