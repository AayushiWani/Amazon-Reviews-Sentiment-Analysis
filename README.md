# Amazon Reviews Sentiment Analysis

Classifies Amazon product reviews as positive or negative using the VADER sentiment lexicon.

## Problem Statement
Determine the sentiment (`pos` / `neg`) of customer reviews and measure how well a rule-based, lexicon approach matches the labelled data.

## Dataset
- 10,000 labelled reviews (`all_rev.tsv`, tab-separated) with columns `review` and `label`
- Roughly balanced: 5,097 negative and 4,903 positive
- File not included in this repo; add your own copy

## Approach
1. Loaded the reviews with pandas
2. Scored each review with **VADER** (`SentimentIntensityAnalyzer`), which returns negative, neutral, positive and compound scores
3. Converted the compound score to a label: `pos` if compound >= 0, otherwise `neg`
4. Evaluated against the true labels with a confusion matrix and classification report

## Results (10,000 reviews)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| neg | 0.87 | 0.52 | 0.65 |
| pos | 0.65 | 0.92 | 0.76 |

**Overall accuracy: 71%**

**Observation:** with a threshold of 0, VADER leans positive. It catches 92% of positive reviews but misses almost half of the negative ones.

## Tech Stack
Python, pandas, scikit-learn (metrics), vaderSentiment

## How to Run
```bash
pip install pandas scikit-learn vaderSentiment
```
1. Place `all_rev.tsv` in your working directory and update the path in the notebook
2. Open `amazon_reviews_sentiment_analysis.ipynb` and run all cells

## Possible Improvements
- Tune the compound-score threshold (for example, use a positive cutoff)
- Add text cleaning and a TF-IDF + Logistic Regression baseline
- Try a transformer-based model for comparison
