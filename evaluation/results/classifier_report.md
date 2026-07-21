# Supervised Document Classifier — Training Report

- Corpus: 900 randomized synthetic samples, 6 classes
- Model: TF-IDF (1–2 grams, sublinear) + Logistic Regression
- Held-out accuracy (20% stratified split): **100.0%**

```
                     precision    recall  f1-score   support

           contract      1.000     1.000     1.000        30
     court_document      1.000     1.000     1.000        30
financial_statement      1.000     1.000     1.000        30
            invoice      1.000     1.000     1.000        30
              other      1.000     1.000     1.000        30
            receipt      1.000     1.000     1.000        30

           accuracy                          1.000       180
          macro avg      1.000     1.000     1.000       180
       weighted avg      1.000     1.000     1.000       180
```

## Confusion matrix

| | contract | court_document | financial_statement | invoice | other | receipt |
|---|---|---|---|---|---|---|
| **contract** | 30 | 0 | 0 | 0 | 0 | 0 |
| **court_document** | 0 | 30 | 0 | 0 | 0 | 0 |
| **financial_statement** | 0 | 0 | 30 | 0 | 0 | 0 |
| **invoice** | 0 | 0 | 0 | 30 | 0 | 0 |
| **other** | 0 | 0 | 0 | 0 | 30 | 0 |
| **receipt** | 0 | 0 | 0 | 0 | 0 | 30 |
