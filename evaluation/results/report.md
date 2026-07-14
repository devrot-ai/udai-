# Evaluation Report

- Documents evaluated: **24**
- Document-type detection accuracy: **100.0%**
- Baseline system: naive-regex (spaCy not installed)

| System | Fields correct | Accuracy |
|---|---|---|
| **Hybrid pipeline** | 150/150 | **100.0%** |
| Baseline | 74/150 | 49.3% |

**Improvement: +50.7 percentage points**

## Per document type (hybrid)

| Type | Hybrid | Baseline |
|---|---|---|
| invoice | 48/48 (100.0%) | 12/48 (25.0%) |
| contract | 36/36 (100.0%) | 12/36 (33.3%) |
| financial_statement | 42/42 (100.0%) | 42/42 (100.0%) |
| court_document | 24/24 (100.0%) | 8/24 (33.3%) |
