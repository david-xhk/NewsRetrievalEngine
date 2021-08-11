# News Retrieval Engine

50.045 Information Retrieval Project

# Tasks

- [x] (Lucas) Create labelled dataset using the pseudocode (refer to week 10 slides, slide 60)
- [x] (HK) Integrate week 9 lab with the labelled dataset and evaluate results 
- [x] (Jisa) Integrate week 10 lab with the labelled dataset (split into train, val, and test) and evaluate results
- [x] (HK) Train a BERT-based model with the labelled dataset (split into train, val, and test) and evaluate results
- [x] (Everyone) Write the report

# Approach

### Retrieval methods

    1. BM25
    2. Week 9 Lab (Query-likelihood model)
    3. Week 10 lab (LTR model using pairwise approach + RankNet loss)
    4. BERT-based model

### Evaluation metric
    MRR

### Heuristic for creating queries
    1. Use the title
    2. Remove stopwords
    3. Stem words
    4. Convert to lowercase
