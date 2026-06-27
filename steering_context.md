# STEERING CONTEXT — Hotel Trust & Fraud Intelligence System (Phase 1)
# Purpose: Token-efficient briefing for the implementing LLM. Read this FIRST.

---

## PROJECT IDENTITY

- **Name:** Hotel Trust & Fraud Intelligence System
- **Current Phase:** Phase 1 — Fake Review Detection (text-only)
- **Task:** Binary classification — Genuine (0) vs Fake (1)
- **Fake definition:** Fabricated positive (self-boosting) + fabricated negative (competitor sabotage)
- **Future phases:** Fraudulent reservations, cancellation abuse, behavioral features

---

## DATA — What's Used and How

### Primary Dataset: Deceptive Opinion Spam Corpus
- **Size:** 1,600 reviews (800 fake + 800 genuine), 20 Chicago hotels
- **Fake source:** Amazon Mechanical Turk (paid writers, fabricated reviews)
- **Genuine source:** TripAdvisor (verified real guests)
- **Role:** ONLY source of labeled data. All model training and primary evaluation happens here.
- **Key property:** Ground-truth labels (not proxy/inferred)

### Supporting Dataset: Yelp Open Dataset (hotel-filtered)
- **Reviews:** ~1,979 hotel reviews (filtered from 50k using business.json)
- **Business:** 393 hotels identified via category filter (hotel/motel/inn/resort/lodging)
- **Labels:** ALL labeled genuine (label=0). Assumption: Yelp's own filters removed most fakes.
- **Role 1 — Embeddings:** Fine-tune GloVe 300d on Yelp hotel review text
- **Role 2 — Validation:** After training, run model on Yelp reviews to measure false positive rate on real-world data

### NOT Used in Phase 1
- `yelp_academic_dataset_user.json` → behavioral features, deferred to Phase 2
- `yelp_academic_dataset_checkin.json` → venue-level signals, deferred to Phase 2

### Known Data Limitation
Yelp "genuine" assumption has ~10-15% error rate. Some Yelp reviews labeled genuine may actually be fake. Acknowledged, not mitigated. Impacts secondary validation (FPR may be slightly inflated).

---

## PREPROCESSING — Two Separate Pipelines

### LSTM Pipeline (raw text input)
```
Lowercase:                YES
Remove special chars:     NO   (punctuation abuse is a deception signal)
Remove stop words:        NO   (fake reviews overuse function words: I, we, my)
Lemmatization:            NO   (preserves nuanced forms: "amazing" vs "amazingly")
Stemming:                 NO
```
**Rationale:** Deception signals live in stylistic features that standard preprocessing destroys.

### Baseline Pipeline (TF-IDF input)
```
Lowercase:                YES
Remove special chars:     YES
Remove stop words:        YES
Lemmatization:            YES
```
**Rationale:** TF-IDF needs clean vocabulary. Stylistic signals are captured separately via 13 handcrafted features.

---

## FEATURE STRATEGY — Clean Separation (CRITICAL)

**Rule: Baselines get handcrafted features. LSTM gets raw text. Never mix.**

### Baseline Features (13 handcrafted text features)
1. text_length
2. word_count
3. sentence_count
4. avg_word_length
5. exclamation_count
6. question_count
7. ellipsis_count
8. caps_ratio
9. sentiment_polarity (TextBlob, -1 to +1)
10. sentiment_subjectivity (TextBlob, 0 to 1)
11. superlative_count
12. max_word_freq
13. unique_word_ratio

### LSTM Input
Raw review text → word-level tokenization → pad/truncate to fixed length → GloVe 300d embeddings
No handcrafted features injected anywhere in the LSTM pipeline.

---

## WORD EMBEDDINGS

- **Base:** GloVe 300d (pre-trained, 6B tokens, Wikipedia + Gigaword)
- **Fine-tuning:** On Yelp hotel reviews (~1,979 reviews) for domain adaptation
- **Trainable during LSTM training:** Yes, with reduced learning rate
- **Do NOT train embeddings from scratch** — dataset too small, will produce garbage vectors

---

## MODEL PROGRESSION (in this exact order)

### Step 1: Baseline — TF-IDF + Logistic Regression / SVM
- Input: TF-IDF vectors + 13 handcrafted text features (concatenated)
- Purpose: Performance floor. Answers "was LSTM worth it?"

### Step 2: Vanilla LSTM
- Input: Raw text → GloVe embeddings
- Architecture: Embedding → LSTM (unidirectional) → last hidden state → Dense → Sigmoid
- Purpose: Can sequential processing beat bag-of-words?

### Step 3: Bidirectional LSTM
- Input: Same as Step 2
- Architecture: Embedding → BiLSTM → concatenated hidden states → Dense → Sigmoid
- Purpose: Does bidirectional context help deception detection?

### Step 4: BiLSTM + Attention ★ HERO MODEL
- Input: Same as Step 2
- Architecture: Embedding → BiLSTM → Attention over all hidden states → weighted context vector → Dense → Sigmoid
- Purpose: (a) Performance via selective focus, (b) XAI via attention weight extraction
- **Extract and store attention weights for every prediction** — needed for XAI Prong 1

---

## OVERFITTING MITIGATIONS (mandatory — only 1,600 labeled samples)

| Technique | Setting |
|-----------|---------|
| Pre-trained GloVe | Non-negotiable — do not skip |
| Dropout | 0.3–0.5 on dense layers |
| Recurrent dropout | 0.2–0.3 on LSTM layers |
| L2 regularization | Apply to dense layers |
| Early stopping | Monitor val_loss, patience=5-10 epochs |
| Stratified K-Fold | 5-fold, preserve class ratios |

---

## EVALUATION FRAMEWORK

### Primary Evaluation: Stratified 5-Fold CV on Deceptive Corpus

**Metric hierarchy (strict order):**

| Priority | Metric | Role | Why |
|----------|--------|------|-----|
| PRIMARY | **Precision** | Optimize for this | Flagging genuine as fake is the worst outcome. Precision = "of reviews flagged fake, how many actually were?" |
| SECONDARY | **F1-Score** | Guardrail | Prevents degenerate precision gaming (model that never predicts fake gets infinite precision). F1 ensures recall stays reasonable |
| COMPLETENESS | **Accuracy** | Report only | Misleading with class imbalance. A "predict all genuine" model gets ~77%. Do NOT optimize for this |

**Required visualizations:**
1. Precision-Recall curve (per model)
2. Confusion matrix with cost framing
3. Threshold analysis table (precision/recall/F1 at thresholds: 0.5, 0.6, 0.7, 0.8, 0.9)
4. Model comparison table (all 4 models, all metrics, single table)

**Report format:** Mean ± standard deviation across 5 folds

### Secondary Validation: Yelp Real-World Check
- Run trained BiLSTM+Attention on ~1,979 Yelp hotel reviews
- All assumed genuine → measure False Positive Rate
- Target: FPR < 5-10%
- Caveat: Some flagged reviews may genuinely be fake (Yelp filter imperfect). Discuss qualitatively.

---

## EXPLAINABILITY — Three Prongs (all three required)

### Prong 1: Attention Weights (built-in, BiLSTM+Attention only)
- Extract attention weights per prediction
- Visualize as heatmap over review text
- Caveat: "Attention is not Explanation" debate — use as ONE signal, not sole explanation

### Prong 2: LIME (post-hoc, model-agnostic)
- Apply to ALL models (baselines + LSTMs)
- Shows which words pushed prediction toward fake vs genuine
- Enables cross-model XAI comparison
- Use `lime.lime_text.LimeTextExplainer`

### Prong 3: Linguistic Feature Analysis (domain knowledge)
- Statistical comparison of 13 text features across fake vs genuine classes
- Methods: distribution plots, t-test / Mann-Whitney U
- Model-independent — empirical data observation

### XAI Convergence Analysis
When all three prongs agree on a signal (e.g., superlative overuse), that's convergent evidence.
When they disagree, flag it as a potential spurious correlation. Document both cases.

---

## IMPLEMENTATION CONSTRAINTS

- Phase 1 is TEXT-ONLY. Do not incorporate behavioral or venue-level features.
- Yelp user.json and checkin.json are NOT used in Phase 1.
- Do not use BERT/transformer embeddings — project focus is LSTM demonstration.
- Every model must be evaluated with the SAME evaluation framework (same folds, same metrics).
- All evaluation uses the Deceptive Corpus. Yelp is supplementary validation only.

---

## DECISION LOG — Quick Reference

| Decision | Chosen | Rejected Alternative | Why Rejected |
|----------|--------|---------------------|--------------|
| Primary data | Deceptive Corpus | Yelp as training data | Yelp has no ground-truth fake labels |
| Preprocessing (LSTM) | Minimal | Standard NLP | Destroys deception-specific stylistic signals |
| Feature mixing | Clean separation | Hybrid (handcrafted + LSTM) | Muddies comparison; breaks XAI narrative |
| Embeddings | GloVe 300d + fine-tune | Train from scratch | ~1,979 reviews is far too small for quality embeddings |
| Embeddings | GloVe 300d + fine-tune | BERT/contextual | Project focus is LSTM, not transformer demonstration |
| Primary metric | Precision | Accuracy | Class imbalance makes accuracy meaningless (~77% by guessing) |
| Primary metric | Precision | Recall | Catching more fakes at cost of flagging genuine reviews is unacceptable |
| Secondary metric | F1 | Accuracy | F1 guards against precision gaming; accuracy doesn't |
| Evaluation | Stratified 5-Fold CV | Single train/test split | 1,600 samples makes single splits unreliable |
| XAI | Three-pronged | Attention only | Attention alone is academically contested; LIME + linguistics provide verification |
