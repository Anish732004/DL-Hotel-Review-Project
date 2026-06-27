# Hotel Trust & Fraud Intelligence System
## Phase 1: Fake Review Detection — Strategy Document

> **Project Scope:** Binary classification of hotel reviews as Genuine (0) or Fake (1)
> **Fake Review Definition:** Fabricated positive reviews (hotel self-boosting) AND fabricated negative reviews (competitor sabotage)
> **Phase Focus:** Text-only signals. Behavioral and venue-level features are deferred to future phases.

---

## 1. Data Strategy

### 1.1 Datasets and Their Roles

#### Dataset 1: Deceptive Opinion Spam Corpus (Primary — Labeled)

| Attribute | Detail |
|-----------|--------|
| **Size** | 1,600 reviews — 800 fake + 800 genuine |
| **Source of fakes** | Amazon Mechanical Turk workers asked to write convincing fabricated reviews |
| **Source of genuine** | Scraped from TripAdvisor (verified real guest reviews) |
| **Domain** | 20 Chicago hotels |
| **Role** | Training AND evaluation via cross-validation |

**Why this dataset:**
This is the **only publicly available dataset with confirmed ground-truth labels** for fake vs. genuine hotel reviews. Without ground-truth labels, supervised learning is impossible — the model cannot learn to distinguish fake from genuine if it's never shown verified examples of both. Every other "labeled" dataset either uses noisy proxy labels (Yelp filter status) or crowd-sourced guesses. The Deceptive Opinion Spam Corpus was constructed by Myle Ott et al. at Cornell using a rigorous protocol: real guests wrote genuine reviews, paid workers fabricated fake ones. This controlled setup gives us confidence that the labels are correct.

**If not used:**
There is no other clean-labeled hotel review dataset. Without it, we would have to either: (a) rely on Yelp's filter status as labels, which has a ~10-15% error rate and would poison our training data with mislabeled examples, or (b) manually label reviews ourselves, which is subjective, time-consuming, and would produce lower-quality labels than the Cornell protocol. Both alternatives would directly degrade model precision — the metric we care most about.

---

#### Dataset 2: Yelp Open Dataset — Reviews (~1,979 hotel-filtered)

| Attribute | Detail |
|-----------|--------|
| **Original size** | 50,000 sampled reviews (multi-category) |
| **After hotel filter** | ~1,979 reviews |
| **Labels** | All assumed genuine (label = 0) |
| **Role** | (a) Embedding fine-tuning source, (b) Real-world validation |

**Why this dataset:**
Serves two purposes. First, it provides a **larger text corpus for fine-tuning word embeddings** — the Deceptive Corpus alone (1,600 reviews) is too small to capture the vocabulary and phrasing diversity of real hotel reviews. Second, it acts as a **real-world generalizability test** — after training on the controlled Deceptive Corpus, we run the model on real Yelp reviews to check: "Does our model wrongly flag genuine real-world reviews as fake?" This directly measures our #1 concern (false positives on real reviews).

**If not used:**
Without Yelp data, our model would only be evaluated on the Deceptive Corpus, which has a specific writing style (TripAdvisor genuine + MTurk fakes, all from Chicago hotels, from one time period). We would have **no evidence** that the model generalizes beyond this narrow distribution. A model that achieves 90% precision on the Deceptive Corpus might flag 40% of real Yelp reviews as fake — and we'd never know.

> [!IMPORTANT]
> **Known Limitation:** Yelp reviews are assumed genuine based on Yelp's internal fraud filters. This is imperfect — studies estimate a 10-15% error rate. Some "genuine" Yelp reviews may actually be fake. This is acknowledged as a limitation and not mitigated in Phase 1. It means our real-world validation (false positive rate on Yelp) may be slightly inflated, since some flagged reviews might genuinely be fake.

---

#### Dataset 3: Yelp Open Dataset — Business (393 hotels)

| Attribute | Detail |
|-----------|--------|
| **Size** | 10,000 businesses loaded → 393 hotels identified |
| **Filter logic** | Categories containing: hotel, motel, inn, resort, lodging |
| **Role** | Filters Yelp reviews to the hotel domain only |

**Why this dataset:**
Without the business filter, Yelp reviews include restaurants, salons, auto shops — completely different domains with different writing patterns. A restaurant review mentioning "the pasta was cold" is stylistically different from a hotel review mentioning "the room was spacious." If we don't filter, we're comparing hotel-domain fakes (Deceptive Corpus) against multi-domain genuines (Yelp), which introduces a **domain confound** — the model might learn "hotel language = fake" rather than actual deception patterns.

**If not used:**
The model would learn domain differences, not deception differences. It would flag any hotel-style review as suspicious simply because the training fakes are hotel-specific while the "genuine" training data spans multiple domains. This is a form of **data leakage through domain mismatch**.

---

#### Datasets NOT Used in Phase 1

| Dataset | Why Deferred |
|---------|-------------|
| **Yelp User** (user.json) | Provides behavioral features (account age, friend count, elite status). Deferred because Phase 1 is text-only. Behavioral signals are the backbone of Phase 2. |
| **Yelp Checkin** (checkin.json) | Provides venue-level legitimacy signals (review-to-checkin ratio). Deferred for the same reason — requires cross-referencing non-text data sources. |

---

### 1.2 Data Flow (Phase 1)

```
Deceptive Opinion Spam Corpus (1,600 labeled reviews)
  |--- Stratified 5-Fold CV ---> Training folds + Evaluation fold
  |                                  (rotated 5 times)
  |
  '--- All 1,600 reviews ---> Primary metrics:
                                 Precision, F1, Accuracy

Yelp Hotel Reviews (~1,979 assumed genuine)
  |--- Text corpus ---> Fine-tune GloVe embeddings
  |
  '--- After model training ---> Secondary validation:
                                    False positive rate
                                    "How many real reviews
                                     get wrongly flagged?"
```

---

## 2. Preprocessing Strategy

### 2.1 LSTM Path — Minimal Preprocessing

| Step | Decision | Rationale |
|------|----------|-----------|
| Lowercase | Yes | Reduces vocabulary size without losing meaning |
| Remove special characters | No | Punctuation abuse (!!!, ..., ???) is a deception signal |
| Remove stop words | No | Fake reviews overuse function words (I, we, me, the). Removing them destroys a key discriminative pattern |
| Lemmatization | No | "amazing" vs. "amazingly" may carry different deception weights. Collapsing them loses nuance |
| Stemming | No | Same reason as lemmatization, plus stemming introduces artifacts |

**Why minimal preprocessing:**
Research on deceptive language (Newman et al., 2003; Ott et al., 2011) has shown that fake reviews have distinct **stylistic fingerprints** that exist in the very features standard NLP preprocessing removes:
- **Function word frequency:** Fake reviews use "I", "we", "my" more often (psychological distancing is hard to fake)
- **Punctuation patterns:** Excessive exclamation marks, ellipsis, and question marks signal emotional exaggeration
- **Capitalization:** "AMAZING HOTEL" vs "nice hotel" — caps ratio differs between fake and genuine

The LSTM needs to see these raw signals to learn them.

**If we used standard preprocessing:**
We would strip out the exact features that distinguish fake from genuine text. The model would be left with only semantic content ("hotel room clean good"), which is easy for a skilled fake writer to replicate. Standard preprocessing essentially **removes the detective's best clues** and forces the model to work with what's left. Studies on the Deceptive Opinion Spam Corpus show up to 3-5% accuracy drops when aggressive preprocessing is applied.

### 2.2 Baseline Path — Standard Preprocessing

| Step | Decision |
|------|----------|
| Lowercase | Yes |
| Remove special characters | Yes |
| Remove stop words | Yes |
| Lemmatization | Yes |

**Why standard preprocessing for baselines:**
TF-IDF (used by baselines) constructs a vocabulary from the corpus. Without preprocessing, "Hotel", "hotel", "HOTEL", "hotels" become four separate features, diluting each one's signal. TF-IDF works best with a clean, consolidated vocabulary. Additionally, the baselines already receive handcrafted features (caps_ratio, exclamation_count, etc.) that explicitly encode the stylistic signals we're preserving in the LSTM path. So the baselines get those signals through engineered features, not through raw text.

---

## 3. Feature Strategy — Clean Separation

### 3.1 The Principle

> **Baselines** receive **handcrafted features** (human-engineered signals).
> **LSTM models** receive **raw text only** (the model learns its own features).
> These two paths are never mixed.

### 3.2 Why Clean Separation

This separation enables a **fair and meaningful comparison** that answers the core research question: *"Can an LSTM automatically learn representations that match or exceed features carefully engineered by domain experts?"*

| Outcome | Interpretation |
|---------|---------------|
| LSTM beats baselines | "LSTM discovers patterns humans didn't encode — deep learning adds value for this task" |
| LSTM ties baselines | "LSTM matches human feature engineering without manual effort — validates the approach" |
| LSTM loses to baselines | "With only 1,600 samples, deep learning needs more data to outperform domain expertise — honest and publishable" |

**Every outcome is a defensible academic conclusion.**

**If we mixed handcrafted features into the LSTM:**
We would get better raw performance numbers, but we would **not know why**. Did the LSTM's learned representations help, or were the handcrafted features doing the heavy lifting? The XAI story also collapses — LIME would attribute importance to engineered features like "sentiment_polarity = 0.92" instead of showing which actual words drove the prediction. The model becomes a black box within a black box.

### 3.3 Baseline Features (13 Handcrafted Text Features)

| Feature | What It Captures | Deception Signal |
|---------|-----------------|------------------|
| text_length | Character count | Fakes often shorter or artificially padded |
| word_count | Word count | Similar to text_length, captures review effort |
| sentence_count | Number of sentences | Genuine reviews tend to have more varied structure |
| avg_word_length | Average characters per word | Fake writers may use simpler vocabulary |
| exclamation_count | Count of "!" | Fakes tend to over-exaggerate with punctuation |
| question_count | Count of "?" | Genuine reviews ask more rhetorical questions |
| ellipsis_count | Count of "..." | Fakes use ellipsis for dramatic effect |
| caps_ratio | Percentage of uppercase characters | "AMAZING!!!" — fake exaggeration signal |
| sentiment_polarity | TextBlob polarity (-1 to +1) | Fakes often show extreme sentiment (very positive or very negative) |
| sentiment_subjectivity | TextBlob subjectivity (0 to 1) | Fakes are often highly subjective, lacking factual detail |
| superlative_count | Count of superlatives (best, worst, etc.) | Fakes overuse superlatives to compensate for lack of real experience |
| max_word_freq | Highest frequency of any single word | Fake writers often repeat key persuasion words |
| unique_word_ratio | Unique words / total words | Low ratio = repetitive writing, common in template-based fakes |

### 3.4 LSTM Input

Raw review text → Tokenized (word-level) → Padded/truncated to fixed sequence length → Embedded via GloVe 300d (fine-tuned on Yelp hotel reviews)

No handcrafted features are injected at any point in the LSTM pipeline.

---

## 4. Word Embeddings

### 4.1 Decision: GloVe 300d + Fine-tuning on Yelp Hotel Reviews (Hybrid Approach)

| Component | Detail |
|-----------|--------|
| **Base embeddings** | GloVe 300d (pre-trained on 6B tokens from Wikipedia + Gigaword) |
| **Fine-tuning corpus** | Yelp hotel reviews (~1,979 reviews) |
| **Embedding dimension** | 300 |
| **Trainable during LSTM training?** | Yes, but with a reduced learning rate to preserve pre-trained knowledge |

### 4.2 Why This Approach

GloVe 300d provides **high-quality general word representations** trained on billions of tokens. Words like "hotel", "room", "service" already have meaningful vectors. Fine-tuning on Yelp hotel reviews adjusts these vectors to capture **domain-specific nuances** — for example, "clean" in a hotel review context carries different connotations than "clean" in a general text.

**If we trained embeddings from scratch on just ~1,979 reviews:**
The embedding layer would have roughly 10,000-15,000 unique tokens to learn representations for, from fewer than 2,000 documents. Most words would appear only a handful of times, producing **noisy, unreliable vectors**. Words like "concierge" or "amenities" might appear 3-5 times in the entire corpus — not enough to learn a meaningful representation. The LSTM would be handicapped before it even starts learning.

**If we used GloVe without fine-tuning:**
This would work reasonably well, but hotel-specific word relationships (e.g., "housekeeping" is closely related to "cleanliness" in this domain) wouldn't be captured. Fine-tuning is a modest improvement for minimal additional effort.

**If we used contextual embeddings (BERT, etc.):**
BERT-style embeddings would likely perform better, but the project focus is specifically on demonstrating LSTM capability. Using BERT would shift the credit from the LSTM architecture to the embeddings, undermining the project's thesis. BERT-based approaches can be discussed as future work.

---

## 5. Model Architecture — Progressive Narrative

### 5.1 The Progression

The models are presented as a deliberate progression, each building on the previous to tell a coherent story:

---

#### Step 1: TF-IDF + Logistic Regression / SVM (Baseline)

**Architecture:**
```
Review text --> Standard preprocessing --> TF-IDF vectorization
                                               |
                                 13 handcrafted text features
                                               |
                                 Concatenated feature vector
                                               |
                                 Logistic Regression / SVM
                                               |
                                       Fake (1) or Genuine (0)
```

**Why this step exists:**
Establishes a **performance floor**. If a simple model with engineered features achieves 85% precision, the LSTM needs to beat or match that to justify its complexity. Without a baseline, there's no reference point — an LSTM precision of 88% means nothing if we don't know that Logistic Regression achieves 87%.

**If we skipped this:**
We'd have no way to answer "Was the LSTM worth the effort?" The evaluators would rightfully ask: "How does this compare to simpler approaches?" Having no answer is a significant academic weakness.

---

#### Step 2: Vanilla LSTM

**Architecture:**
```
Review text --> Minimal preprocessing --> Tokenization --> Padding
                                                             |
                                                   GloVe 300d Embeddings
                                                             |
                                                   LSTM Layer (unidirectional)
                                                             |
                                                   Last hidden state
                                                             |
                                                   Dense --> Sigmoid
                                                             |
                                                   Fake (1) or Genuine (0)
```

**Why this step exists:**
The first deep learning model in the progression. Uses raw text instead of engineered features. Shows whether sequential text processing (reading the review word by word) can detect deception patterns that bag-of-words (TF-IDF) misses — specifically, **word order matters**. "Not a bad hotel" vs. "a bad hotel, not" have different meanings that TF-IDF cannot distinguish but LSTM can.

**If we skipped this:**
We'd jump from a non-sequential model (baseline) to a bidirectional sequential model (BiLSTM), missing the intermediate insight: "Does directionality matter?" The progression would be incomplete.

---

#### Step 3: Bidirectional LSTM

**Architecture:**
```
Review text --> Minimal preprocessing --> Tokenization --> Padding
                                                             |
                                                   GloVe 300d Embeddings
                                                             |
                               Forward LSTM ----+---- Backward LSTM
                                                |
                                    Concatenated hidden states
                                                |
                                          Dense --> Sigmoid
                                                |
                                        Fake (1) or Genuine (0)
```

**Why this step exists:**
A unidirectional LSTM reads "The hotel was not clean" and by the time it reaches "clean", it has context from "The hotel was not." But it processes "clean" without knowing what comes AFTER it. A BiLSTM reads in both directions, capturing **full context around each word**. For deception detection, backward context is valuable — knowing that a review ends with "...definitely recommend!" changes how earlier hedging words should be interpreted.

**If we skipped this:**
We'd lose the ability to show whether bidirectional context improves deception detection. This is a low-cost addition (just one extra LSTM layer running backward) with potentially meaningful gains.

---

#### Step 4: BiLSTM + Attention (Hero Model)

**Architecture:**
```
Review text --> Minimal preprocessing --> Tokenization --> Padding
                                                             |
                                                   GloVe 300d Embeddings
                                                             |
                               Forward LSTM ----+---- Backward LSTM
                                                |
                                    All hidden states (sequence)
                                                |
                                        Attention Mechanism
                                    (learns weight for each word)
                                                |
                                    Weighted sum --> Context vector
                                                |
                                          Dense --> Sigmoid
                                                |
                                        Fake (1) or Genuine (0)

                                    + Attention weights extracted
                                      for XAI visualization
```

**Why this step exists:**
This is the **culmination model** that serves dual purposes: (a) performance — attention allows the model to focus on the most deception-relevant words rather than compressing the entire review into a single fixed-size vector, and (b) **explainability** — the attention weights tell us WHICH words the model considered most important for its prediction, directly feeding into the XAI requirement.

**If we skipped this:**
We'd lose our strongest XAI capability. Without attention, explaining LSTM decisions requires purely post-hoc methods (LIME, SHAP). With attention, we have a **built-in interpretability mechanism** that can be visualized as heatmaps over the review text. This is the only model in the progression that is interpretable by design.

> [!NOTE]
> **Academic caveat to document:** The "Attention is not Explanation" debate (Jain & Wallace, 2019 vs. Wiegreffe & Pinter, 2019) argues that attention weights don't always faithfully represent the model's decision process. Our strategy addresses this by using attention as **one of three XAI signals**, not the sole explanation. LIME provides a complementary, model-agnostic verification.

---

### 5.2 Overfitting Mitigations

With only 1,600 labeled samples, deep learning models are prone to memorization. The following mitigations are mandatory:

| Technique | Why It's Used | If Not Used |
|-----------|---------------|-------------|
| **Pre-trained GloVe embeddings** | Reduces the number of parameters the model needs to learn from scratch. The embedding layer (often the largest layer) comes pre-initialized. | The model must learn word representations from 1,600 samples — nearly impossible. Performance would be near-random. |
| **Dropout (0.3-0.5)** | Randomly deactivates neurons during training, forcing the network to not rely on any single feature. | The model memorizes training data, achieving 99% training accuracy but ~60% test accuracy. Classic overfitting. |
| **Recurrent Dropout (0.2-0.3)** | Same as dropout but applied to the recurrent connections within the LSTM. Prevents the hidden state from memorizing sequence-specific patterns. | LSTM hidden states co-adapt, memorizing specific review sequences rather than learning general deception patterns. |
| **L2 Regularization** | Penalizes large weights, keeping the model simpler. | Model develops extreme weight values that perfectly fit training data but generalize poorly. |
| **Early Stopping (patience=5-10)** | Monitors validation loss and stops training when it starts increasing, even if training loss is still decreasing. | Model trains for too many epochs, progressively overfitting. The "sweet spot" is missed. |
| **Stratified K-Fold CV** | Ensures every data point is used for both training and validation across folds, and class proportions are preserved. | A single unlucky train/test split could give misleading results. With 1,600 samples, variance between splits is high. |

---

## 6. Evaluation Framework

### 6.1 Primary Evaluation — Stratified 5-Fold CV on Deceptive Corpus

**Why 5-Fold Cross-Validation:**
With 1,600 samples, a single 80/20 split gives us only 320 test samples (160 fake, 160 genuine). Results would be highly sensitive to which specific reviews ended up in the test set — one unlucky split could swing precision by 10+ percentage points. 5-Fold CV rotates through 5 different splits, using each subset once for testing, and averages the results. This produces **stable, reliable metrics**.

**If we used a single train/test split:**
One "lucky" split might give 92% precision; another might give 78%. We wouldn't know which is real. With K-Fold, we get a mean and standard deviation, enabling honest reporting: "Our model achieves 88% +/- 3% precision."

**Why stratified:**
The combined dataset is imbalanced (~23% fake, ~77% genuine if Yelp is included; 50/50 if Deceptive only). Stratification ensures each fold preserves the class ratio. Without it, one fold might accidentally get 90% of the fake reviews, producing artificially high recall and low precision.

### 6.2 Metrics — Hierarchy and Rationale

#### Primary Metric: Precision

```
                    True Positives
Precision = -----------------------------------
             True Positives + False Positives
```

**Why Precision is primary:**
The project's core constraint is: **flagging a genuine review as fake is worse than missing a fake review.** A false positive means a real hotel guest's legitimate feedback is suppressed, undermining trust in the platform and potentially exposing the system to legal/reputational risk. Precision directly measures: "Of all the reviews we flagged as fake, what percentage were actually fake?"

**If we optimized for recall instead:**
The model would aggressively flag reviews as fake to "catch more fakes," inevitably increasing false positives. A recall-optimized model might catch 95% of fakes but also wrongly flag 30% of genuine reviews — unacceptable for a trust system.

#### Secondary Metric: F1-Score

```
                  2 x Precision x Recall
F1-Score = -----------------------------------
                Precision + Recall
```

**Why F1 is secondary:**
F1 is the harmonic mean of precision and recall, providing a **balanced view**. It acts as a guardrail against a degenerate precision-maximizing strategy: a model could achieve 100% precision by only flagging the single most obvious fake review and ignoring the other 799. F1 penalizes this behavior because recall would be near 0%, pulling F1 down to near zero.

**If we didn't track F1:**
We could technically achieve "100% precision" by making the model extremely conservative (only predicting fake when 99.9% confident), catching almost no fakes. F1 ensures we maintain a reasonable balance.

#### Reported for Completeness: Accuracy

```
               Correct Predictions
Accuracy = ----------------------------
              Total Predictions
```

**Why accuracy is NOT a primary or secondary metric:**
With the combined dataset (~77% genuine, ~23% fake), a trivially stupid model that predicts "genuine" for EVERY review achieves ~77% accuracy. This is meaningless. Accuracy is reported because stakeholders expect it, but it is explicitly flagged as unreliable for this task.

**If we used accuracy as the primary metric:**
We would be incentivized to build a model that simply predicts the majority class. A "100% genuine" model looks great on accuracy but catches zero fakes. The entire project would be pointless.

### 6.3 Evaluation Visualizations

| Visualization | What It Shows | Why It Matters |
|---------------|--------------|----------------|
| **Precision-Recall Curve** | How precision and recall trade off at different classification thresholds | More informative than ROC for imbalanced problems. Shows the "sweet spot" where precision is high enough without sacrificing too much recall |
| **Confusion Matrix** | 2x2 grid of TP, FP, TN, FN with cost framing | Makes errors tangible: "12 genuine reviews were wrongly flagged (FP) while 8 fake reviews were missed (FN)" |
| **Threshold Analysis Table** | Precision/Recall/F1 at thresholds 0.5, 0.6, 0.7, 0.8, 0.9 | Shows that the system can be tuned: "At 0.9 threshold, we achieve 95% precision at the cost of catching fewer fakes." Demonstrates operationalizability |
| **Model Comparison Table** | All metrics for Baseline, LSTM, BiLSTM, BiLSTM+Attention | Single table showing the progressive improvement story |

### 6.4 Secondary Validation — Yelp Real-World Check

After the model is trained and evaluated on the Deceptive Corpus:

1. Run the trained BiLSTM+Attention model on all ~1,979 Yelp hotel reviews
2. All Yelp reviews are assumed genuine (label = 0)
3. Measure: **False Positive Rate** = (Number of Yelp reviews flagged as fake) / (Total Yelp reviews)
4. Ideal result: Low FPR (< 5-10%) — meaning the model doesn't aggressively flag real-world reviews

**Why this validation:**
The Deceptive Corpus is a controlled, lab-created dataset. Real-world reviews differ in vocabulary, length, style, and diversity. A model might achieve 90% precision on lab data but flag 40% of real Yelp reviews as fake because Yelp reviews "look different" from TripAdvisor reviews (the genuine source in the Deceptive Corpus). This validation catches that failure mode.

**If we skipped this:**
We'd have no evidence of real-world generalizability. The model's value proposition — "it can be deployed on real hotel platforms" — would be unsupported. Evaluators could rightfully ask: "Does this work outside the lab?"

> [!WARNING]
> **Interpretation caveat:** If the model flags, say, 12% of Yelp reviews as fake, we cannot immediately conclude those are all false positives. Given that Yelp's filter has a ~10-15% error rate, some of those flagged reviews might actually be fake. This should be discussed qualitatively in the analysis.

---

## 7. Explainability — Three-Pronged XAI Strategy

### 7.1 The Principle

A fraud detection system that simply says "this review is fake" without justification is **unusable and untrustworthy**. Hotel managers, platform moderators, and legal teams need to understand WHY a review was flagged. Our XAI strategy provides three independent, complementary explanation signals.

### 7.2 Prong 1: Attention Weight Visualization (Built-in)

| Attribute | Detail |
|-----------|--------|
| **Source** | BiLSTM+Attention model's attention layer |
| **Output** | Heatmap over review text — each word colored by its attention weight |
| **Interpretation** | Words with high attention weights were "focused on" by the model when making its prediction |

**Example output:**
```
"The [hotel] was [absolutely] [AMAZING] I [definitely] [recommend] this place to everyone"
       0.02       0.15        0.31     0.22          0.18
```
High-attention words: "AMAZING", "definitely", "recommend" — exaggeration and superlative patterns.

**Why this method:**
Attention weights are computed during the forward pass — no additional computation or approximation needed. They provide an **intuitive, word-level visualization** that non-technical stakeholders can understand: "The model flagged this review because it focused on these exaggerated words."

**If we didn't include attention:**
We'd rely solely on post-hoc methods (LIME), which are approximations. Attention gives us a window into the model's actual internal mechanism, even if imperfect.

> [!NOTE]
> **Academic caveat:** Jain and Wallace (2019) demonstrated that attention weights don't always faithfully represent a model's true decision process — alternative attention distributions can produce identical predictions. Wiegreffe and Pinter (2019) counter-argued that attention does carry explanatory signal in many cases. We use attention as **one of three explanation sources**, not the sole basis for interpretation. LIME provides independent verification.

### 7.3 Prong 2: LIME — Local Interpretable Model-agnostic Explanations (Post-hoc)

| Attribute | Detail |
|-----------|--------|
| **Source** | LIME library applied to any model in the pipeline |
| **Method** | Perturbs input text (randomly removing words), observes how predictions change, fits a local linear model |
| **Output** | List of words with positive (pushes toward fake) or negative (pushes toward genuine) contribution scores |

**Why this method:**
LIME is **model-agnostic** — it works identically on the Logistic Regression baseline and the BiLSTM+Attention model. This enables cross-model XAI comparison: "Both the baseline and the LSTM agree that 'definitely recommend' is a deception signal." When LIME and attention weights agree, explanations are more trustworthy. When they disagree, it reveals where the model's attention might be misleading.

**If we didn't include LIME:**
We'd have only attention weights (potentially unreliable) and linguistic analysis (model-external). LIME bridges the gap — it's a rigorous, peer-reviewed method that provides word-level explanations through controlled perturbation experiments. Without it, our XAI story has a credibility gap.

### 7.4 Prong 3: Linguistic Feature Analysis (Domain Knowledge)

| Attribute | Detail |
|-----------|--------|
| **Source** | Statistical analysis of the 13 handcrafted text features across fake vs. genuine classes |
| **Method** | Distribution comparison, statistical testing (t-test, Mann-Whitney U), visualization |
| **Output** | "Fake reviews have 2.3x more exclamation marks, 40% higher sentiment polarity, and 15% shorter average word length than genuine reviews" |

**Why this method:**
This is the **most human-interpretable** explanation and the only one grounded in domain expertise rather than model internals. It answers: "What linguistic patterns objectively distinguish fake from genuine reviews in our data?" This is independent of any model — it's an empirical observation about the data itself. It serves as a sanity check: if the LSTM focuses on words that align with known linguistic deception patterns, we have convergent evidence.

**If we didn't include this:**
Our explanations would be entirely model-dependent. A stakeholder could ask: "But WHY do fake reviews use more exclamation marks? Is that a real pattern or a model artifact?" The linguistic analysis answers this definitively with data, not model outputs.

### 7.5 XAI Convergence — The Strength of Three Prongs

The real power is when all three agree:

```
Attention says:   "AMAZING" had the highest attention weight
LIME says:        Removing "AMAZING" reduced fake probability by 0.23
Linguistics say:  Fake reviews use 2.1x more superlatives than genuine

--> CONVERGENT EVIDENCE: Superlative overuse is a verified deception signal,
    the model has correctly learned it, and removing it changes the prediction.
    This is a trustworthy explanation.
```

When they disagree, that's also informative — it flags areas where the model may be relying on spurious correlations.

---

## 8. Summary — Decision Map

| Component | Decision | Core Rationale |
|-----------|----------|----------------|
| Primary data | Deceptive Opinion Spam Corpus | Only dataset with ground-truth fake/genuine labels |
| Supporting data | Yelp hotel reviews | Embedding source + real-world validation |
| Phase 1 scope | Text-only | Behavioral features deferred to Phase 2 |
| LSTM preprocessing | Minimal (keep stops, punctuation) | Deception signals live in stylistic features that standard preprocessing removes |
| Baseline preprocessing | Standard (lemmatize, remove stops) | TF-IDF needs clean vocabulary; handcrafted features capture style separately |
| Feature strategy | Clean separation (handcrafted for baselines, raw text for LSTM) | Enables fair comparison and clean XAI |
| Embeddings | GloVe 300d + Yelp fine-tuning | Too few samples for from-scratch training; GloVe provides strong base |
| Model progression | Baseline, LSTM, BiLSTM, BiLSTM+Attention | Progressive narrative; each step justified |
| Overfitting defense | Dropout, recurrent dropout, L2, early stopping, pre-trained embeddings | 1,600 samples is dangerously small for deep learning |
| Primary metric | Precision | False positives (flagging genuine as fake) are the worst outcome |
| Secondary metric | F1 | Guards against precision gaming (predicting fake only when 99.9% sure) |
| Accuracy | Reported, not optimized | Misleading with class imbalance; a naive model gets ~77% |
| Primary evaluation | Stratified 5-Fold CV on Deceptive Corpus | Stable metrics from limited data; every sample used for testing |
| Secondary evaluation | FPR on Yelp hotel reviews | Real-world generalizability check |
| XAI method 1 | Attention weights | Built-in word-level heatmap; acknowledged academic debate |
| XAI method 2 | LIME | Model-agnostic post-hoc verification; cross-model comparison |
| XAI method 3 | Linguistic feature analysis | Data-grounded, model-independent domain evidence |

---

## 9. Known Limitations and Future Work

### Phase 1 Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Deceptive Corpus fakes are MTurk-generated, not real-world fakes | Model may not catch sophisticated paid review farms or LLM-generated fakes | Acknowledged; Yelp validation partially addresses generalizability |
| Yelp "genuine" assumption has ~10-15% error rate | Some test "genuine" reviews may actually be fake, slightly inflating FPR | Acknowledged as limitation; discussed qualitatively |
| Small dataset (1,600 labeled samples) | Deep learning may not reach full potential | Mitigated with pre-trained embeddings, regularization, K-Fold CV |
| Chicago hotels only (Deceptive Corpus) | Geographic and cultural bias in writing patterns | Yelp adds geographic diversity |
| Text-only analysis | Misses behavioral fraud patterns (account age, review timing, social connections) | Deferred to Phase 2 |

### Phase 2 Preview

- Integrate behavioral features from Yelp user.json (account age, friend count, elite status, review frequency)
- Add venue-level signals from Yelp checkin.json (review-to-checkin ratio)
- Handle NaN behavioral features for Deceptive Corpus records (imputation strategy TBD)
- Explore hybrid architecture: LSTM text encoding + behavioral feature vector --> fusion layer --> classification
