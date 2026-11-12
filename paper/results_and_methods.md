# Paper material: related work, method and results

Working notes for the journal paper. Every number in sections 3 and 4 comes from a run
recorded in the repository README; nothing here is estimated. Section 1 reports what
other papers say and must be checked against their PDFs before it is cited.

Contents: 0. Framing: a modified version of the base paper · 1. Related work and what
this study adds · 2. Open items before submission · 3. Method · 4. Results ·
5. Limitations

---

## 0. Framing: a modified version of the base paper

The paper presents a modified version of the method in *"Enhanced Large Language Model
with a Self-Attention Mechanism for Social Media Rumor Detection"* (the base paper).
The base paper's pipeline has four stages: text pre-processing, RoBERTa feature
extraction, a GPT classifier, and hyperparameter optimisation with the Deer Hunting
Optimization Algorithm (DHOA). Each stage is changed here.

The base paper is an unpublished manuscript that a journal returned. It is the starting
point of this work, not a published result: its reported accuracy is not a baseline
the new paper has to match or explain, and the stage-by-stage mapping below is a working
aid, not text for the paper. The paper's baselines are the published models and
protocols in section 1.1.

### 0.1 What is changed, stage by stage

| Stage | Base paper | This work | What the experiments show |
|---|---|---|---|
| Pre-processing | URL removal, stop-word removal, tokenisation, stemming, lemmatisation | Minimal normalisation only (URLs → `URL`, mentions → `@USER`), so the pretrained encoder sees natural text | — |
| Text representation | RoBERTa embeddings used as features | RoBERTa fine-tuned end to end on the source tweet | Needs up to 30 epochs on the small Twitter sets: 0.688 → 0.784 on Twitter16, 0.736 → 0.770 on Twitter15; no change on PHEME (0.867 → 0.868) |
| Classifier | GPT (self-attention) on the RoBERTa features | Linear head on RoBERTa; the GPT stage is removed | — |
| Propagation (new) | Not used | BiGCN branch on the reply tree | 0.804 / 0.797 / 0.858; above RoBERTa on Twitter15/16 |
| Lexical (new) | Not used | TF-IDF of source tweet and thread words, logistic regression | 0.848 / 0.840 / 0.871; the strongest single branch |
| Combination (new) | — | Validation-weighted average of the branches' probabilities (stacking); joint fusion as an alternative | Three-branch stack against the lexical branch alone: −0.2 / +1.1 / +0.8 points; significant on Twitter15 and PHEME (section 4.7) |
| Hyperparameter search | DHOA over learning rate, batch size, layers, heads, dropout | Grey Wolf Optimizer, nested inside cross-validation so the test fold is never used for tuning | No gain on Twitter16 (+0.1 points, p = 1.0) |
| Evaluation | Split and validation procedure not described | Grouped 5-fold cross-validation with a separate validation fold, paired significance tests, and a story-held-out split | Section 4 |

### 0.2 The proposed pipeline

Three branches trained on the same folds, then combined:

1. **Lexical branch** — TF-IDF + logistic regression.
2. **Text branch** — fine-tuned RoBERTa (`text_long`).
3. **Propagation branch** — BiGCN.
4. **Combination** — the three branches' class probabilities are averaged with weights
   chosen on the validation fold (`stack_full`).

The three-branch combination scores 0.846 / 0.852 / 0.879 on Twitter16 / Twitter15 /
PHEME. Against the lexical branch alone (0.848 / 0.840 / 0.871) that is −0.2 points
(p = 0.88), +1.1 points (p = 0.021) and +0.8 points (p = 0.0059). It is better than the
text branch and the propagation branch alone on all three datasets. See section 4.7
(`notebooks/colab_full_pipeline.ipynb`).

### 0.3 Ablation: what each component contributes (grouped folds, pooled accuracy)

| Configuration | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| Text branch alone, 6 epochs | 0.688 | 0.736 | 0.867 |
| Text branch alone, ≤ 30 epochs | 0.784 | 0.770 | 0.868 |
| Propagation branch alone | 0.804 | 0.797 | 0.858 |
| Propagation branch + GWO search | 0.806 | not run | not run |
| Text + propagation, joint fusion | 0.786 | 0.781 | 0.871 |
| Lexical branch alone | 0.848 | 0.840 | 0.871 |
| Lexical + propagation, stacked | 0.840 | 0.844 | 0.872 |
| Lexical + text (≤ 30 epochs), stacked | 0.856 | 0.844 | 0.880 |
| All three branches, stacked (proposed) | 0.846 | 0.852 | 0.879 |

The table mixes two sets of runs; section 4.7 gives the set in which every
configuration was run together.

### 0.4 Two points to settle with the mentor

1. **How the new paper relates to the returned manuscript.** The base paper's
   99.1–99.5% accuracy was never published, so the new paper does not need to cite or
   explain it. Reviewers will compare the scores (0.85–0.88) with published results on
   the same benchmarks, many of which are higher because they were measured on random
   splits; section 1.1 lists the published work showing that such scores drop sharply
   once leakage is removed. To settle: why the manuscript was returned (reviewer
   comments, if any), and whether the new paper is a revision of it by the same authors.
2. **The strongest component is the simplest one.** The lexical branch does most of
   the work. The full pipeline adds about one point over it on Twitter15 and PHEME
   (significant) and nothing on Twitter16; that gain comes from the RoBERTa branch,
   and the BiGCN branch has no consistent effect. The ablation table has to show that
   as it is.

---

## 1. Related work and what this study adds

### 1.1 What is already published

The central observation of this project, that benchmark scores on these datasets are
inflated by overlap between training and test stories, is already in the literature.

| Paper | What it shows | Overlap with this study |
|---|---|---|
| Wu and Hooi, *Probing Spurious Correlations in Popular Event-Based Rumor Detection Benchmarks*, ECML-PKDD 2022 ([arXiv:2209.08799](https://arxiv.org/abs/2209.08799)) | Twitter15, Twitter16 and PHEME contain many near-identical posts per event. With event-separated splits (298 events in Twitter15, 182 in Twitter16, taken from released event ids) accuracy falls from 74–89% to 31–43% on Twitter15 and from 73–90% to 29–44% on Twitter16. BiGCN scores 42.8 / 44.3 / 43.4. A simple neural bag-of-words over the thread ("MeanText") beats BiGCN on Twitter15/16. | Same datasets, same finding, larger drop. Their event ids are a stronger grouping than our text clusters. |
| Mu, Bontcheva and Aletras, *It's about Time: Rethinking Evaluation on Rumor Detection Benchmarks using Chronological Splits*, Findings of EACL 2023 ([arXiv:2302.03147](https://arxiv.org/abs/2302.03147)) | Random splits put near-identical posts in training and test (23% of Twitter15 and 30% of Twitter16 test posts). With chronological splits, macro-F1 falls from 85–90 to 41–57 on Twitter15. Includes a TF-IDF logistic regression baseline next to BERT and BERTweet; no graph model. Binary task. | Same leakage finding, same kind of baseline; different remedy (time instead of story). |
| Pelrine, Danovitch and Rabbany, *The Surprising Performance of Simple Baselines for Misinformation Detection*, WWW 2021 ([arXiv:2104.06952](https://arxiv.org/abs/2104.06952)) | Fine-tuned transformers on text alone are competitive with published graph and propagation models. Reports duplicate tweets and label leakage through tweet ids in Twitter15/16. | Same "simple text model is competitive" message, with transformers instead of a linear model. |
| Kochkina, Liakata and Zubiaga, *All-in-one: Multi-task Learning for Rumour Verification*, COLING 2018 ([arXiv:1806.03713](https://arxiv.org/abs/1806.03713)) | Leave-one-event-out evaluation on PHEME as the standard protocol for unseen events. | Our PHEME story folds are a 5-fold version of this. |
| Bian et al., *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks*, AAAI 2020 ([arXiv:2001.06362](https://arxiv.org/abs/2001.06362)) | The BiGCN model used as the graph arm here. | Model source. |

The numbers in this table were read from HTML conversions of the papers and must be
verified against the PDFs before they are quoted.

### 1.2 What this study adds

The modified pipeline of section 0 is the paper's subject. Beyond the pipeline itself,
the evaluation gives four results that the papers above cover only partly:

1. **A near-duplicate-grouped protocol between the two extremes.** Published work
   compares random splits with event-separated or chronological splits. Here an
   intermediate protocol removes only near-copies (character n-gram cosine ≥ 0.9) and
   already costs RoBERTa 11 points on Twitter16 (0.799 → 0.688).
2. **A non-neural lexical baseline compared head-to-head with BiGCN and RoBERTa on
   identical folds, with paired tests.** Logistic regression on TF-IDF of the source
   tweet and the thread beats BiGCN by 4.4 / 4.3 / 1.3 points (McNemar p ≤ 0.0003).
   Wu and Hooi show this for a neural bag-of-words under event-separated splits; Mu et
   al. include logistic regression but no graph model.
3. **Accuracy as a function of story granularity.** On Twitter15/16, accuracy falls
   steadily as the definition of "same story" is loosened, with no plateau (0.79 → 0.51
   on Twitter16). An unseen-story score is therefore a point on a curve. The coarse end
   of our curve (234 stories on Twitter16, 0.506) approaches the range Wu and Hooi
   report with 182 released events (29–44%).
4. **Negative results for common additions.** Grey Wolf Optimizer tuning of BiGCN
   (+0.1 points, p = 1.0), joint RoBERTa + BiGCN fusion (not better than BiGCN on
   Twitter), and stacking BiGCN on the lexical branch (no gain on any dataset).

Points 1, 3 and 4 are the parts least covered by the papers above. Point 2 is a
confirmation with a stricter statistical treatment.

---

## 2. Open items before submission

| Item | Why it matters | Cost |
|---|---|---|
| BiGCN and RoBERTa on the story-held-out folds | Only the lexical arm has story-held-out numbers; the comparison is the core table | GPU: `notebooks/colab_story.ipynb` |
| Event ids for Twitter15/16 | Wu and Hooi use released event ids (298 / 182 events). Folds built from them would replace the text clustering and make results directly comparable | Find and check the released ids; CPU |
| More than one seed per fold | Each fold has one training run, so run-to-run variation of the neural arms is not measured | GPU |
| Verify the related-work numbers | Read from HTML conversions | Reading |

---

## 3. Method

### 3.1 Datasets and tasks

| | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| Source tweets (threads) | 818 | 1,490 | 6,425 |
| Task | 4-way | 4-way | rumour vs non-rumour |
| Non-rumour | 205 | 374 | 4,022 |
| False | 205 | 370 | 638 |
| True | 207 | 372 | 1,067 |
| Unverified | 201 | 374 | 698 |
| Majority-class accuracy | 0.253 | 0.251 | 0.626 |
| Near-duplicate clusters | 735 | 1,376 | 6,039 |
| Tweets that have a near-duplicate | 132 | 192 | 658 |
| Stories (text clusters, cosine ≥ 0.3) / events | 498 | 828 | 9 events |

Twitter15 and Twitter16 are the releases of Ma et al. (2017); thread words come from
the public RvNN bag-of-words release (5,000-word vocabulary). PHEME is the 9-event
release; on PHEME the three rumour classes are merged for the binary task.

Text normalisation is minimal: HTML entities are unescaped, URLs become `URL`,
@mentions become `@USER`. No stemming, lemmatisation or stop-word removal.

### 3.2 Models

**Lexical.** Three TF-IDF blocks with sublinear term frequency, fitted on the training
rows of each fold only: word 1–2-grams of the source tweet, character 2–5-grams of the
source tweet, and the bag of words of every tweet in the thread. Multinomial logistic
regression with balanced class weights. The regularisation strength C is chosen from
{1, 5, 20, 100} by validation macro-F1.

**Text (RoBERTa).** `roberta-base` with a linear head on the first-token state, source
tweet only, maximum 128 tokens. AdamW, learning rate 2e-5, weight decay 0.01, batch 16,
10% linear warm-up then linear decay, dropout 0.1, label smoothing 0.1, gradient
clipping at 1.0, mixed precision. `text`: 6 epochs, patience 3. `text_long`: up to 30
epochs, patience 10. The epoch with the best validation macro-F1 is kept.

**Graph (BiGCN).** Top-down and bottom-up two-layer GCNs with root-feature enhancement
and DropEdge, as in Bian et al. (2020). Node features are bag-of-words vectors. Hidden
and output width 64, dropout 0.5, DropEdge 0.2, learning rate 5e-4, weight decay 1e-4,
batch 64, constant learning rate, up to 100 epochs with patience 10, label smoothing
0.1. On PHEME the 5,000-word node vocabulary is re-selected from the training threads
of each fold.

**Fusion.** The RoBERTa embedding and the BiGCN embedding are concatenated and fed to
one linear layer; both branches are trained jointly (learning rates 2e-5 and 5e-4,
batch 16, 8 epochs, patience 3).

**Stack.** A weighted average of the class probabilities of two arms. The weights are
chosen on the validation fold from a grid with step 0.1 (validation macro-F1, ties
broken by log-loss); no member is retrained.

**Hyperparameter search.** Grey Wolf Optimizer over eight BiGCN hyperparameters, 6
wolves × (1 + 5 rounds) = 36 candidates per test fold, nested: candidates are scored
only on the four folds that are not the test fold.

### 3.3 Evaluation protocol

*Grouped 5-fold cross-validation.* Near-duplicate source tweets are clustered
(character 3–5-gram TF-IDF, cosine ≥ 0.9, connected components). Folds are stratified
by label with the clusters as groups, so no test tweet has a near-copy in training or
validation. For test fold k, fold k+1 is the validation set and the other three folds
are the training set: 60% / 20% / 20%. The validation fold is used only for model
selection and is never trained on. Each test fold is predicted once, with the selected
model. One training run per fold, seed 42 + k.

*Story-held-out 5-fold cross-validation.* Same protocol with coarser groups. On PHEME
the group is the event; the nine events are packed into five folds by size. Twitter15
and Twitter16 are used without event ids, so source tweets are clustered by
average-linkage on word TF-IDF vectors (cosine ≥ 0.3) and merged with the
near-duplicate clusters. The threshold is varied from 0.5 to 0.1 as a sensitivity
analysis.

*Statistics.* Accuracy and macro-F1 are computed on the pooled test predictions of the
five folds, so every tweet is tested exactly once. Accuracy is given with a Wilson 95%
interval. Two arms are compared on the same tweets with the paired accuracy difference
(95% interval, normal approximation) and the exact two-sided McNemar test.

---

## 4. Results

### 4.1 Grouped folds: accuracy [Wilson 95% interval] and macro-F1

| Arm | Twitter16 (n = 818) | Twitter15 (n = 1,490) | PHEME (n = 6,425) |
|---|---|---|---|
| Majority class | 0.253 | 0.251 | 0.626 |
| Text, 6 epochs | 0.688 [0.656, 0.719] · 0.684 | 0.736 [0.713, 0.757] · 0.736 | 0.867 [0.858, 0.875] · 0.859 |
| Text, ≤ 30 epochs | 0.784 [0.754, 0.810] · 0.782 | 0.770 [0.748, 0.790] · 0.770 | 0.868 [0.860, 0.876] · 0.860 |
| Graph (BiGCN) | 0.804 [0.776, 0.830] · 0.805 | 0.797 [0.776, 0.817] · 0.798 | 0.858 [0.849, 0.866] · 0.850 |
| Graph + GWO search | 0.806 [0.777, 0.831] · 0.805 | not run | not run |
| Fusion | 0.786 [0.757, 0.813] · 0.783 | 0.781 [0.759, 0.801] · 0.780 | 0.871 [0.863, 0.879] · 0.864 |
| **Lexical** | **0.848 [0.822, 0.871] · 0.849** | **0.840 [0.821, 0.858] · 0.841** | **0.871 [0.862, 0.878] · 0.862** |
| Stack: lexical + graph | 0.840 [0.813, 0.863] · 0.840 | 0.844 [0.824, 0.861] · 0.844 | 0.872 [0.863, 0.880] · 0.863 |
| Stack: lexical + text ≤ 30 | 0.856 [0.830, 0.878] · 0.856 | 0.844 [0.824, 0.861] · 0.844 | 0.880 [0.872, 0.888] · 0.873 |
| Stack: all three branches | 0.846 [0.820, 0.869] · 0.847 | 0.852 [0.833, 0.869] · 0.852 | 0.879 [0.871, 0.887] · 0.871 |

The rows with ≤ 30 epochs and the three-branch stack come from the set of runs in
section 4.7, where the lexical branch scored the same and BiGCN within 0.7 points.

### 4.2 Paired comparisons on the grouped folds

Difference in accuracy points [95% interval]; tweets only the first / only the second
arm classifies correctly; exact McNemar p.

| Comparison | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| Lexical − graph | +4.4 [+2.2, +6.6]; 60 / 24; p = 0.0001 | +4.3 [+2.6, +6.0]; 113 / 49; p < 0.0001 | +1.3 [+0.6, +1.9]; 283 / 202; p = 0.0003 |
| Lexical − text (6 ep.) | +16.0 [+12.7, +19.3]; 169 / 38; p < 0.0001 | +10.5 [+8.2, +12.7]; 233 / 77; p < 0.0001 | +0.4 [−0.4, +1.2]; 386 / 361; p = 0.38 |
| Graph − text (6 ep.) | +11.6 [+8.3, +15.0]; 151 / 56; p < 0.0001 | +6.2 [+3.8, +8.6]; 213 / 121; p < 0.0001 | −0.9 [−1.8, +0.0]; 393 / 449; p = 0.058 |
| Fusion − graph | −1.8 [−4.6, +0.9]; 57 / 72; p = 0.22 | −1.7 [−3.8, +0.5]; 119 / 144; p = 0.14 | +1.3 [+0.5, +2.2]; 417 / 331; p = 0.002 |
| Stack (lex. + graph) − lexical | −0.9 [−1.8, +0.1]; 4 / 11; p = 0.12 | +0.3 [−0.0, +0.7]; 6 / 1; p = 0.13 | +0.1 [−0.2, +0.4]; 49 / 42; p = 0.53 |
| GWO graph − graph | +0.1 [−1.3, +1.6]; 19 / 18; p = 1.0 | not run | not run |
| Text ≤ 30 ep. − lexical | −6.5 [−9.2, −3.7]; 41 / 94; p < 0.0001 | −7.0 [−9.1, −5.0]; 70 / 175; p < 0.0001 | −0.2 [−1.1, +0.6]; 365 / 381; p = 0.58 |
| Stack (lex. + text ≤ 30) − lexical | +0.7 [−0.7, +2.2]; 22 / 16; p = 0.42 | +0.3 [−0.2, +0.9]; 11 / 6; p = 0.33 | +1.0 [+0.3, +1.7]; 283 / 220; p = 0.0056 |
| Stack (all three) − lexical | −0.2 [−1.8, +1.3]; 21 / 23; p = 0.88 | +1.1 [+0.2, +2.1]; 33 / 16; p = 0.021 | +0.8 [+0.3, +1.4]; 213 / 159; p = 0.0059 |

The 6-epoch and 30-epoch RoBERTa runs were made on different accounts, so they have no
paired test; their intervals do not overlap ([0.656, 0.719] and [0.754, 0.810]).

### 4.3 Effect of near-duplicates on RoBERTa (Twitter16)

| Split | Accuracy |
|---|---|
| Random 70/10/20 split, full test set (3 seeds) | 0.799 ± 0.006 |
| Same split, test tweets with no near-duplicate in training | 0.771 |
| Grouped 5-fold | 0.688 |

The random split had near-copies of 16.5% of its test tweets in training.

### 4.4 Story-held-out folds (lexical arm)

| | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| Grouped folds | 0.848 | 0.840 | 0.871 |
| Story-held-out folds | 0.683 [0.651, 0.714] | 0.686 [0.662, 0.709] | 0.701 [0.690, 0.712] |
| Macro-F1, story-held-out | 0.683 | 0.687 | 0.635 |
| Majority class | 0.253 | 0.251 | 0.626 |

PHEME by test fold:

| Test fold (events) | n | Accuracy | Macro-F1 | Majority class in fold |
|---|---|---|---|---|
| Charlie Hebdo | 2,079 | 0.811 | 0.742 | 0.780 |
| Sydney siege | 1,221 | 0.710 | 0.671 | 0.572 |
| Ferguson | 1,143 | 0.760 | 0.608 | 0.752 |
| Ottawa shooting, Gurlitt | 1,028 | 0.658 | 0.639 | 0.517 |
| Germanwings, Prince–Toronto, Putin missing, Ebola–Essien | 954 | 0.423 | 0.369 | 0.637 |

### 4.5 Sensitivity to the story threshold (Twitter15/16, lexical arm)

| Cosine threshold | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|
| Twitter16: stories | 633 | 570 | 498 | 393 | 234 |
| Twitter16: tweets in multi-tweet stories | 279 | 364 | 442 | 570 | 750 |
| Twitter16: accuracy | 0.793 | 0.758 | 0.683 | 0.661 | 0.506 |
| Twitter15: stories | 1,118 | 978 | 828 | 648 | 373 |
| Twitter15: tweets in multi-tweet stories | 574 | 733 | 895 | 1,101 | 1,402 |
| Twitter15: accuracy | 0.785 | 0.743 | 0.686 | 0.595 | 0.548 |

Reproduce with `python scripts/story_threshold_sensitivity.py`.

### 4.6 Training behaviour of the longer RoBERTa run (Twitter16)

Training loss reaches 0.351 by about epoch 10 in every fold; the minimum possible with
label smoothing 0.1 and four classes is 0.349, so the training set is fitted almost
exactly. Validation accuracy stops improving after epochs 6–9 and then stays level
(0.74–0.83 depending on the fold). Best epochs: 11, 8, 14, 25, 12. Mean best validation
macro-F1 is 0.789 and mean test accuracy 0.783, so selecting the epoch on validation
did not inflate the test estimate.

### 4.7 Full pipeline on all three datasets (all configurations in one set of runs)

Run on a second account: lexical, `text_long` and BiGCN were trained there and all
stacks were built from those runs. Pooled accuracy [Wilson 95% interval] · macro-F1.

| Configuration | Twitter16 (n = 818) | Twitter15 (n = 1,490) | PHEME (n = 6,425) |
|---|---|---|---|
| Text branch (≤ 30 epochs) | 0.784 [0.754, 0.810] · 0.782 | 0.770 [0.748, 0.790] · 0.770 | 0.868 [0.860, 0.876] · 0.860 |
| Propagation branch (BiGCN) | 0.811 [0.782, 0.836] · 0.810 | 0.798 [0.777, 0.818] · 0.799 | 0.858 [0.849, 0.867] · 0.850 |
| Lexical branch | 0.848 [0.822, 0.871] · 0.849 | 0.840 [0.821, 0.858] · 0.841 | 0.871 [0.862, 0.878] · 0.862 |
| Lexical + propagation | 0.844 [0.817, 0.867] · 0.844 | 0.840 [0.821, 0.858] · 0.841 | 0.872 [0.863, 0.880] · 0.863 |
| Lexical + text | 0.856 [0.830, 0.878] · 0.856 | 0.844 [0.824, 0.861] · 0.844 | 0.880 [0.872, 0.888] · 0.873 |
| All three branches (proposed) | 0.846 [0.820, 0.869] · 0.847 | 0.852 [0.833, 0.869] · 0.852 | 0.879 [0.871, 0.887] · 0.871 |

Difference in points [95% interval]; only first / only second arm correct; McNemar p.

| Comparison | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| All three − lexical | −0.2 [−1.8, +1.3]; 21 / 23; p = 0.88 | +1.1 [+0.2, +2.1]; 33 / 16; p = 0.021 | +0.8 [+0.3, +1.4]; 213 / 159; p = 0.0059 |
| All three − lexical + text | −1.0 [−1.8, −0.2]; 2 / 10; p = 0.039 | +0.8 [+0.0, +1.6]; 24 / 12; p = 0.065 | −0.1 [−0.5, +0.2]; 76 / 85; p = 0.53 |
| All three − propagation | +3.5 [+1.5, +5.6]; 53 / 24; p = 0.0013 | +5.4 [+3.7, +7.0]; 121 / 41; p < 0.0001 | +2.1 [+1.4, +2.8]; 323 / 190; p < 0.0001 |
| All three − text | +6.2 [+4.0, +8.5]; 72 / 21; p < 0.0001 | +8.2 [+6.3, +10.0]; 164 / 42; p < 0.0001 | +1.1 [+0.5, +1.7]; 225 / 155; p = 0.0004 |
| Lexical + text − lexical | +0.7 [−0.7, +2.2]; 22 / 16; p = 0.42 | +0.3 [−0.2, +0.9]; 11 / 6; p = 0.33 | +1.0 [+0.3, +1.7]; 283 / 220; p = 0.0056 |
| Lexical + text − text | +7.2 [+4.9, +9.5]; 78 / 19; p < 0.0001 | +7.4 [+5.4, +9.3]; 169 / 59; p < 0.0001 | +1.2 [+0.8, +1.7]; 161 / 82; p < 0.0001 |
| Lexical + propagation − lexical | −0.5 [−1.3, +0.3]; 4 / 8; p = 0.39 | +0.0 [−0.2, +0.2]; 1 / 1; p = 1.0 | +0.1 [−0.2, +0.4]; 49 / 42; p = 0.53 |
| Lexical − propagation | +3.8 [+1.8, +5.8]; 52 / 21; p = 0.0004 | +4.2 [+2.5, +6.0]; 121 / 58; p < 0.0001 | +1.2 [+0.6, +1.9]; 282 / 203; p = 0.0004 |
| Text − lexical | −6.5 [−9.2, −3.7]; 41 / 94; p < 0.0001 | −7.0 [−9.1, −5.0]; 70 / 175; p < 0.0001 | −0.2 [−1.1, +0.6]; 365 / 381; p = 0.58 |
| Text − propagation | −2.7 [−5.5, +0.1]; 58 / 80; p = 0.074 | −2.8 [−5.0, −0.6]; 119 / 161; p = 0.014 | +1.0 [+0.1, +1.9]; 464 / 401; p = 0.035 |

**Pipeline against each single branch.** The three-branch pipeline is better than the
text branch and the propagation branch alone on all three datasets (p ≤ 0.0013).
Against the lexical branch it gains +1.1 points on Twitter15 (p = 0.021) and +0.8 on
PHEME (p = 0.0059), and is level on Twitter16 (−0.2, p = 0.88). With a Holm correction
over the three datasets the Twitter15 and PHEME results stay below 0.05 (adjusted
p = 0.043 and 0.018). Over all 8,733 threads the pipeline is right on 267 threads the
lexical branch gets wrong and wrong on 198 it gets right (+0.8 points, exact McNemar
p = 0.0016); PHEME supplies 74% of those threads.

**Which added branch carries the gain.** Lexical + text is above the lexical branch on
all three datasets (+0.7 / +0.3 / +1.0 points) but significantly so only on PHEME
(p = 0.0056); pooled, 316 against 242 threads (+0.8 points, p = 0.0020). Adding the
propagation branch to lexical + text changes the result by −1.0 points on Twitter16
(p = 0.039), +0.8 on Twitter15 (p = 0.065) and −0.1 on PHEME (p = 0.53); pooled, 102
against 107 threads (−0.1 points, p = 0.78). Stacking the propagation branch on the
lexical branch alone gives no gain on any dataset. The measured gain of the pipeline
over the lexical branch therefore comes from the text branch; the propagation branch
has no consistent effect.

**The three-branch stack contains the smaller configurations.** Its weight grid
includes weight 0 for any branch, so the lexical branch alone and lexical + text are
special cases, and its validation score is never below theirs. Choosing among the three
by validation score therefore always returns the three-branch stack. It was fixed as
the proposed pipeline before the Twitter15 and PHEME runs.

**Longer schedule.** Up to 30 epochs raises the text branch on Twitter15 (0.736 with 6
epochs, 0.770 with up to 30; separate runs, intervals [0.713, 0.757] and
[0.748, 0.790]) and makes no difference on PHEME (0.867 and 0.868), where a fold has
about eight times as many training threads as a Twitter16 fold.

**Run-to-run variation.** BiGCN scored 0.811 on Twitter16 in this set of runs and 0.804
in the first (same folds, same seeds, a different Colab runtime); on Twitter15 0.798
and 0.797; on PHEME 0.858 in both. The lexical branch gave the same accuracy in both
sets.

### 4.8 Stacking weights of the three-branch pipeline

Weights chosen on the validation fold (grid step 0.1), per fold, and their mean.

| Dataset | Branch | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|---|---|
| Twitter16 | Lexical | 0.6 | 0.5 | 0.8 | 0.7 | 0.7 | 0.66 |
| | Text | 0.4 | 0.3 | 0.0 | 0.3 | 0.1 | 0.22 |
| | Propagation | 0.0 | 0.2 | 0.2 | 0.0 | 0.2 | 0.12 |
| Twitter15 | Lexical | 0.6 | 0.5 | 0.6 | 0.9 | 1.0 | 0.72 |
| | Text | 0.2 | 0.2 | 0.1 | 0.1 | 0.0 | 0.12 |
| | Propagation | 0.2 | 0.3 | 0.3 | 0.0 | 0.0 | 0.16 |
| PHEME | Lexical | 0.4 | 0.4 | 0.5 | 0.3 | 0.6 | 0.44 |
| | Text | 0.4 | 0.4 | 0.2 | 0.3 | 0.4 | 0.34 |
| | Propagation | 0.2 | 0.2 | 0.3 | 0.4 | 0.0 | 0.22 |

Validation macro-F1 of the chosen weights, mean over folds, against pooled test
macro-F1: 0.869 / 0.847 (Twitter16), 0.869 / 0.852 (Twitter15), 0.881 / 0.871 (PHEME).
The validation figure is higher because the weights and each branch's best model were
chosen on those rows; the test figure is the one reported.

The lexical branch receives about two thirds of the weight on the Twitter datasets and
under half on PHEME, where the three branches are closest in accuracy. The weights
differ considerably between folds of the same dataset (the propagation branch receives
between 0 and 0.4, and in Twitter15 fold 4 the lexical branch receives all of it), so
they are not estimated precisely from one validation fold of 160–300 tweets on the
Twitter datasets.

---

## 5. Limitations

- One training run per fold; seed variance of the neural arms is not measured.
- Story-held-out results exist for the lexical arm only.
- Twitter15/16 stories come from text clustering, not from event ids, and the result
  depends on the threshold (section 4.5).
- The GWO search was run on Twitter16 only.
- The pipeline's gain over the lexical branch is about one point where it is
  significant, the same size as the run-to-run variation seen for BiGCN (0.7 points on
  Twitter16).
- Twitter15 and Twitter16 node features use the public RvNN vocabulary, which was built
  from all threads including test threads.
- The lexical runs in 4.4 and 4.5 used scikit-learn 1.5 on CPU; the grouped-fold
  lexical numbers were the same to three decimals on Colab with scikit-learn 1.9.
