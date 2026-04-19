# Working Notes

## Shopee Product Matching

Kaggle competition:

- `shopee-product-matching`
- Kernel: `kevinxuj/shopee-phash-title-tfidf-baseline`

Best completed scored submission so far:

```text
description:  phash char and word tfidf baseline
status:       COMPLETE
publicScore:  0.660
privateScore: 0.648
```

Image-embedding submission:

```text
description: phash title tfidf image embedding baseline
status:      COMPLETE
publicScore: 0.509
privateScore: 0.505
```

The handcrafted image embedding hurt the leaderboard score, so it should stay
disabled by default unless it is replaced by a stronger pretrained visual
embedding.

Previous text-only submission:

```text
description: phash and title tfidf kernel baseline
status:      COMPLETE
publicScore: 0.654
privateScore: 0.644
```

Word-TF-IDF submission:

```text
description: phash char and word tfidf baseline
status:      COMPLETE
publicScore: 0.660
privateScore: 0.648
change:      exact phash + char TF-IDF threshold 0.72 + word TF-IDF threshold 0.66
```

The winning leaderboard score shown by Kaggle is `0.780`.

## Current Algorithm

The active Shopee baseline predicts product matches by combining three
independent signals:

```text
final matches =
  self match
  + exact image_phash matches
  + character title TF-IDF nearest-neighbor matches
  + word title TF-IDF nearest-neighbor matches
```

### 1. Exact image_phash matches

Rows with the same `image_phash` are grouped together. If products `A` and `B`
share the same perceptual hash, each is predicted as a match for the other.
Every product is always matched to itself.

This signal is precise but only catches near-identical images.

### 2. Character title TF-IDF matches

Product titles are normalized by lowercasing, replacing non-alphanumeric
characters with spaces, collapsing whitespace, and filling empty titles with
`missing`.

The baseline builds character n-gram TF-IDF features:

```text
analyzer:      char_wb
ngram_range:   3 to 5
max_features:  100000
```

Cosine nearest neighbors are computed over title vectors. A neighbor is added
as a match when similarity is at least `0.72`.

This catches products with similar title spelling or wording.

### 3. Word title TF-IDF matches

The improvement branch adds a second TF-IDF matcher using word unigrams and
bigrams:

```text
analyzer:      word
ngram_range:   1 to 2
max_features:  100000
```

Cosine nearest neighbors are computed over these word-level vectors. A neighbor
is added when similarity is at least `0.66`.

This branch catches near-duplicate titles where whole-word overlap is strong
but character n-gram similarity misses the row. On two 8k-row train samples,
the gain was small but positive.

### 4. Lightweight image embeddings

The image embedding is self-contained and does not use pretrained model
weights. Each product image is converted into a compact vector:

1. Open the image with Pillow.
2. Convert to RGB and respect EXIF orientation.
3. Resize into a `16x16` white canvas.
4. Flatten the RGB pixels.
5. Add RGB channel means.
6. Add RGB channel standard deviations.
7. Add per-channel color histograms.
8. L2-normalize the vector.

Cosine nearest neighbors were computed over those vectors. A neighbor was added
as a match when similarity was at least `0.985`.

This can catch visually similar products when `image_phash` differs in theory,
but the Kaggle score dropped from `0.654/0.644` to `0.509/0.505`. It is much
weaker than CLIP, EfficientNet, ViT, or other pretrained vision models and is
currently opt-in only in `src/shopee_baseline.py`.

### 5. Match union

All match sources are unioned.

Example:

```text
phash says: A -> A B
char says:  A -> A C
word says:  A -> A D

final:      A -> A B C D
```

The final Kaggle output is:

```csv
posting_id,matches
A,A B C D
```

## Local Evaluation

The local script `src/shopee_baseline.py` can evaluate on train rows using
`label_group`. It computes a mean set-F1 metric matching the competition style.

Diagnostics are capped by default:

```text
diagnostics_limit: 5000
```

This keeps local checks fast while still giving a rough signal.

## Kaggle Kernel

The Kaggle kernel lives under:

```text
kaggle_shopee_kernel/
```

Important files:

- `kernel.py`: self-contained Kaggle script.
- `kernel-metadata.json`: Kaggle kernel metadata.

The kernel discovers Shopee files under Kaggle's mounted input directories and
writes:

```text
/kaggle/working/submission.csv
```

## Next Improvements

Best next steps for better score:

1. Build a pairwise reranker over pHash, title, and image candidate pairs.
2. Tune title and image thresholds on full local train F1 and public score.
3. Add multilingual text embeddings for semantic title similarity.
4. Use multilingual text embeddings for Indonesian/English title similarity.
5. Train a Siamese or metric-learning model from `label_group` pairs.
6. Build a candidate set from many weak signals, then tune per-signal thresholds
   on local F1.

The biggest expected jump should come from real pretrained image embeddings,
not from the current handcrafted image vector.

## ResNet18 Image Candidate

The ResNet18 experiment uses a private Kaggle dataset for offline weights:

```text
dataset: kevinxuj/shopee-resnet18-weights
file:    resnet18-f37072fd.pth
```

The submitted kernel enables GPU and falls back to the current text+pHash
baseline if the weights or test images are unavailable.

Small local sanity grid on a 64-row downloaded image subset:

```text
text+pHash baseline mean_f1:      0.770
ResNet18 threshold 0.85 mean_f1:  0.853
mean match count at 0.85:         2.19
```

Kaggle scored the ResNet18 submission as a tie with the current best:

```text
description: phash title tfidf resnet18 image baseline
status:      COMPLETE
publicScore: 0.660
privateScore: 0.648
```

This means ResNet18 did not hurt, but it also did not improve the leaderboard
score over the lighter pHash + character TF-IDF + word TF-IDF baseline.

## CLIP ViT-B/32 Image Candidate

The CLIP experiment uses a private Kaggle dataset for offline OpenAI CLIP JIT
weights:

```text
dataset: kevinxuj/shopee-clip-vit-b32-weights
file:    ViT-B-32.pt
```

The kernel enables GPU because the OpenAI JIT graph expects CUDA. If CLIP
loading fails, the kernel falls back to the current pHash + character TF-IDF +
word TF-IDF baseline.

Small local sanity grid on a 79-row downloaded image subset:

```text
text+pHash baseline mean_f1:         0.821
CLIP threshold 0.75 mean_f1:         0.905
mean match count at 0.75:            2.34
```

Kaggle scored the CLIP submission as another tie with the current best:

```text
description: phash title tfidf clip vit b32 image baseline
status:      COMPLETE
publicScore: 0.660
privateScore: 0.648
```

CLIP was stronger than ResNet18 on the small local image subset, but it did not
move the leaderboard score. The next step should shift from adding raw neighbor
sources to ranking candidate pairs more carefully.
