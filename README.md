# doc-mt-eval

Code to train and run learned metrics for machine translation (MT) evaluation that work at sentence and document level. Each metric is a regression model on top of a decoder LLM. Given a source text, its machine translation and optionally a reference translation, it predicts a quality score between 0 and 1. The same model can score translations with or without a reference.

This approach also tries to answer the question whether pretrained decoder-only models can be used similarly to pretrained encoder-only models, i.e. as representation extractors for downstream tasks, without the expensive and slow autoregressive text generation at inference time.


Two approaches are implemented:

| Model | Backbone | Input format |
|---|---|---|
| **GemmaItXComet** | `google/gemma-3-1b-it` | chat template; separators `<unused0>` / `<unused1>` |
| **QwenXComet** | `Qwen/Qwen3-Embedding-4B` | plain `Instruct: … Query: …` text; the EOS token is used as separator |

GemmaItXComet is the metric we submitted to the WMT 2026 shared task as Vertical, described in [Nakhlé et al., 2026](https://aclanthology.org/2026.wmt-main.105/) (see [Citation](#citation)). However, the code here differs from the submitted system in two ways:
- **Training stages:** the stages and their data were organised slightly differently for the submission.
- **Bug fix:** in the submitted system, reference tokens leaked into the reference-free (SRC) score, so the reference-free variant wasn't properly trained. This bug is fixed in this code. 
- The [results](#results) reported at the end of this README are therefore not those of the submitted metric but the corrected ones. Because of the differences in training and evaluation data, they are not directly comparable to the scores described in the paper.

## How the model works

The source, the translation and the reference go into a single prompt, with a separator token after the translation and another after the reference:

```
instruction source translation [sep1] reference [sep2]
```

The backbone is a causal (left-to-right) model, so each token only sees the tokens before it. That's why there are two separators:

- the hidden state at **sep1** has seen only source + translation → the **SRC** score (quality estimation, no reference needed);
- the hidden state at **sep2** has seen everything → the **SRC+REF** score.

One forward pass gives both scores. For each variant, the hidden states of all layers are combined with a learned layer-wise attention, under a mask that stops at that variant's separator. The hidden state corresponding to the separator serves as the sequence representation and goes through a small feed-forward regressor. (Due to compute limitations, the model using Qwen as backbone only uses the last layer's hidden state.)

- **Training loss:** `(1 − w) · MSE(SRC) + w · MSE(SRC+REF)`. Rows without a reference only contribute to the SRC term. We set `w` to 0.5 but this can be tuned or learned.


Training runs in two stages, each with its own data:
1. **Stage 1:** sentence-level data and reconsutred paragraphs from WMT human judgements (WMT19–24) plus synthetic data.
2. **Stage 2:** starts from the stage 1 model weights and continues on document-level WMT25 data, synthetic data, and a smaller sample of sentence-level data so that the model keeps its sentence-level quality.

## Repository structure

```
configs/
  data_configs/          example YAML data configs (one per training stage)
  prompts/               prompt template for GemmaItXComet
example-data/            small sample of training/dev data to test the code
metrics/
  binarize_data_xcomet.py   tokenize data for GemmaItXComet
  binarize_data_qwen.py     tokenize data for QwenXComet
  train_GemmaItXComet.py    train GemmaItXComet
  train_QwenXComet.py       train QwenXComet
  modules/                  model classes + layer-wise attention
  inference/inference_regression-model.py   score CSV files with a trained model
  utils/                    language names, data-config loading, training helpers
example-run.sh                  end-to-end test on example-data (both approaches)
```

## Training data format

Each CSV file contains one language pair, with the columns:

| Column | Content |
|---|---|
| `src` | source text (required) |
| `tgt` | machine translation (may be empty) |
| `ref` | reference translation (optional; leave empty if unavailable) |
| `label_01` | human quality score, normalised to [0, 1] (training only) |

Our experiments use human scores normalised to a common 0–1 scale (`label_01`). To train on another column, use `--label_col`. The binarizers check that labels are in [0, 1].

**The language pair is read from the file name**, which must contain it as `xx-yy`, e.g. `wmt24_en-cs_esa.csv` or `synth_ja-zh.csv`. The full language names are inserted into the prompt.

A data config is a YAML file listing absolute CSV paths per split. Environment variables are expanded:

```yaml
train:
  - ${REPO_ROOT}/example-data/wmt19-24/train/wmt24_en-cs_esa.csv
dev_sent:
  - ...
dev_paraf:
  - ...
```

Accepted splits are `train`, `dev`, `dev_sent` and `dev_paraf`. Choose which dev set selects the best checkpoint with `--primary_eval_dataset`.

To use the example configs in `configs/data_configs/`, set `REPO_ROOT` to the repository root first (`export REPO_ROOT=/path/to/doc-mt-eval`). `example-run.sh` does this for you.

## Usage

A complete example can be found in `example-run.sh`. It runs all steps for both models. Its settings (data fraction, epochs, model size) are only meant for testing.

The pipeline is: tokenize the data of each stage (`binarize_data_*.py`), train stage 1, then train stage 2 starting from stage 1 (`--init_from`). To score translations using the trained models, use `inference/inference_regression-model.py`:

```bash
python inference/inference_regression-model.py --archi GemmaItXComet \
    --ckpt path/to/checkpoint --csv data.csv --src_lang en --tgt_lang cs --output scores.csv
```

The output has three score columns: `_srcref` (use it for rows with a reference), `_src` (reference-free score) and `_combined` (an optional blend of the two). To score many files with one model load, use `--csv_manifest`.

## Testing the code

`example-data/` holds a small sample of data (2 language pairs, en-cs and ja-zh, at most 100 rows per file), including some rows without a reference:

| | Train | Sentence-level dev | Document-level dev |
|---|---:|---:|---:|
| Stage 1 | 1,600 | 700 | 200 |
| Stage 2 | 1,174 | 700 | 200 |


## Results

Meta-evaluation with [MTME](https://github.com/google-research/mt-metrics-eval) on a held-out part of the WMT25 data (our models were trained on the rest of WMT25, with no overlap between training and evaluation data). We report SPA (soft pairwise accuracy, system level) and acc23 (pairwise accuracy with tie calibration, segment level). Baseline scores are those reported by MTME.

`(src)` is the reference-free variant score and `(src_ref)` the score with a reference. These results come from the corrected code in this repository, not from the system submitted to WMT 2026 (see above).

### All 14 language pairs with references

| Metric | SPA (system) | acc23 (segment) |
|---|---:|---:|
| Gemini2.5-Pro | 0.8439 | 0.5896 |
| GPT-4.1 | 0.8418 | 0.5014 |
| **QwenXComet (src_ref)** | **0.8412** | **0.6256** |
| TASER | 0.8369 | 0.5873 |
| Claude-4 | 0.8230 | 0.3465 |
| **GemmaItXComet (src_ref)** | **0.8104** | **0.6023** |
| GEMBA-v2 | 0.8048 | 0.5744 |
| CommandA | 0.7989 | 0.3849 |
| **QwenXComet (src)** | **0.7988** | **0.6010** |
| DeepSeek-V3 | 0.7881 | 0.3596 |
| Qwen3-235B | 0.7768 | 0.3749 |
| chrF | 0.7714 | 0.5892 |
| YiSi-1 | 0.7685 | 0.5982 |
| spBLEU | 0.7647 | 0.5899 |
| BERTScore | 0.7539 | 0.5891 |
| BLEU | 0.7488 | 0.5783 |
| **GemmaItXComet (src)** | **0.7154** | **0.5654** |
| MetricX-25 | 0.7100 | 0.5707 |
| COMET22 | 0.6944 | 0.5721 |
| xCOMET-XL (src_ref) | 0.6583 | 0.5466 |
| xCOMET-XL (src) | 0.5680 | 0.5153 |

At segment level, both our models with a reference and QwenXComet without a reference rank above all baselines. At system level, QwenXComet (src_ref) is on par with the best LLM-based metrics, at a fraction of their size and without text generation.

### The 8 language pairs also present in earlier WMT editions

Half of the WMT25 language pairs were new in 2025. Our models saw these languages in training (WMT25 data), while learned baselines such as COMET22 and MetricX-25 were trained on earlier WMT editions only. While no real data leakage is present, we provide an additional result table for a stricter comparison, where we only consider the 8 language pairs that also appear in earlier editions:

| Metric | SPA (system) | acc23 (segment) |
|---|---:|---:|
| Gemini2.5-Pro | 0.8583 | 0.5814 |
| TASER | 0.8537 | 0.5814 |
| GPT-4.1 | 0.8502 | 0.4823 |
| Claude-4 | 0.8493 | 0.3204 |
| GEMBA-v2 | 0.8346 | 0.5847 |
| DeepSeek-V3 | 0.8325 | 0.3306 |
| **QwenXComet (src_ref)** | **0.8271** | **0.6016** |
| CommandA | 0.8180 | 0.3582 |
| Qwen3-235B | 0.8175 | 0.3696 |
| **QwenXComet (src)** | **0.7871** | **0.5763** |
| **GemmaItXComet (src_ref)** | **0.7777** | **0.5727** |
| chrF | 0.7321 | 0.5721 |
| YiSi-1 | 0.7320 | 0.5815 |
| spBLEU | 0.7297 | 0.5717 |
| MetricX-25 | 0.7225 | 0.5861 |
| BLEU | 0.7085 | 0.5590 |
| BERTScore | 0.7056 | 0.5643 |
| COMET22 | 0.6748 | 0.5694 |
| xCOMET-XL (src_ref) | 0.6731 | 0.5667 |
| **GemmaItXComet (src)** | **0.6703** | **0.5350** |
| xCOMET-XL (src) | 0.5759 | 0.5323 |

We can see that on these language pairs our metrics rank lower, however QwenXComet (src_ref) is still the best metric at segment level. Therefore, its lead does not come only from the exposure to new pairs. At system level, the prompted LLMs are stronger.

Overall, these results suggest that the pre-training of decoder-only LLMs can be leveraged for downstream tasks by using them as representation extractors, without autoregressive text generation, at least for MT evaluation.

## Citation

If you use this code or GemmaItXComet, please cite:

```bibtex
@inproceedings{nakhle-etal-2026-vertical,
    title = "Vertical: Quality Estimation of Machine Translation from Universit{\'e} Grenoble Alpes",
    author = "Nakhl{\'e}, Mariam  and
      Qader, Raheel  and
      Dinarelli, Marco  and
      Blanchon, Herv{\'e}",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Eleventh Conference on Machine Translation",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wmt-main.105/",
    doi = "10.18653/2026.wmt-main.105",
    pages = "1716--1722",
}
```
