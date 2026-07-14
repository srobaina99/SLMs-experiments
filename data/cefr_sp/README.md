# CEFR-SP checkpoint (Arase et al.)

Official pretrained contrastive sentence-difficulty model from:

> Yuki Arase, Satoru Uchida, and Tomoyuki Kajiwara. 2022.
> CEFR-Based Sentence Difficulty Annotation and Assessment.
> EMNLP 2022. https://aclanthology.org/2022.emnlp-main.416/

## Provenance

| Field | Value |
|-------|-------|
| Zenodo DOI | [10.5281/zenodo.7234096](https://doi.org/10.5281/zenodo.7234096) |
| File | `level_estimator.ckpt` (~1.2 GB) |
| MD5 | `2448ff49f6e8a9c504ac8ba02116e043` |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Architecture | `bert-base-cased`, `lm_layer=11`, `num_prototypes=3`, 6-way A1–C2 |
| Code | Vendored thin copies under `src/slm_experiments/evaluation/cefr_sp_vendor/` |

**Do not commit the checkpoint.** It is gitignored (`data/cefr_sp/*.ckpt`).

## Download

```bash
./venv/bin/python scripts/download_cefr_sp_ckpt.py
```

Optional extras (torch / transformers / pytorch-lightning):

```bash
./venv/bin/pip install -e ".[cefr-sp]"
```

Environment override: set `CEFR_SP_CKPT` or pass `--cefr-sp-ckpt` / `ExperimentConfig.cefr_sp_ckpt_path`.

Scoring is **on by default** for experiment runs. Disable with `--no-enable-cefr-sp` or `enable_cefr_sp=False`.

## Tokenization

Training encodes **whitespace-split words** with `is_split_into_words=True` (not raw sentence strings). The façade in `evaluation/cefr_sp.py` matches that path.

## Load notes

Do **not** use the upstream `level_estimator.py` CLI for inference: when `--pretrained` is set it loads the checkpoint and then **unconditionally reinstantiates** a fresh untrained `LevelEstimaterContrastive`, discarding the loaded weights.

### Zenodo load quirks

`CefrSpScorer._ensure_loaded` `torch.load`s the Zenodo file, fixes hparams, constructs `LevelEstimaterContrastive`, then `load_state_dict(..., strict=True)`.

**1. Trainer-local `pretrained_model` path**

```text
hyper_parameters.pretrained_model = '../pretrained_model/bert-base-cased/'
```

That relative path does not exist outside the original training machine. The scorer overrides it to the Hub id `bert-base-cased` before constructing the module (first run downloads the base encoder from Hugging Face; the ~1.2 GB checkpoint then overwrites encoder + prototype weights).

**2. Obsolete `position_ids` buffer**

The checkpoint state_dict includes `lm.embeddings.position_ids` from an older `transformers` BertModel. Current BertModel does not register that buffer, so a naïve `strict=True` load fails on an unexpected key. The scorer drops that single key before `load_state_dict` so other mismatches still raise.

Extras pin `pytorch-lightning>=2.0`. If load fails on your torch/Lightning/transformers combo, see the error from `CefrSpScorer` and pin compatible 2.x releases.
