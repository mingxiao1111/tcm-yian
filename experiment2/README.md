# Experiment 2 — Model Comparison for Noise-Resistant Structured Extraction of TCM Medical Cases

This directory contains the complete inference and evaluation code of
Experiment 2: six language models are compared on structured extraction of
Traditional Chinese Medicine (TCM) medical cases from **1,500 full-length,
noise-preserved case texts** (theoretical narration, post-case efficacy
descriptions, and commentator notes included). Each model outputs a JSON
record with seven clinical-element fields: patient information, main
symptoms, tongue manifestations, pulse manifestations, syndrome, treatment
method, and Chinese herbs.

## Models and results

| Model | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|
| **YIAN-4B (ours)** | **92.38** | **93.88** | **93.07** |
| GPT-4.1 mini | 85.51 | 94.82 | 89.88 |
| Qwen3-4B-Instruct-2507 (base) | 84.81 | 90.52 | 87.39 |
| GLM-4-9B-Chat | 83.11 | 88.97 | 85.88 |
| Llama-3.1-8B-Instruct | 76.04 | 85.03 | 79.98 |
| Gemma-3-4B-it | 66.42 | 89.21 | 75.91 |

YIAN-4B is Qwen3-4B-Instruct-2507 fine-tuned with the TCM-YIAN
instruction-tuning dataset. The LoRA adapter is available at
<https://huggingface.co/mark1111222/YIAN-4B>.

## Inference protocol

All open-weight models are served by vLLM 0.11.0 under identical settings:
temperature 0.2, max_tokens 2048, max_model_len 8192, batch chat inference
with the same prompt. YIAN-4B loads the LoRA adapter via `LoRARequest`.
GPT-4.1 mini was evaluated through an OpenAI-compatible API (temperature
0.15, max_tokens 2500).

## Evaluation protocol

- **1,463 samples** are scored (of the 1,500 inputs, 37 whose predictions
  failed JSON parsing are excluded);
- **Field-level relaxed macro-F1** over the seven fields with equal weight;
- Patient information and Chinese herbs: exact match after normalization
  (dosages stripped from herb names);
- Main symptoms, tongue, pulse, syndrome, treatment method: maximum
  one-to-one matching where a prediction fragment counts as correct if it is
  identical to, or mutually contains, a gold fragment.

## Repository layout

```
├── train_mix.yaml                    # LLaMA-Factory training config of YIAN-4B
├── infer_eval2.py                    # vLLM batch inference, one config per model
├── evaluate_macro_f1_comparison.py   # field-level relaxed macro-F1 evaluation
├── data/
│   ├── eval_input_1500.json         # inference inputs (id / instruction / input)
│   └── ground_truth_1463.json       # gold-standard seven-field JSON records
└── results/
    ├── predictions_<model>.json     # raw model outputs, one file per model
    └── relaxed_macro_report.txt     # generated evaluation report
```

## Usage

Reproduce YIAN-4B training with LLaMA-Factory (register the TCM-YIAN dataset
in `dataset_info.json` as `al3` and edit the paths in `train_mix.yaml`):

```bash
llamafactory-cli train train_mix.yaml
```

Run inference (edit the model paths in `CONFIGS` first; the YIAN-4B adapter
can be downloaded from Hugging Face):

```bash
python infer_eval2.py YIAN-4B
```

Score all models present in `results/`:

```bash
python evaluate_macro_f1_comparison.py
```

Missing prediction files are skipped with a warning. The committed
`results/` directory reproduces the table above.

## License

- Code: **Apache-2.0** (see `LICENSE`).
- Data files under `data/` and model outputs under `results/`:
  **CC BY-NC 4.0**, derived from published TCM case-record compilations,
  for non-commercial research use only.

## Citation

```bibtex
@article{yian4b2026,
  title  = {YIAN-4B: A Noise-Resistant Instruction-Tuned Model for Structured Extraction of TCM Medical Cases},
  author = {Zhao, ...},
  journal= {...},
  year   = {2026}
}
```
