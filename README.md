# TCM-YIAN

Code and evaluation resources for **TCM-YIAN**: an instruction-tuning
dataset and model for noise-resistant structured extraction of Traditional
Chinese Medicine (TCM) medical cases.

- **TCM-YIAN dataset** — 12,500 instructions in four tasks over seven
  clinical element categories (patient information, main symptoms, tongue
  manifestations, pulse manifestations, syndrome, treatment method, and
  Chinese herbs), derived from published TCM case-record compilations
  (Zenodo, CC BY-NC 4.0: https://doi.org/10.5281/zenodo.23053182).
- **TCED entity database** — the clinical element corpus underlying
  TCM-YIAN (Zenodo, CC BY 4.0:
  https://doi.org/10.5281/zenodo.23053127).
- **YIAN-4B model** — Qwen3-4B-Instruct-2507 fine-tuned with TCM-YIAN via
  LoRA: <https://huggingface.co/mark1111222/YIAN-4B>.

## Contents

| Directory | Description |
|---|---|
| `experiment1/` | Ablation study on 703-case development set: inference inputs, gold standard, BIO sequence-labeling data (RoBERTa-BiLSTM-CRF baseline), and evaluation results |
| `experiment2/` | Six-model comparison on 1,500 noisy case texts: vLLM inference, field-level relaxed macro-F1 evaluation, predictions and reports |

## License

- Code: **Apache-2.0**.
- Data and model outputs: **CC BY-NC 4.0** (derived from published TCM
  case-record compilations, non-commercial research use only).

## Citation

```bibtex
@article{yian4b2026,
  title  = {YIAN-4B: A Noise-Resistant Instruction-Tuned Model for Structured Extraction of TCM Medical Cases},
  author = {Zhao, ...},
  journal= {...},
  year   = {2026}
}
```
