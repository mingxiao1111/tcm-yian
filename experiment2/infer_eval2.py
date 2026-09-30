#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: vLLM batch inference (1,500 noisy case texts -> seven-element JSON predictions).

Usage:
  python infer_eval2.py YIAN-4B     # fine-tuned model (base + LoRA adapter)
  python infer_eval2.py Qwen3-4B    # base model
  python infer_eval2.py GLM4-9B
  python infer_eval2.py Llama3.1-8B
  python infer_eval2.py Gemma3-4B

GPT-4.1 mini was evaluated via the OpenAI API with the same prompt and
decoding settings (see README).

Output: results/predictions_<model>.json, format {"<eval_id>": "<model output text>"}.

Note: local model paths below refer to an AutoDL GPU instance layout; replace
them with your own paths. The YIAN-4B adapter can be downloaded from
https://huggingface.co/mark1111222/YIAN-4B and passed as the lora path.
"""

import json
import sys
import time

EVAL_INPUT = "data/eval_input_1500.json"
OUT_DIR = "results"

CONFIGS = {
    "YIAN-4B": {
        "model": "/root/autodl-tmp/Qwen3-4B-Instruct-2507",
        "lora": "/root/autodl-tmp/output/tcm_yian_mix",
        "nothink": True,
    },
    "Qwen3-4B": {"model": "/root/autodl-tmp/Qwen3-4B-Instruct-2507", "nothink": True},
    "GLM4-9B": {"model": "/root/autodl-tmp/GLM-4-9B-Chat"},
    "Llama3.1-8B": {"model": "/root/autodl-tmp/Llama-3.1-8B-Instruct"},
    "Gemma3-4B": {"model": "/root/autodl-tmp/gemma-3-4b-it"},
}


def main():
    name = sys.argv[1]
    cfg = CONFIGS[name]

    from vllm import LLM, SamplingParams

    data = json.load(open(EVAL_INPUT, encoding="utf-8"))
    msgs = [[{"role": "user", "content": r["instruction"] + "\n" + r["input"]}] for r in data]

    kwargs = dict(
        model=cfg["model"],
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )
    if cfg.get("lora"):
        kwargs.update(enable_lora=True, max_lora_rank=8)

    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.2, max_tokens=2048)

    chat_kwargs = {}
    if cfg.get("lora"):
        from vllm.lora.request import LoRARequest
        chat_kwargs["lora_request"] = LoRARequest("yian", 1, cfg["lora"])
    if cfg.get("nothink"):
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.time()
    outs = llm.chat(msgs, sp, **chat_kwargs)

    preds = {}
    for rec, o in zip(data, outs):
        preds[rec["id"]] = o.outputs[0].text

    out_path = f"{OUT_DIR}/predictions_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=1)
    n_nonempty = sum(1 for v in preds.values() if v.strip())
    print(json.dumps({
        "model": name, "samples": len(preds), "nonempty": n_nonempty,
        "elapsed_min": round((time.time() - t0) / 60, 1), "output": out_path,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
