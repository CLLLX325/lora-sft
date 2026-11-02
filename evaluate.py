# -*- coding: utf-8 -*-
"""评测：让模型在测试集上生成 JSON，逐字段比对。

为什么要分字段算，而不是只算"整条对不对"：
    整条对的比例很低（8 个字段全中才算），信息量小，看不出改进在哪。
    分字段能看出"模型到底学会了什么"：
    - 类目、情绪、是否升级 是分类字段，用 F1 衡量
    - 订单号、商品名、金额 是抽取字段，用精确匹配衡量
    - JSON 合法率单独看，衡量格式遵从度

用法：
    python evaluate.py --tag base            # 基座模型，不挂 adapter
    python evaluate.py --tag lora            # 挂 lora-out 里的 adapter
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_base_model():
    """按优先级找基座模型目录：环境变量 → 项目内 models/ → 上一级 models/。

    和 train_lora.py 里同样的三级兜底，避免写死本机绝对路径。
    """
    candidates = []
    if os.environ.get("BASE_MODEL"):
        candidates.append(os.environ["BASE_MODEL"])
    candidates.append(os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct"))
    candidates.append(os.path.abspath(os.path.join(ROOT, "..", "models", "Qwen2.5-0.5B-Instruct")))
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


BASE_MODEL = _find_base_model()
BATCH_SIZE = 8
MAX_NEW_TOKENS = 96
CLASS_FIELDS = ["category", "sentiment", "escalate"]
ENTITY_FIELDS = ["order_id", "product", "amount"]


def load_rows(name):
    with open(os.path.join(ROOT, "data", f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json(text):
    """从模型输出里抠出 JSON 对象，容忍 ```json 包裹和前后噪声。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def field_equal(name, pred, gold):
    """逐个字段比对。金额允许 0.01 的误差，其余要求字面相等（去空格）。"""
    if pred is None and gold is None:
        return True
    if name == "amount":
        try:
            return abs(float(pred) - float(gold)) < 0.01
        except (TypeError, ValueError):
            return False
    if isinstance(gold, bool):
        return pred is gold or str(pred).lower() == str(gold).lower()
    return str(pred).replace(" ", "") == str(gold).replace(" ", "")


def _norm(value):
    """把布尔/字符串统一成可比较的字符串，避免 True 和 "true" 被当成两个类别。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def macro_f1(pairs):
    """pairs: [(pred, gold)]，返回 macro F1（只统计 gold 中出现的类别）。"""
    pairs = [(_norm(p), _norm(g)) for p, g in pairs]
    labels = sorted({g for _, g in pairs})
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in pairs if p == lab and g == lab)
        fp = sum(1 for p, g in pairs if p == lab and g != lab)
        fn = sum(1 for p, g in pairs if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def score_rows(rows, texts):
    """把"文本输出 + 标准答案"变成指标。抽出来是为了能在不加载模型的情况下自测。"""
    n = len(rows)
    valid = 0
    field_hits = defaultdict(int)
    cls_pairs = {name: [] for name in CLASS_FIELDS}
    all_correct = 0
    details = []

    for idx, (row, text) in enumerate(zip(rows, texts)):
        gold = json.loads(row["output"])
        pred = parse_json(text)
        rec = {"id": row.get("id", "t%03d" % idx), "raw": text, "pred": pred, "gold": gold}
        details.append(rec)
        if pred is None:          # JSON 都不合法，所有字段算错
            continue
        valid += 1
        ok = True
        for name in CLASS_FIELDS:
            hit = field_equal(name, pred.get(name), gold.get(name))
            field_hits[name] += hit
            ok = ok and hit
            cls_pairs[name].append((pred.get(name), gold.get(name)))
        pred_ent = pred.get("entities") or {}
        gold_ent = gold.get("entities") or {}
        for name in ENTITY_FIELDS:
            hit = field_equal(name, pred_ent.get(name), gold_ent.get(name))
            field_hits[name] += hit
            ok = ok and hit
        if ok:
            all_correct += 1

    names = CLASS_FIELDS + ENTITY_FIELDS
    return {
        "n": n,
        "json_valid": valid / n,
        "field_accuracy": {name: field_hits[name] / n for name in names},
        "macro_f1": {name: macro_f1(cls_pairs[name]) for name in CLASS_FIELDS},
        "all_fields_correct": all_correct / n,
    }, details


def load_model(path):
    """加载基座模型。transformers 5.x 用 dtype，旧版本用 torch_dtype，这里两个都兼容。"""
    try:
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, trust_remote_code=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, trust_remote_code=True)


@torch.no_grad()
def generate(model, tokenizer, rows):
    """批量生成。用左 padding，只解码新 token。"""
    tokenizer.padding_side = "left"
    outputs = []
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        prompts = []
        for row in batch:
            messages = [
                {"role": "system", "content": row["instruction"]},
                {"role": "user", "content": row["input"]},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False))
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(model.device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print("  已生成 %d/%d" % (min(i + BATCH_SIZE, len(rows)), len(rows)), flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="结果文件名标记，如 base / lora")
    ap.add_argument("--adapter", default="lora-out", help="adapter 目录，none 表示不挂")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于冒烟测试")
    args = ap.parse_args()

    rows = load_rows("test")
    if args.limit:
        rows = rows[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(BASE_MODEL).to("cuda")
    if args.adapter != "none":
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, args.adapter))
        print("已挂载 adapter:", args.adapter)
    model.eval()

    print("测试集 %d 条，开始生成..." % len(rows))
    texts = generate(model, tokenizer, rows)
    summary, details = score_rows(rows, texts)
    summary["tag"] = args.tag
    summary["adapter"] = args.adapter

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(os.path.join(ROOT, "results-%s.json" % args.tag), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
    print("结果已写入 results-%s.json" % args.tag)


if __name__ == "__main__":
    main()
