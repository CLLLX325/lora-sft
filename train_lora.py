# -*- coding: utf-8 -*-
"""用 LoRA 微调 Qwen2.5-0.5B-Instruct，让它稳定输出固定格式的工单 JSON。

为什么选 0.5B：
    RTX 4060 Laptop 只有 8GB 显存，训练 7B 需要 4bit 量化 + 各种省显存技巧，
    第一次做微调很容易卡在 OOM 上。0.5B 全精度 LoRA 只要 ~2GB 显存，
    10 分钟能跑完一轮，能把"数据 → 训练 → 评测"这条链路走通。

为什么用 LoRA 而不是全参微调：
    全参要存优化器状态（约 12 倍参数量），0.5B 也要 6GB+；LoRA 只训练
    attention 里的 q/k/v/o 四个投影矩阵，可训练参数约 1.6M（0.3%），
    显存 2GB 出头，adapter 文件只有几 MB。

关键细节：
    1. labels 只保留 assistant 回答部分，prompt 部分置 -100（不参与 loss）。
       否则模型会去学"复述用户问题"。
    2. 答案末尾拼 eos，模型才知道什么时候停。
    3. 训练集和测试集按模板切分，测试集模板训练时没见过（见 data_gen.py）。
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_base_model():
    """按优先级找基座模型目录：环境变量 → 项目内 models/ → 上一级 models/。

    原来这里写死了 C:\\deepwork\\models\\...，别人 clone 下来必然报错，
    所以改成三级兜底：优先用 BASE_MODEL 环境变量，找不到再退回相对路径。
    """
    candidates = []
    if os.environ.get("BASE_MODEL"):
        candidates.append(os.environ["BASE_MODEL"])
    candidates.append(os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct"))
    candidates.append(os.path.abspath(os.path.join(ROOT, "..", "models", "Qwen2.5-0.5B-Instruct")))
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]        # 都不存在时返回第一个，让报错信息指向它


BASE_MODEL = _find_base_model()
OUT_DIR = os.path.join(ROOT, "lora-out")
MAX_LEN = 512


def load_rows(name):
    path = os.path.join(ROOT, "data", f"{name}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_features(rows, tokenizer):
    """把一条样本转成 input_ids / labels。

    prompt 部分用 chat template 渲染（和推理时完全一致），
    answer 部分单独 tokenize 后拼在后面，labels 前面填 -100。
    """
    features = []
    for row in rows:
        messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        # 坑：transformers 5.x 里 tokenize=True 返回的是 BatchEncoding（dict），
        # 旧版本直接返回 list。这里两种情况都兼容。
        prompt_ids = list(encoded["input_ids"]) if hasattr(encoded, "keys") else list(encoded)
        answer_ids = tokenizer(row["output"], add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:MAX_LEN]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]
        features.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        })
    return features


class PadCollator:
    """把变长序列 pad 到 batch 内最长，并保持 labels 的 -100 不被 pad 覆盖。"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, masks = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }


def load_base(path):
    """加载基座模型。transformers 5.x 用 dtype，旧版本用 torch_dtype，这里两个都兼容。"""
    try:
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, trust_remote_code=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, trust_remote_code=True)


def main():
    print("加载分词器与基座模型：", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = load_base(BASE_MODEL)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_rows("train")
    val_rows = load_rows("val")
    train_ds = Dataset.from_list(build_features(train_rows, tokenizer))
    val_ds = Dataset.from_list(build_features(val_rows, tokenizer))
    print(f"训练 {len(train_ds)} 条 / 验证 {len(val_ds)} 条")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,   # transformers 5.x 去掉了 warmup_ratio，改用步数
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    trainer.train()

    # 只保存 LoRA adapter（几 MB），不存整个模型
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    print("adapter 已保存到", OUT_DIR)

    metrics = trainer.evaluate()
    print("验证集 loss:", round(metrics["eval_loss"], 4))
    with open(os.path.join(OUT_DIR, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"eval_loss": metrics["eval_loss"],
                   "train_rows": len(train_ds),
                   "val_rows": len(val_ds)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
