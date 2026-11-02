# -*- coding: utf-8 -*-
"""不依赖模型的单元自检：确认评测脚本里的解析、比对、打分逻辑是对的。

做法：把 evaluate.py 里不需要 GPU 的部分（前一半）单独 exec 出来，
     用构造的假输出喂给 score_rows，验证指标算得对不对。
"""

import json

src = open("evaluate.py", encoding="utf-8").read()
head = src.split("@torch.no_grad()")[0]
for line in ["import torch", "from peft import PeftModel",
             "from transformers import AutoModelForCausalLM, AutoTokenizer"]:
    head = head.replace(line, "")
ns = {"__file__": "evaluate.py", "__name__": "evaluate_head"}
exec(compile(head, "evaluate_head", "exec"), ns)

parse_json, field_equal, macro_f1 = ns["parse_json"], ns["field_equal"], ns["macro_f1"]
score_rows = ns["score_rows"]

# 1. 正常 JSON
assert parse_json('{"category": "退款"}') == {"category": "退款"}
# 2. 带 ```json 包裹
assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
# 3. 前后有解释文字
assert parse_json('结果如下 {"a": 2} 以上') == {"a": 2}
# 4. 根本不是 JSON
assert parse_json("我不知道") is None
assert parse_json('{"a": 1') is None

# 5. 字段比对
assert field_equal("amount", 699.0, 699)
assert field_equal("amount", "699.00", 699)
assert not field_equal("amount", "abc", 699)
assert field_equal("category", " 退款 ", "退款")
assert field_equal("escalate", True, True)
assert not field_equal("escalate", False, True)
assert field_equal("order_id", None, None)

# 6. macro F1 手算：退款 tp=1 fp=1 fn=1 -> 0.5；退货 tp=0 fn=1 -> 0.0；macro=0.25
pairs = [("退款", "退款"), ("退货", "退款"), ("退款", "退货")]
assert abs(macro_f1(pairs) - 0.25) < 1e-6, macro_f1(pairs)

# 7. score_rows 端到端：3 条样本，1 条完全正确、1 条 JSON 坏了、1 条部分错
gold = json.dumps({"category": "退款", "sentiment": "负面", "escalate": False,
                   "entities": {"order_id": "A2001", "product": "耳机", "amount": 299}},
                  ensure_ascii=False)
rows = [{"output": gold, "id": "s1"}, {"output": gold, "id": "s2"}, {"output": gold, "id": "s3"}]
texts = [
    gold,                                                     # 完全正确
    "抱歉，我无法处理",                                        # JSON 不合法
    '{"category": "退款", "sentiment": "中性", "escalate": false, '
    '"entities": {"order_id": "A2001", "product": "耳机", "amount": 299.0}}',  # 情绪错
]
summary, details = score_rows(rows, texts)
assert abs(summary["json_valid"] - 2 / 3) < 1e-9, summary["json_valid"]
assert abs(summary["all_fields_correct"] - 1 / 3) < 1e-9, summary["all_fields_correct"]
assert abs(summary["field_accuracy"]["category"] - 2 / 3) < 1e-9
assert abs(summary["field_accuracy"]["sentiment"] - 1 / 3) < 1e-9
assert abs(summary["field_accuracy"]["amount"] - 2 / 3) < 1e-9   # 299.0 与 299 应算对
assert len(details) == 3 and details[1]["pred"] is None

print("evaluate.py 逻辑自检通过：解析 / 比对 / macro-F1 / 端到端打分 全部符合预期")
