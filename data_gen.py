# -*- coding: utf-8 -*-
"""生成"售后台单 → 结构化 JSON"的训练/验证/测试数据。

为什么用模板生成而不是人工标注：
    1. 真实工单拿不到（隐私），自己写 600 条不现实；
    2. 模板能把"标签怎么来的"写死——类目、情绪、是否升级都由模板决定，
       不会出现"人工标注两人不一致"的问题；
    3. 关键是**按模板分组切分**：测试集用的模板训练时完全没见过，
       这样测出来的指标才不是"背题"。

标签体系（4 个字段）：
    category   类目：退款 / 退货 / 换货 / 物流 / 发票 / 投诉 / 咨询
    sentiment  情绪：负面 / 中性 / 正面
    escalate   是否升级：true / false（涉及金额超过 2000、人身安全、监管投诉时为 true）
    entities   关键实体：order_id / product / amount（没有就写 null）

产物：
    data/train.jsonl / val.jsonl / test.jsonl，每行 {"instruction":..,"output":..,"template":..}
"""

import io
import json
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

CATEGORIES = ["退款", "退货", "换货", "物流", "发票", "投诉", "咨询"]

# ---------------------------------------------------------------- 模板
# 每个模板：(模板文本, 类目, 情绪, 是否升级)
# 占位符：{order} {product} {amount} {city} {days} {date}
TEMPLATES = [
    # ---- 退款 ----
    ("订单{order}的{product}我申请退款已经{days}天了，钱还没到账，到底什么时候能退？", "退款", "负面", False),
    ("{order}这个单子我不要了，直接退钱吧，{product}还没拆封。", "退款", "中性", False),
    ("请问{product}退款一般几个工作日到账？我的订单是{order}。", "退款", "中性", False),
    ("{order}退款失败三次了，银行说没收到，你们平台是不是在拖？金额{amount}元。", "退款", "负面", True),
    ("退款已经到账了，谢谢，{order}这单处理得挺快。", "退款", "正面", False),
    ("我买的{product}降价了，能退差价吗？订单{order}，买的时候{amount}元。", "退款", "中性", False),
    ("{order}这单只退款不退货可以吗？东西有问题但我懒得寄回去。", "退款", "中性", False),
    # ---- 退货 ----
    ("{product}收到就坏了，我要退货，订单号{order}，请尽快安排。", "退货", "负面", False),
    ("{order}的{product}不想要了，七天无理由还能退吗？已经签收{days}天。", "退货", "中性", False),
    ("退货寄回去了，{order}这个单子商家一直不确认收货，{days}天了。", "退货", "负面", False),
    ("{product}尺码不对，想退货重买，订单{order}。", "退货", "中性", False),
    ("{order}退货的运费谁出？商品{amount}元，运费要我自己付吗？", "退货", "中性", False),
    ("东西退回去了钱也退了，体验不错，{order}。", "退货", "正面", False),
    # ---- 换货 ----
    ("{product}有划痕，想换一个同款的，订单{order}。", "换货", "负面", False),
    ("{order}的{product}颜色发错了，能换正确的颜色吗？", "换货", "中性", False),
    ("换货申请提交{days}天了还没发货，{order}，能不能催一下？", "换货", "负面", False),
    ("{product}用了一周就出故障，要求换新，订单{order}，金额{amount}元。", "换货", "负面", False),
    # ---- 物流 ----
    ("{order}的包裹在{city}停了{days}天没动，是不是丢了？", "物流", "负面", False),
    ("{product}什么时候发货？下单{order}已经{days}天了。", "物流", "中性", False),
    ("快递员把{product}放快递柜了，我没收到取件码，订单{order}。", "物流", "负面", False),
    ("{order}显示已签收但我根本没收到，{amount}元的东西，麻烦查一下。", "物流", "负面", True),
    ("物流很快，第二天就到了，{order}。", "物流", "正面", False),
    ("{city}这边能不能送货上门？{product}太重了自己搬不动，订单{order}。", "物流", "中性", False),
    # ---- 发票 ----
    ("{order}的发票能开公司抬头吗？金额{amount}元。", "发票", "中性", False),
    ("发票开错了，公司名称写错一个字，{order}能重开吗？", "发票", "负面", False),
    ("{product}的发票{days}天了还没收到，订单{order}。", "发票", "负面", False),
    ("电子发票在哪里下载？{order}这单需要报销。", "发票", "中性", False),
    # ---- 投诉 ----
    ("客服态度太差了，我问了三遍都没人理，订单{order}，要投诉。", "投诉", "负面", True),
    ("{product}用了之后皮肤过敏，这是安全问题，{order}，要求赔偿{amount}元。", "投诉", "负面", True),
    ("我要向市场监管部门投诉你们，{order}这单拖了{days}天没人处理。", "投诉", "负面", True),
    ("商家虚假宣传，{product}和页面描述完全不一样，订单{order}。", "投诉", "负面", False),
    ("上次投诉的问题已经解决了，处理结果满意，谢谢。", "投诉", "正面", False),
    ("{city}网点的服务太差了，{order}这个件被扔在门口。", "投诉", "负面", False),
    # ---- 咨询 ----
    ("请问{product}支持七天无理由吗？", "咨询", "中性", False),
    ("{amount}元以上的订单是不是包邮？", "咨询", "中性", False),
    ("会员等级怎么升？我现在想问V3需要消费多少。", "咨询", "中性", False),
    ("{product}的保修期是多久？订单{order}。", "咨询", "中性", False),
    ("跨境订单清关要多久？{order}这单在海关停了{days}天。", "咨询", "中性", False),
    ("生鲜坏果怎么赔？我买的{product}，订单{order}。", "咨询", "中性", False),
    ("价保怎么申请？{product}我买的时候{amount}元，现在便宜了。", "咨询", "中性", False),
    ("以旧换新怎么估价？旧{product}想换新的，订单{order}。", "咨询", "中性", False),
    ("优惠券过期了能补吗？", "咨询", "中性", False),
    ("{order}这单可以改地址吗？还没发货。", "咨询", "中性", False),
]

PRODUCTS = ["无线降噪耳机", "机械键盘", "笔记本电脑", "空气炸锅", "智能手表", "人体工学椅",
            "咖啡机", "平板电脑", "运动鞋", "羽绒服", "洗发水", "车厘子", "空调", "电饭煲"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
AMOUNTS = [99, 129, 199, 259, 399, 699, 999, 1299, 1899, 2599, 5299]

INSTRUCTION = """你是电商售后工单的结构化抽取助手。请把用户工单抽取成 JSON，字段如下：

category   类目，取值：退款 / 退货 / 换货 / 物流 / 发票 / 投诉 / 咨询
sentiment  情绪，取值：负面 / 中性 / 正面
escalate   是否升级处理，取值：true / false。涉及金额超过 2000 元、人身安全、监管投诉时为 true
entities   关键实体对象，包含 order_id（订单号，没有写 null）、product（商品名，没有写 null）、amount（金额数字，没有写 null）

只输出 JSON，不要输出解释。"""


def make_sample(template, rng):
    """按模板填槽，返回 (输入文本, 期望 JSON 字符串)。"""
    text_tpl, category, sentiment, escalate = template
    slots = {
        "order": "A%d" % rng.randint(2001, 2030),
        "product": rng.choice(PRODUCTS),
        "amount": rng.choice(AMOUNTS),
        "city": rng.choice(CITIES),
        "days": rng.choice([2, 3, 5, 7, 10, 15]),
        "date": "2026-%02d-%02d" % (rng.randint(9, 11), rng.randint(1, 28)),
    }
    text = text_tpl.format(**slots)

    # 实体只保留文本里真实出现过的，避免"标签里有、原文里没有"的幻觉式标注
    order_id = slots["order"] if "{order}" in text_tpl else None
    product = slots["product"] if "{product}" in text_tpl else None
    amount = slots["amount"] if "{amount}" in text_tpl else None

    target = {
        "category": category,
        "sentiment": sentiment,
        "escalate": escalate,
        "entities": {"order_id": order_id, "product": product, "amount": amount},
    }
    return text, json.dumps(target, ensure_ascii=False)


def build_split(template_ids, size, seed):
    """在一个模板子集上生成 size 条样本（轮转取模板，保证分布均匀）。"""
    rng = random.Random(seed)
    picked = [TEMPLATES[i] for i in template_ids]
    rows = []
    for i in range(size):
        tpl = picked[i % len(picked)]
        text, target = make_sample(tpl, rng)
        rows.append({
            "instruction": INSTRUCTION,
            "input": text,
            "output": target,
            "template": TEMPLATES.index(tpl),
        })
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    all_ids = list(range(len(TEMPLATES)))
    rng = random.Random(42)
    rng.shuffle(all_ids)

    # 按模板分组切分：测试集模板在训练时完全没出现，避免"同模板换词"泄漏
    n = len(all_ids)
    n_test = max(6, n // 6)
    n_val = max(6, n // 6)
    test_ids = all_ids[:n_test]
    val_ids = all_ids[n_test:n_test + n_val]
    train_ids = all_ids[n_test + n_val:]

    splits = {
        "train": build_split(train_ids, 600, 1),
        "val": build_split(val_ids, 150, 2),
        "test": build_split(test_ids, 150, 3),
    }

    for name, rows in splits.items():
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条 -> {path}")

    print(f"模板总数 {n}，训练 {len(train_ids)} 个 / 验证 {len(val_ids)} 个 / 测试 {len(test_ids)} 个")
    print("训练集与测试集模板交集：", set(train_ids) & set(test_ids))


if __name__ == "__main__":
    main()
