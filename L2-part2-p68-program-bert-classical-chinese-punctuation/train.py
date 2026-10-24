# -*- coding: utf-8 -*-
"""
L2-part2 第68页 编程练习 任务一：微调 BERT，得到古文断句标点模型

做法：把断句标点看成“逐字序列标注”——给每个字预测它后面跟的标点（或不跟标点 O），
     用 BertForTokenClassification 在已训练好的 BERT 上加一个分类层微调。

需要：torch、transformers（首次运行会自动下载预训练模型，约 400MB）
运行：PyCharm 右键 Run，或  python train.py --epochs 3 --batch-size 16
输出：./bert_punct_model/（微调好的模型，供 punctuate_long.py 使用）
"""

import argparse
import contextlib
import os
import random
import time

# 国内访问 huggingface.co 不稳定，默认走镜像；能直连的话可以删掉这一行
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import HERE, LABEL2ID, LABELS, chunk, load_docs, score, split_docs, to_text


def make_batch(tokenizer, samples, device):
    """samples: [(字列表, 标签列表)] -> BERT 输入。每个字直接当作一个 token，保证字和标签一一对应。"""
    max_len = max(len(c) for c, _ in samples) + 2  # +[CLS] [SEP]
    ids = torch.full((len(samples), max_len), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)  # -100 的位置不计算损失（[CLS] [SEP] 和 padding）
    for i, (chars, labs) in enumerate(samples):
        x = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(chars) + [tokenizer.sep_token_id]
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
        if labs is not None:
            labels[i, 1:1 + len(labs)] = torch.tensor([LABEL2ID[l] for l in labs])
    return ids.to(device), mask.to(device), labels.to(device)


@torch.no_grad()
def predict(model, tokenizer, windows, device):
    """windows: [字列表] -> [标签列表]。punctuate_long.py 也用这个函数。"""
    ids, mask, _ = make_batch(tokenizer, [(w, None) for w in windows], device)
    # GPU 上用半精度推理，更快；CPU 上保持单精度
    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
    with amp:
        logits = model(input_ids=ids, attention_mask=mask).logits
    pred = logits.argmax(-1).tolist()
    return [[LABELS[j] for j in pred[i][1:1 + len(w)]] for i, w in enumerate(windows)]


def evaluate(model, tokenizer, samples, device, batch_size=32):
    model.eval()
    preds = []
    for b in range(0, len(samples), batch_size):
        preds += predict(model, tokenizer, [c for c, _ in samples[b:b + batch_size]], device)
    model.train()
    return score([l for _, l in samples], preds), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese", help="预训练模型名或本地路径")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256, help="训练时每段的最大字数")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", default=os.path.join(HERE, "bert_punct_model"))
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU（会比较慢）")

    # ---- 数据：按篇划分训练/测试，再切成小段
    train_docs, test_docs = split_docs(load_docs())
    train = [s for _, c, l in train_docs for s in chunk(c, l, args.max_len)]
    test = [s for _, c, l in test_docs for s in chunk(c, l, args.max_len)]
    print("训练 %d 篇 → %d 段；测试 %d 篇 → %d 段" % (len(train_docs), len(train), len(test_docs), len(test)))

    # ---- 模型：预训练 BERT + 一个 8 分类的线性层（新加的层随机初始化）
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID).to(device)
    unk = sum(t == tokenizer.unk_token_id for _, c, _ in train_docs for t in tokenizer.convert_tokens_to_ids(c))
    print("训练集中不在 BERT 词表里的字（会变成 [UNK]）：%d 个" % unk)

    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)

    print("微调前（分类层是随机的）：", evaluate(model, tokenizer, test, device)[0])
    model.train()
    for ep in range(args.epochs):
        random.shuffle(train)
        t0, total = time.time(), 0.0
        for b in range(0, len(train), args.batch_size):
            ids, mask, labels = make_batch(tokenizer, train[b:b + args.batch_size], device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            total += loss.item()
        n_batch = (len(train) + args.batch_size - 1) // args.batch_size
        metrics, preds = evaluate(model, tokenizer, test, device)
        print("第 %d 轮  训练损失 %.4f  用时 %.0fs  测试集 断句F1 %.3f  标点F1 %.3f"
              % (ep + 1, total / n_batch, time.time() - t0, metrics["断句F1"], metrics["标点F1"]))

    print("\n测试样例（上：原文，下：模型）")
    for (chars, labs), pred in list(zip(test, preds))[:3]:
        print("  原文：", to_text(chars, labs)[:80])
        print("  模型：", to_text(chars, pred)[:80])

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("\n模型已保存到", args.out)


if __name__ == "__main__":
    main()
