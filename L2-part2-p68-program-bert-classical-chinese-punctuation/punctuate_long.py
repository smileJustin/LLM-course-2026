# -*- coding: utf-8 -*-
"""
L2-part2 第68页 编程练习 任务二：长古文的高效率断句（Batch 解码）

BERT 一次最多看 512 个 token，长文必须切窗口。比较三种做法：
  A. 课件基础方法：每次 400 字、只取前 300 字里最后一个句号之前的结果，下一窗从句号后开始；一次一个窗口（batch=1）
  B. 基础方法 + 多篇并行：每一步把多篇文章“当前的窗口”拼成一个 batch
  C. 重叠滑窗 + 全量批处理：窗口位置事先固定（400 字一窗、步长 300、重叠 100），
     所有文章的所有窗口互不依赖，一次性按长度排序后打包成大 batch
详细原理见 common.py 里的 decode_by_period / decode_by_overlap。

运行：先运行 train.py 得到 ./bert_punct_model，再运行本文件
     python punctuate_long.py --batch-size 16
     python punctuate_long.py --file 我的古文.txt     # 给任意文本加标点，结果写到 我的古文_标点.txt
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import (HERE, decode_by_overlap, decode_by_period, load_docs, score, split_docs, tee_stdout,
                    text_to_chars_labels, to_text)
from train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "bert_punct_model"))
    ap.add_argument("--batch-size", type=int, default=16, help="12G 显卡开 8~16 没问题，更大的卡可开 32~64")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--file", default=None, help="给这个文本文件加标点")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    if device.type == "cuda":
        model.half()  # 推理用半精度：更快、更省显存
    predict_fn = lambda wins: predict(model, tokenizer, wins, device)

    # ---- 给任意文本加标点
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            chars, _ = text_to_chars_labels(f.read())  # 先去掉原有标点
        (labels,), _ = decode_by_overlap([chars], predict_fn, args.batch_size, args.window)
        out = os.path.splitext(args.file)[0] + "_标点.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(to_text(chars, labels))
        print("已写入", out)
        return

    # ---- 在测试集的整篇长文上比较三种解码方式
    tee_stdout("运行结果_punctuate_batch%d.txt" % args.batch_size)
    print("设备：", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    _, test = split_docs(load_docs())
    docs = [c for _, c, _ in test]
    golds = [l for _, _, l in test]
    print("测试集 %d 篇整文，共 %d 字，最长 %d 字（超过 512 的有 %d 篇）\n" %
          (len(docs), sum(map(len, docs)), max(map(len, docs)), sum(len(d) > 512 for d in docs)))

    predict_fn([docs[0][:args.window]] * 2)  # 预热一次，避免第一次调用的初始化时间算进去
    B, W = args.batch_size, args.window
    methods = [("A 课件基础法 batch=1", lambda: decode_by_period(docs, predict_fn, 1, W, W * 3 // 4)),
               ("B 基础法+多篇并行 batch=%d" % B, lambda: decode_by_period(docs, predict_fn, B, W, W * 3 // 4)),
               ("C 重叠滑窗批处理 batch=%d" % B, lambda: decode_by_overlap(docs, predict_fn, B, W, W * 3 // 4))]
    print("%-28s %8s %9s %8s %8s" % ("方法", "前向次数", "用时(秒)", "断句F1", "标点F1"))
    results = {}
    for name, run in methods:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        preds, n_forward = run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        m = score(golds, preds)
        results[name] = preds
        print("%-28s %8d %9.2f %8.3f %8.3f" % (name, n_forward, time.time() - t0, m["断句F1"], m["标点F1"]))

    i = max(range(len(docs)), key=lambda k: len(docs[k]))
    print("\n最长一篇《%s》开头（上：原文，下：方法 C）" % test[i][0])
    print("  原文：", to_text(docs[i], golds[i])[:150])
    print("  模型：", to_text(docs[i], results[methods[2][0]][i])[:150])


if __name__ == "__main__":
    main()
