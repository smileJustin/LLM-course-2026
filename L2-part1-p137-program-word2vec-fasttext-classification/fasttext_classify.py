# -*- coding: utf-8 -*-
"""
L2.part1 第137页 可选编程作业（二）：用 fastText 做文本分类

数据：人民日报 1998 年 1 月（9801），每篇文章取前 200 个词。
标签（自选）：文章所在的“版面号”。行编号 19980101-06-003-002 中的 06 就是第 6 版（国际）。
分类数量由少到多：2 类 → 4 类 → 12 类；另做两组对照：
    - 去掉 bigram 特征（wordNgrams=1）；
    - 用 word2vec_gensim.py 训练的 Skip-gram 词向量初始化（pretrainedVectors），即“基于向量的文本分类”。

依赖：fasttext（fasttext-wheel）、numpy<2
运行：PyCharm 右键 Run；结果同时写入本文件夹的 运行结果_fasttext.txt
"""

import argparse
import os
import random
import time
from collections import Counter, defaultdict

import fasttext

from common import DEFAULT_DATA, PRETRAINED_VEC, ft_workdir, load_articles, make_logger

# fastText 的超参数（对应论文 Joulin et al. 2017 的模型：词/ngram 向量取平均 → 线性分类器）
BASE_PARAMS = dict(epoch=25, lr=1.0, dim=100, wordNgrams=2, minCount=1, loss="softmax",
                   thread=1, seed=1, verbose=0)  # thread=1 + seed 固定，结果可复现


def write_ft_file(path, data):
    """fastText 的输入格式：每行一篇，'__label__类别 词 词 词 ...'"""
    with open(path, "w", encoding="utf-8") as f:
        for y, words in data:
            f.write("__label__%s %s\n" % (y, " ".join(words)))


def run(name, arts, pages, log, workdir, show_confusion=False, **override):
    data = [(y, w) for y, w in arts if y in pages]
    random.Random(0).shuffle(data)  # 固定种子，每组实验的划分一致
    cut = int(len(data) * 0.8)
    train, test = data[:cut], data[cut:]
    train_path = os.path.join(workdir, "train.txt")
    test_path = os.path.join(workdir, "test.txt")
    write_ft_file(train_path, train)
    write_ft_file(test_path, test)

    params = dict(BASE_PARAMS, **override)
    t0 = time.time()
    model = fasttext.train_supervised(input=train_path, **params)
    cost = time.time() - t0
    n, precision, _ = model.test(test_path)  # 单标签时 precision@1 就是准确率
    majority = Counter(y for y, _ in train).most_common(1)[0][0]
    base = sum(y == majority for y, _ in test) / len(test)

    log("\n== %s" % name)
    log("  训练 %d 篇 / 测试 %d 篇，训练用时 %.2fs，词表 %d" % (len(train), n, cost, len(model.words)))
    log("  准确率 %.1f%%   （全猜最多的类“%s”只有 %.1f%%）" % (precision * 100, majority, base * 100))

    if show_confusion:
        conf = defaultdict(Counter)
        for y, words in test:
            pred = model.predict(" ".join(words))[0][0].replace("__label__", "")
            conf[y][pred] += 1
        labels = sorted(pages)
        log("  混淆矩阵（行=真实版面，列=预测版面）")
        log("       " + " ".join("%3s" % c for c in labels) + "   该版召回率")
        for y in labels:
            row = [conf[y][c] for c in labels]
            log("    %s " % y + " ".join("%3s" % (v or ".") for v in row)
                + "   %5.1f%%" % (100.0 * conf[y][y] / max(1, sum(row))))
    return precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    log = make_logger("运行结果_fasttext.txt")
    workdir = ft_workdir()
    arts = load_articles(args.data)
    log("== 9801 共 %d 篇文章；各版面篇数：%s" % (len(arts), dict(sorted(Counter(y for y, _ in arts).items()))))
    log("== 参数：%s" % BASE_PARAMS)
    log("== 中间文件目录：%s" % workdir)

    all_pages = set(y for y, _ in arts)
    run("2 类：06 国际 vs 08 体育", arts, {"06", "08"}, log, workdir)
    run("4 类：01 要闻 / 06 国际 / 08 体育 / 12", arts, {"01", "06", "08", "12"}, log, workdir, show_confusion=True)
    run("12 类：全部版面", arts, all_pages, log, workdir, show_confusion=True)
    run("对照 1：12 类，只用单词（wordNgrams=1）", arts, all_pages, log, workdir, wordNgrams=1)

    vec = os.path.join(workdir, PRETRAINED_VEC)
    if os.path.exists(vec):
        run("对照 2：12 类，用 word2vec Skip-gram 词向量初始化（pretrainedVectors）",
            arts, all_pages, log, workdir, pretrainedVectors=vec)
    else:
        log("\n== 对照 2 跳过：没找到 %s，请先运行 word2vec_gensim.py" % vec)


if __name__ == "__main__":
    main()
