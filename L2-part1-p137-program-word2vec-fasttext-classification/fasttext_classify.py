# -*- coding: utf-8 -*-
"""
L2.part1 第137页 可选编程作业（二）：基于向量的文本分类 —— 自己实现 fastText 分类器

只用 Python 标准库，在 PyCharm 里右键 Run 即可。
对应论文：Joulin et al. 2017, "Bag of Tricks for Efficient Text Classification"
对应 fastText 命令：fasttext supervised -input train.txt -dim 10 -wordNgrams 2 -bucket 262144 -epoch 25 -lr 1.0

数据：人民日报 1998 年 1 月（课件说的 9801）切分语料。
标签（自选）：每篇文章所在的“版面号”。行编号 19980101-06-003-002 中的 06 就是第 6 版。
     人民日报各版面相对固定：01 要闻、06 国际、08 体育……所以版面号可以近似当作“栏目类别”。
分类数量由少到多：2 类 → 4 类 → 全部 12 个版面。

模型（与论文一致的最小版本）：
    文章 → 词 + 相邻词二元组(bigram) → 哈希到桶 → 查表得到向量 → 取平均 → 线性层 → softmax
"""

import argparse
import glob
import math
import os
import random
import re
import time
import zlib
from collections import Counter, OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "1998-上半年-带音(1)", "1998-上半年-带音")

LINE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-\d{3}\s+(.*)$")
PINYIN = re.compile(r"\{[^}]*\}")
FULL2HALF = {c: c - 0xFEE0 for c in list(range(0xFF10, 0xFF1A)) +
             list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}
PUNCT = set("，。、；：？！“”‘’（）《》—…·「」『』〓")

PAGE_NAMES = {"01": "要闻", "06": "国际", "08": "体育"}


def load_articles(data_dir, max_words=200):
    """把同一 日期-版面-文章号 的各段合成一篇文章，返回 [(版面号, 词列表)]。"""
    path = sorted(glob.glob(os.path.join(data_dir, "*.txt")))[0]  # 1 月 = 9801
    arts = OrderedDict()
    with open(path, encoding="gbk", errors="ignore") as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            text = PINYIN.sub("", m.group(4)).translate(FULL2HALF)
            key = (m.group(1), m.group(2), m.group(3))
            arts.setdefault(key, []).extend(w for w in text.split() if w not in PUNCT)
    # 只取每篇前 max_words 个词（标题 + 导语已经包含主要信息，也让纯 Python 跑得动）
    return [(key[1], words[:max_words]) for key, words in arts.items() if words]


# ---------------------------------------------------------------- fastText 模型
class FastText:
    def __init__(self, labels, dim=10, buckets=2 ** 18, word_ngrams=2, seed=1):
        self.labels = labels
        self.dim, self.buckets, self.word_ngrams = dim, buckets, word_ngrams
        self.rng = random.Random(seed)
        self.A = {}  # 输入向量表：桶号 -> 向量（用到才创建，省内存）
        self.B = [[0.0] * dim for _ in labels]  # 输出层（与 fastText 一样初始化为 0）

    def features(self, words):
        feats = list(words)
        if self.word_ngrams >= 2:
            feats += [a + "_" + b for a, b in zip(words, words[1:])]
        # 哈希技巧：不建词表，所有特征直接映射到固定数量的桶，bigram 再多也不会撑爆内存
        return [zlib.crc32(f.encode("utf-8")) % self.buckets for f in feats]

    def _vec(self, i):
        v = self.A.get(i)
        if v is None:
            v = self.A[i] = [self.rng.uniform(-1.0 / self.dim, 1.0 / self.dim) for _ in range(self.dim)]
        return v

    def _forward(self, ids):
        h = [0.0] * self.dim
        for i in ids:
            v = self._vec(i)
            for k in range(self.dim):
                h[k] += v[k]
        h = [x / len(ids) for x in h]  # 文章向量 = 特征向量的平均
        logits = [sum(a * b for a, b in zip(row, h)) for row in self.B]
        mx = max(logits)
        e = [math.exp(z - mx) for z in logits]
        s = sum(e)
        return h, [x / s for x in e]

    def train(self, data, epochs=5, lr0=0.5):
        feats = [(self.features(w), self.labels.index(y)) for y, w in data]
        total = epochs * len(feats)
        step = 0
        for ep in range(epochs):
            self.rng.shuffle(feats)
            loss = 0.0
            for ids, y in feats:
                if not ids:
                    continue
                lr = lr0 * (1 - step / total)
                step += 1
                h, p = self._forward(ids)
                loss += -math.log(max(p[y], 1e-12))
                # softmax 交叉熵的梯度：p - onehot
                grad_h = [0.0] * self.dim
                for c, row in enumerate(self.B):
                    g = p[c] - (1.0 if c == y else 0.0)
                    for k in range(self.dim):
                        grad_h[k] += g * row[k]
                        row[k] -= lr * g * h[k]
                scale = lr / len(ids)  # 平均层：梯度平均分给每个输入特征
                for i in ids:
                    v = self.A[i]
                    for k in range(self.dim):
                        v[k] -= scale * grad_h[k]
            print("    第 %d 轮 平均损失 %.3f" % (ep + 1, loss / len(feats)))

    def predict(self, words):
        ids = self.features(words)
        if not ids:
            return self.labels[0]
        _, p = self._forward(ids)
        return self.labels[max(range(len(p)), key=p.__getitem__)]


def run(arts, pages, args, word_ngrams=2, show_confusion=False):
    data = [(y, w) for y, w in arts if y in pages]
    random.Random(0).shuffle(data)
    cut = int(len(data) * 0.8)
    train, test = data[:cut], data[cut:]
    majority = Counter(y for y, _ in train).most_common(1)[0][0]
    base = sum(y == majority for y, _ in test) / len(test)

    t0 = time.time()
    model = FastText(sorted(pages), dim=args.dim, word_ngrams=word_ngrams)
    model.train(train, epochs=args.epochs, lr0=args.lr)
    preds = [model.predict(w) for _, w in test]
    acc = sum(p == y for p, (y, _) in zip(preds, test)) / len(test)
    print("  训练 %d 篇 / 测试 %d 篇，用时 %.1fs" % (len(train), len(test), time.time() - t0))
    print("  准确率 %.1f%%   （全猜最多的类“%s”只有 %.1f%%）" % (acc * 100, majority, base * 100))

    if show_confusion:
        conf = defaultdict(Counter)
        for p, (y, _) in zip(preds, test):
            conf[y][p] += 1
        labels = sorted(pages)
        print("  混淆矩阵（行=真实版面，列=预测版面）")
        print("       " + " ".join("%3s" % c for c in labels) + "   该版召回率")
        for y in labels:
            row = [conf[y][c] for c in labels]
            print("    %s " % y + " ".join("%3s" % (v or ".") for v in row) + "   %5.1f%%" % (100.0 * conf[y][y] / max(1, sum(row))))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1.0)
    args = ap.parse_args()

    arts = load_articles(args.data)
    print("== 共 %d 篇文章；各版面篇数：%s" % (len(arts), dict(sorted(Counter(y for y, _ in arts).items()))))

    all_pages = sorted(set(y for y, _ in arts))
    exps = [("2 类：06 国际 vs 08 体育", {"06", "08"}),
            ("4 类：01 要闻 / 06 国际 / 08 体育 / 12", {"01", "06", "08", "12"}),
            ("12 类：全部版面", set(all_pages))]
    for name, pages in exps:
        print("\n== %s（词 + bigram）" % name)
        run(arts, pages, args, show_confusion=(len(pages) > 2))

    print("\n== 消融：12 类只用单词、不加 bigram")
    run(arts, set(all_pages), args, word_ngrams=1)


if __name__ == "__main__":
    main()
