# -*- coding: utf-8 -*-
"""
L2.part1 第137页 可选编程作业（一）：表示学习 —— 自己实现 word2vec（Skip-gram + 负采样）

只用 Python 标准库，在 PyCharm 里右键 Run 即可。
对应论文：Mikolov et al. 2013, "Distributed Representations of Words and Phrases and their Compositionality"
对应 Gensim：gensim.models.Word2Vec(sg=1, negative=5, window=5, vector_size=50, min_count=5, sample=1e-4)

流程：
    1. 读入人民日报 1998 年 1 月切分语料，清理（同 L1 第125页作业）
    2. 建词表（去掉低频词）、高频词下采样、构造负采样分布 (词频^0.75)
    3. 逐句训练 Skip-gram：用中心词预测周围词，每个正例配 5 个负例
    4. 输出若干词的最近邻，并把词向量存成 word2vec 文本格式（Gensim 可直接读取）
"""

import argparse
import glob
import math
import os
import random
import re
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "1998-上半年-带音(1)", "1998-上半年-带音")

LINE_ID = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+")
PINYIN = re.compile(r"\{[^}]*\}")
FULL2HALF = {c: c - 0xFEE0 for c in list(range(0xFF10, 0xFF1A)) +
             list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}
PUNCT = set("，。、；：？！“”‘’（）《》—…·「」『』")


def load_sentences(data_dir, max_tokens):
    """读第一个月的语料，返回句子列表（每句是词列表），标点不参与训练。"""
    path = sorted(glob.glob(os.path.join(data_dir, "*.txt")))[0]
    sents, total = [], 0
    with open(path, encoding="gbk", errors="ignore") as f:
        for line in f:
            line = PINYIN.sub("", LINE_ID.sub("", line.strip())).translate(FULL2HALF)
            words = [w for w in line.split() if w not in PUNCT]
            if words:
                sents.append(words)
                total += len(words)
                if total >= max_tokens:
                    break
    return sents, total


class Word2Vec:
    def __init__(self, sentences, dim=50, window=5, negative=5, min_count=5, sample=1e-4, seed=1):
        self.dim, self.window, self.negative, self.sample = dim, window, negative, sample
        self.rng = random.Random(seed)
        cnt = Counter(w for s in sentences for w in s)
        self.words = [w for w, c in cnt.most_common() if c >= min_count]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = [cnt[w] for w in self.words]
        self.total = sum(self.counts)
        V = len(self.words)
        # 输入向量随机初始化，输出向量初始化为 0（与原始 C 代码、Gensim 一致）
        self.W_in = [[(self.rng.random() - 0.5) / dim for _ in range(dim)] for _ in range(V)]
        self.W_out = [[0.0] * dim for _ in range(V)]
        # 负采样表：按 词频^0.75 的比例重复填充，抽样时随机取一个位置即可
        pw = [c ** 0.75 for c in self.counts]
        z = sum(pw)
        self.neg_table = []
        for i, p in enumerate(pw):
            self.neg_table += [i] * max(1, int(p / z * 1_000_000))
        # 高频词下采样的保留概率（论文公式的 C 代码版本）
        self.keep_prob = [min(1.0, (math.sqrt(c / (sample * self.total)) + 1) * (sample * self.total) / c)
                          for c in self.counts]

    def train(self, sentences, epochs=1, lr0=0.025):
        total_steps = epochs * len(sentences)
        step = 0
        for ep in range(epochs):
            t0, loss_sum, n_pairs = time.time(), 0.0, 0
            for sent in sentences:
                lr = max(lr0 * (1 - step / total_steps), lr0 * 1e-4)  # 学习率线性衰减
                step += 1
                ids = [self.index[w] for w in sent if w in self.index]
                ids = [i for i in ids if self.rng.random() < self.keep_prob[i]]  # 下采样
                for pos, center in enumerate(ids):
                    b = self.rng.randint(1, self.window)  # 动态窗口：近的词被采到的机会更多
                    for j in range(max(0, pos - b), min(len(ids), pos + b + 1)):
                        if j != pos:
                            loss_sum += self._update(center, ids[j], lr)
                            n_pairs += 1
            print("  第 %d 轮：%d 个正例对，平均损失 %.3f，用时 %.1fs"
                  % (ep + 1, n_pairs, loss_sum / max(1, n_pairs), time.time() - t0))

    def _update(self, center, context, lr):
        """一个 (中心词, 上下文词) 正例 + negative 个负例的 SGD 更新，返回该样本的损失。"""
        v = self.W_in[center]
        grad_v = [0.0] * self.dim
        loss = 0.0
        targets = [(context, 1)] + [(self.neg_table[self.rng.randrange(len(self.neg_table))], 0)
                                    for _ in range(self.negative)]
        for t, label in targets:
            if label == 0 and t == context:
                continue
            u = self.W_out[t]
            f = sum(a * b for a, b in zip(v, u))
            f = max(-6.0, min(6.0, f))
            sig = 1.0 / (1.0 + math.exp(-f))
            loss += -math.log(sig if label else 1 - sig)
            g = (label - sig) * lr  # 对数几率的梯度
            for k in range(self.dim):
                grad_v[k] += g * u[k]
                u[k] += g * v[k]
        for k in range(self.dim):
            v[k] += grad_v[k]
        return loss

    # ------------------------------------------------ 使用词向量
    def normalized(self, remove_top=True):
        """单位化后的词向量。

        小语料、训练轮数少时，几乎所有词向量都挤在同一个方向上（主要编码词频），
        直接算余弦相似度会全部接近 1。remove_top=True 时先减去平均向量、再去掉第一主成分
        （All-but-the-Top, Mu & Viswanath 2018），剩下的方向才体现语义差别。
        """
        vecs = [list(v) for v in self.W_in]
        if remove_top:
            mean = [sum(col) / len(vecs) for col in zip(*vecs)]
            vecs = [[a - b for a, b in zip(v, mean)] for v in vecs]
            d = [self.rng.random() - 0.5 for _ in range(self.dim)]  # 幂迭代求第一主成分方向
            for _ in range(30):
                proj = [sum(a * b for a, b in zip(v, d)) for v in vecs]
                d = [sum(p * v[k] for p, v in zip(proj, vecs)) for k in range(self.dim)]
                n = math.sqrt(sum(x * x for x in d))
                d = [x / n for x in d]
            vecs = [[a - p * b for a, b in zip(v, d)]
                    for v, p in ((v, sum(x * y for x, y in zip(v, d))) for v in vecs)]
        out = []
        for v in vecs:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def most_similar(self, word, topn=8, normed=None):
        normed = normed or self.normalized()
        q = normed[self.index[word]]
        sims = [(sum(a * b for a, b in zip(q, v)), w) for w, v in zip(self.words, normed) if w != word]
        sims.sort(reverse=True)
        return sims[:topn]

    def save(self, path):
        """word2vec 文本格式，可用 gensim.models.KeyedVectors.load_word2vec_format(path) 读取。"""
        with open(path, "w", encoding="utf-8") as f:
            f.write("%d %d\n" % (len(self.words), self.dim))
            for w, v in zip(self.words, self.W_in):
                f.write(w + " " + " ".join("%.5f" % x for x in v) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--max-tokens", type=int, default=300_000, help="用多少词训练（纯 Python 较慢，先用一部分）")
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    sents, total = load_sentences(args.data, args.max_tokens)
    print("== 语料：%d 段，%d 词（不含标点）" % (len(sents), total))
    model = Word2Vec(sents, dim=args.dim, window=args.window)
    print("== 词表：%d 个词（出现 ≥5 次）" % len(model.words))

    print("== 训练 Skip-gram + 负采样")
    model.train(sents, epochs=args.epochs)
    model.save(os.path.join(HERE, "vectors.txt"))
    print("== 词向量已保存到 vectors.txt")

    raw = model.normalized(remove_top=False)
    print("\n== 直接用原始向量算最近邻（注意相似度几乎全是 1，见 README）")
    print("  经济：%s" % "  ".join("%s(%.2f)" % (x, s) for s, x in model.most_similar("经济", normed=raw)))

    print("\n== 去掉公共方向后的最近邻（余弦相似度）")
    normed = model.normalized()
    for w in ["中国", "经济", "北京", "足球", "银行", "学生", "电脑", "春节", "一月", "增长"]:
        if w in model.index:
            sims = model.most_similar(w, normed=normed)
            print("  %s：%s" % (w, "  ".join("%s(%.2f)" % (x, s) for s, x in sims)))


if __name__ == "__main__":
    main()
