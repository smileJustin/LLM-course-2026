# -*- coding: utf-8 -*-
"""
L2.part1 第137页 可选编程作业（一）：用 Gensim 做表示学习（word2vec）

实验内容：
    1. 在人民日报 1998 年 1~6 月语料上，分别训练 Skip-gram 和 CBOW 两种 word2vec，比较训练时间；
    2. 用四种方式检查词向量学到了什么：最近邻、词类比、词对相似度、找出不同类的词；
    3. 把 Skip-gram 词向量导出成 .vec 文件，供 fasttext_classify.py 做“预训练向量 + 分类”实验。

依赖：gensim
运行：PyCharm 右键 Run；结果同时写入本文件夹的 运行结果_word2vec.txt
"""

import argparse
import os
import time

from gensim.models import Word2Vec

from common import DEFAULT_DATA, PRETRAINED_VEC, ft_workdir, load_sentences, make_logger

NEIGHBOR_WORDS = ["经济", "增长", "银行", "北京", "足球", "电脑", "学生", "春节", "农民", "导弹"]
# a : b = c : ?   用 b - a + c 求答案
ANALOGIES = [("中国", "北京", "日本", "东京"),
             ("中国", "北京", "法国", "巴黎"),
             ("中国", "北京", "俄罗斯", "莫斯科"),
             ("父亲", "母亲", "儿子", "女儿"),
             ("一月", "二月", "三月", "四月")]
PAIRS = [("经济", "发展"), ("足球", "篮球"), ("北京", "上海"), ("银行", "贷款"), ("足球", "银行")]
ODD_ONE_OUT = [["足球", "篮球", "排球", "银行"], ["北京", "上海", "天津", "苹果"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="训练用的 CPU 线程数")
    args = ap.parse_args()

    log = make_logger("运行结果_word2vec.txt")
    t0 = time.time()
    sents = load_sentences(args.data, args.months)
    log("== 语料：%d 个月，%d 段，%d 词（不含标点），读入用时 %.1fs"
        % (args.months, len(sents), sum(map(len, sents)), time.time() - t0))

    # ---------------------------------------------------------------- 1. 训练两种模型
    log("\n== 1. 训练（vector_size=%d, window=5, min_count=5, negative=5, sample=1e-4, epochs=%d）"
        % (args.dim, args.epochs))
    models = {}
    for name, sg in [("Skip-gram", 1), ("CBOW", 0)]:
        t0 = time.time()
        models[name] = Word2Vec(sentences=sents, vector_size=args.dim, window=5, min_count=5,
                                sg=sg, hs=0, negative=5, sample=1e-4, epochs=args.epochs,
                                workers=args.workers, seed=1)
        log("  %-9s 训练用时 %6.1fs，词表 %d 个词" % (name, time.time() - t0, len(models[name].wv)))

    # ---------------------------------------------------------------- 2. 最近邻
    log("\n== 2. 最近邻（余弦相似度）")
    for w in NEIGHBOR_WORDS:
        for name, m in models.items():
            if w not in m.wv:
                log("  %s [%s]：不在词表中" % (w, name))
                continue
            sims = m.wv.most_similar(w, topn=8)
            log("  %s [%-9s]：%s" % (w, name, "  ".join("%s(%.2f)" % (x, s) for x, s in sims)))

    # ---------------------------------------------------------------- 3. 词类比
    log("\n== 3. 词类比  a : b = c : ?   （答案 = b - a + c 最近的词）")
    for name, m in models.items():
        hit1 = hit5 = n = 0
        for a, b, c, expect in ANALOGIES:
            if not all(w in m.wv for w in (a, b, c, expect)):
                log("  [%-9s] %s:%s = %s:?  有词不在词表中，跳过" % (name, a, b, c))
                continue
            top = [x for x, _ in m.wv.most_similar(positive=[b, c], negative=[a], topn=5)]
            n += 1
            hit1 += top[0] == expect
            hit5 += expect in top
            log("  [%-9s] %s:%s = %s:?  前5：%s   （期望：%s）" % (name, a, b, c, "、".join(top), expect))
        log("  [%-9s] 第 1 名命中 %d/%d，前 5 名命中 %d/%d" % (name, hit1, n, hit5, n))

    # ---------------------------------------------------------------- 4. 词对相似度 & 找不同
    log("\n== 4. 词对相似度")
    for a, b in PAIRS:
        vals = ["%s %.2f" % (name, m.wv.similarity(a, b)) for name, m in models.items() if a in m.wv and b in m.wv]
        log("  %s - %s：%s" % (a, b, "，".join(vals) or "有词不在词表中"))
    log("\n== 5. 找出不同类的词（doesnt_match）")
    for group in ODD_ONE_OUT:
        for name, m in models.items():
            if all(w in m.wv for w in group):
                log("  [%-9s] %s → %s" % (name, "、".join(group), m.wv.doesnt_match(group)))

    # ---------------------------------------------------------------- 5. 导出词向量给 fastText 用
    path = os.path.join(ft_workdir(), PRETRAINED_VEC)
    models["Skip-gram"].wv.save_word2vec_format(path)
    log("\n== Skip-gram 词向量已导出到 %s（供 fasttext_classify.py 使用，不在仓库里）" % path)


if __name__ == "__main__":
    main()
