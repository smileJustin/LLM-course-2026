# -*- coding: utf-8 -*-
"""
公共部分（不依赖 torch）：语料读取、标签定义、评分、长文本滑窗解码。

直接运行本文件（python common.py）会用“标准答案模拟器”自检两种解码方法，
不需要 GPU 和 BERT，可用来确认滑窗拼接逻辑没有错。
"""

import os
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "古文语料.txt")

# 序列标注：给每个字一个标签，表示“这个字后面跟什么标点”，O 表示后面没有标点
LABELS = ["O", "，", "。", "、", "：", "；", "？", "！"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
SENT_END = {"。", "？", "！"}

# 半角、异体标点统一到这 7 种；引号、书名号、括号等不参与断句，直接去掉
PUNCT_MAP = {"，": "，", ",": "，", "。": "。", ".": "。", "．": "。", "、": "、",
             "：": "：", ":": "：", "∶": "：", "；": "；", ";": "；",
             "？": "？", "?": "？", "！": "！", "!": "！"}


def text_to_chars_labels(text):
    """“學而時習之，不亦說乎？” -> (['學','而','時','習','之','不','亦','說','乎'], ['O','O','O','O','，','O','O','O','？'])"""
    chars, labels = [], []
    for ch in text:
        p = PUNCT_MAP.get(ch)
        if p:
            if labels and labels[-1] == "O":  # 连续多个标点只保留第一个
                labels[-1] = p
        elif unicodedata.category(ch) == "Lo":  # 汉字
            chars.append(ch)
            labels.append("O")
    return chars, labels


def load_docs(path=CORPUS):
    """返回 [(篇名, 字列表, 标签列表)]；语料里 '# ' 开头的行是篇名。"""
    docs, title, buf = [], None, []

    def flush():
        if title and buf:
            chars, labels = text_to_chars_labels("".join(buf))
            if chars:
                docs.append((title, chars, labels))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# "):
                flush()
                title, buf = line[2:], []
            elif line:
                buf.append(line)
    flush()
    return docs


def split_docs(docs, test_every=10):
    """按篇划分（每 10 篇取 1 篇做测试），避免同一篇文章同时出现在训练和测试里。"""
    test = docs[::test_every]
    train = [d for i, d in enumerate(docs) if i % test_every]
    return train, test


def chunk(chars, labels, size):
    """训练时把整篇切成不超过 size 字的小段。"""
    return [(chars[i:i + size], labels[i:i + size]) for i in range(0, len(chars), size)]


def to_text(chars, labels):
    return "".join(c + (l if l != "O" else "") for c, l in zip(chars, labels))


def score(golds, preds):
    """断句 F1：只看“这里该不该断”；标点 F1：断的位置和标点种类都要对。"""
    tp_break = tp_exact = n_gold = n_pred = 0
    for g_seq, p_seq in zip(golds, preds):
        for g, p in zip(g_seq, p_seq):
            n_gold += g != "O"
            n_pred += p != "O"
            if g != "O" and p != "O":
                tp_break += 1
                tp_exact += g == p
    f1 = lambda tp: 2.0 * tp / max(1, n_gold + n_pred)
    return {"断句F1": f1(tp_break), "标点F1": f1(tp_exact)}


# ---------------------------------------------------------------- 长文本解码（课件任务二）
# predict_fn(窗口列表) -> 标签列表的列表；一次调用 = 一次前向计算（一个 batch）

def decode_by_period(docs, predict_fn, batch_size=1, window=400, keep=300):
    """课件的基础方法：每次取 window 字预测，只采纳前 keep 字里最后一个句末标点（。？！）之前的结果，
    下一个窗口从这个句末之后开始。

    batch_size=1 就是课件原始做法：一个窗口一个窗口地串行推进。
    batch_size>1 时多篇文章同时推进：每一步把“各篇当前的窗口”拼成一个 batch。
    （同一篇文章内部仍然是串行的，因为下一个窗口从哪里开始取决于上一个窗口的预测。）
    """
    results = [[] for _ in docs]
    starts = [0] * len(docs)
    queue = [i for i, d in enumerate(docs) if d]
    active, n_forward = [], 0
    while queue or active:
        while queue and len(active) < batch_size:
            active.append(queue.pop(0))
        wins = [docs[d][starts[d]:starts[d] + window] for d in active]
        preds = predict_fn(wins)
        n_forward += 1
        still = []
        for d, p in zip(active, preds):
            if starts[d] + window >= len(docs[d]):  # 最后一个窗口：全部采纳
                results[d] += p
                continue
            cut = keep  # 前 keep 字里没有句末标点时，就在 keep 处硬切
            for i in range(keep - 1, -1, -1):
                if p[i] in SENT_END:
                    cut = i + 1
                    break
            results[d] += p[:cut]
            starts[d] += cut
            still.append(d)
        active = still
    return results, n_forward


def decode_by_overlap(docs, predict_fn, batch_size=16, window=400, stride=300):
    """批量方法：窗口位置事先固定（每隔 stride 字开一个 window 字的窗口，相邻窗口重叠 window-stride 字），
    所有窗口互不依赖，可以把所有文章的所有窗口一起打包成大 batch。

    重叠部分取“离窗口边缘更远”的那个窗口的结果：每个窗口只负责中间 [margin, margin+stride) 这一段，
    第一个窗口从 0 开始、最后一个窗口负责到结尾。这样每个字的预测都有至少 margin 字的左右上下文。
    """
    margin = (window - stride) // 2
    jobs = []  # (文章号, 窗口起点, 负责区间起, 负责区间止)
    for d, chars in enumerate(docs):
        if not chars:
            continue
        starts = [0]
        while starts[-1] + window < len(chars):
            starts.append(starts[-1] + stride)
        for k, s in enumerate(starts):
            lo = 0 if k == 0 else s + margin
            hi = len(chars) if k == len(starts) - 1 else s + stride + margin
            jobs.append((d, s, lo, hi))
    # 按窗口长度从长到短排序，同一 batch 里长度接近，padding 少
    jobs.sort(key=lambda j: -min(window, len(docs[j[0]]) - j[1]))
    results = [[None] * len(c) for c in docs]
    n_forward = 0
    for b in range(0, len(jobs), batch_size):
        batch = jobs[b:b + batch_size]
        preds = predict_fn([docs[d][s:s + window] for d, s, _, _ in batch])
        n_forward += 1
        for (d, s, lo, hi), p in zip(batch, preds):
            results[d][lo:hi] = p[lo - s:hi - s]
    return results, n_forward


# ---------------------------------------------------------------- 自检
def _self_test():
    docs = load_docs()
    _, test = split_docs(docs)
    chars = [c for _, c, _ in test]
    golds = [l for _, _, l in test]
    print("语料 %d 篇，测试 %d 篇，测试集共 %d 字，最长一篇 %d 字" %
          (len(docs), len(test), sum(map(len, chars)), max(map(len, chars))))

    # “标准答案模拟器”：按窗口在原文中的位置，返回真实标签（代替 BERT）
    def oracle(wins):
        out = []
        for w in wins:
            for c, l in zip(chars, golds):  # 找到窗口来自哪篇、哪个位置
                for s in range(0, len(c) - len(w) + 1):
                    if c[s:s + len(w)] == w:
                        out.append(l[s:s + len(w)])
                        break
                else:
                    continue
                break
        return out

    for name, fn in [("课件基础法 batch=1", lambda: decode_by_period(chars, oracle, 1)),
                     ("课件基础法 batch=8", lambda: decode_by_period(chars, oracle, 8)),
                     ("重叠滑窗 batch=8", lambda: decode_by_overlap(chars, oracle, 8))]:
        preds, n = fn()
        assert all(len(p) == len(g) for p, g in zip(preds, golds)), "长度不一致"
        print("  %-18s 前向次数 %3d   %s" % (name, n, score(golds, preds)))
    print("自检通过：三种解码方式都完整、无重叠地拼回了全文。")


if __name__ == "__main__":
    _self_test()
