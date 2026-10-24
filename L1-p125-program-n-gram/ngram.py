# -*- coding: utf-8 -*-
"""
L1 第125页 编程作业：用人民日报 1998 切分语料训练 n-gram 语言模型并续写文本

只用 Python 标准库，不需要安装任何第三方包。
在 PyCharm 里直接右键 Run 即可；也可以在命令行带参数运行，例如：
    python ngram.py --months 1 --max-n 4
    python ngram.py --temperature 1.2 --top-k 20

流程：
    1. 读取语料并清理（去掉行首编号、拼音标注，全角数字/字母转半角，按句切开）
    2. 划分训练集 / 测试集
    3. 分别训练 1-gram、2-gram、3-gram ...，记录训练时间
    4. 用插值平滑（Jelinek-Mercer）在测试集上算困惑度（perplexity）
    5. 以“在阳光明媚的五月，我们学校胜利召开了”为开头，用不同参数续写
"""

import argparse
import glob
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

# 语料默认位置：本脚本所在文件夹的上一级里的“1998-上半年-带音(1)/1998-上半年-带音”
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "..", "1998-上半年-带音(1)", "1998-上半年-带音")

BOS, EOS = "<s>", "</s>"
PROMPT = "在阳光明媚的五月，我们学校胜利召开了"

# ---------------------------------------------------------------- 1. 读取与清理
LINE_ID = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+")  # 行首编号，如 19980101-01-001-001
PINYIN = re.compile(r"\{[^}]*\}")                      # 拼音标注，如 这{zhe4}
SENT_END = {"。", "！", "？"}

# 全角数字/字母 -> 半角（“１９９８年” -> “1998年”），标点保持不变
FULL2HALF = {c: c - 0xFEE0 for c in list(range(0xFF10, 0xFF1A)) +
             list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}


def clean_line(line):
    """返回清理后的词列表；空行返回 []。"""
    line = LINE_ID.sub("", line.strip())
    line = PINYIN.sub("", line)
    line = line.translate(FULL2HALF)
    return line.split()


def load_corpus(data_dir, months):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))[:months]
    if not files:
        raise SystemExit("在 %s 下没有找到语料 txt，请用 --data 指定语料文件夹" % data_dir)
    sentences = []
    for f in files:
        with open(f, encoding="gbk", errors="ignore") as fh:  # 原始语料是 GBK 编码
            for line in fh:
                words = clean_line(line)
                # 按句号/问号/叹号把一段切成多句，句子才是 <s> ... </s> 的合理单位
                cur = []
                for w in words:
                    cur.append(w)
                    if w in SENT_END:
                        sentences.append(cur)
                        cur = []
                if cur:
                    sentences.append(cur)
        print("  读入 %s" % os.path.basename(f))
    return sentences


# ---------------------------------------------------------------- 2. n-gram 模型
class NGramModel:
    def __init__(self, n):
        self.n = n
        # counts[k][context] = Counter(下一个词 -> 次数)，k = 上下文长度 0..n-1
        self.counts = [defaultdict(Counter) for _ in range(n)]
        self.vocab = set()

    def train(self, sentences):
        n = self.n
        for sent in sentences:
            toks = [BOS] * (n - 1) + sent + [EOS]
            self.vocab.update(sent)
            for i in range(n - 1, len(toks)):
                w = toks[i]
                for k in range(n):  # 同时统计所有更低阶，供平滑与回退使用
                    ctx = tuple(toks[i - k:i])
                    self.counts[k][ctx][w] += 1
        self.vocab.add(EOS)
        # 预先算好每个上下文的总次数，避免反复求和
        self.totals = [{c: sum(cnt.values()) for c, cnt in level.items()} for level in self.counts]

    # ---- 插值平滑：P = λ_n·P_ML(n阶) + ... + λ_1·P_ML(1阶)，1阶再用加一平滑兜底未登录词
    def prob(self, ctx, w, lambdas):
        V = len(self.vocab) + 1  # +1 给未登录词 <unk>
        uni_cnt = self.counts[0][()].get(w, 0)
        p = lambdas[0] * (uni_cnt + 1) / (self.totals[0][()] + V)
        for k in range(1, self.n):
            c = tuple(ctx[-k:])
            total = self.totals[k].get(c)
            if total:
                p += lambdas[k] * self.counts[k][c].get(w, 0) / total
            else:
                p += lambdas[k] * (uni_cnt + 1) / (self.totals[0][()] + V)  # 没见过的上下文：退回1阶
        return p

    def perplexity(self, sentences):
        lambdas = default_lambdas(self.n)
        log_sum, N = 0.0, 0
        for sent in sentences:
            toks = [BOS] * (self.n - 1) + sent + [EOS]
            for i in range(self.n - 1, len(toks)):
                log_sum += -math.log(self.prob(toks[i - self.n + 1:i] if self.n > 1 else [], toks[i], lambdas))
                N += 1
        return math.exp(log_sum / N)

    # ---- 生成：从最长的已见上下文开始“回退”，按温度和 top-k 采样
    def next_word(self, history, temperature=1.0, top_k=0, rng=random):
        for k in range(self.n - 1, -1, -1):
            ctx = tuple(history[-k:]) if k else ()
            if len(ctx) == k and ctx in self.counts[k]:
                cands = self.counts[k][ctx].most_common(top_k or None)
                break
        if temperature <= 0:  # 温度 0 = 贪心，每次取最高频
            return cands[0][0]
        weights = [c ** (1.0 / temperature) for _, c in cands]
        return rng.choices([w for w, _ in cands], weights=weights)[0]

    def generate(self, prompt_words, max_len=60, temperature=1.0, top_k=0, seed=0):
        rng = random.Random(seed)
        history = [BOS] * (self.n - 1) + list(prompt_words)
        out = []
        for _ in range(max_len):
            w = self.next_word(history, temperature, top_k, rng)
            if w == EOS:
                break
            out.append(w)
            history.append(w)
        return out

    def size(self):
        """最高阶 n-gram 的不同条目数（如 3-gram 就是不同的 (w1,w2,w3) 个数）。"""
        return sum(len(cnt) for cnt in self.counts[self.n - 1].values())


def default_lambdas(n):
    """简单的固定插值权重：越高阶权重越大，总和为 1。"""
    raw = [2 ** k for k in range(n)]  # 例如 n=3 -> 1:2:4
    s = float(sum(raw))
    return [r / s for r in raw]


# ---------------------------------------------------------------- 3. 开头句切词
def fmm_segment(text, vocab, max_word_len=6):
    """正向最大匹配：把续写开头切成语料里出现过的词，保证和训练时的切分方式一致。"""
    words, i = [], 0
    while i < len(text):
        for L in range(min(max_word_len, len(text) - i), 0, -1):
            piece = text[i:i + L]
            if L == 1 or piece in vocab:
                words.append(piece)
                i += L
                break
    return words


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="语料文件夹")
    ap.add_argument("--months", type=int, default=6, help="使用前几个月的语料（1~6）")
    ap.add_argument("--max-n", type=int, default=3, help="训练 1..max_n 阶模型做对比")
    ap.add_argument("--temperature", type=float, default=None, help="只用这个温度续写")
    ap.add_argument("--top-k", type=int, default=None, help="只用这个 top-k 续写")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("== 1. 读取并清理语料")
    t0 = time.time()
    sents = load_corpus(args.data, args.months)
    print("  句子数 %d，词数 %d，用时 %.1fs" % (len(sents), sum(map(len, sents)), time.time() - t0))
    print("  清理后示例：", " ".join(sents[0]))

    # 每 20 句取 1 句做测试集，其余训练
    test = sents[::20]
    train = [s for i, s in enumerate(sents) if i % 20]
    print("  训练句 %d，测试句 %d" % (len(train), len(test)))

    print("\n== 2. 训练与评测（困惑度越低越好）")
    print("  %-6s %10s %14s %10s" % ("模型", "训练用时s", "n-gram条目数", "困惑度"))
    models = {}
    for n in range(1, args.max_n + 1):
        m = NGramModel(n)
        t0 = time.time()
        m.train(train)
        cost = time.time() - t0
        ppl = m.perplexity(test)
        models[n] = m
        print("  %-7s %10.1f %14d %10.1f" % ("%d-gram" % n, cost, m.size(), ppl))

    print("\n== 3. 续写：%s……" % PROMPT)
    vocab = models[args.max_n].vocab
    prompt = fmm_segment(PROMPT, vocab)
    print("  开头切词：", " / ".join(prompt))
    if args.temperature is not None or args.top_k is not None:
        settings = [(args.temperature if args.temperature is not None else 1.0, args.top_k or 0)]
    else:
        settings = [(0, 0), (0.7, 10), (1.0, 0), (1.5, 50)]  # (温度, top-k)，0 表示不限
    for n, m in models.items():
        print("\n  [%d-gram]" % n)
        for temp, k in settings:
            text = "".join(m.generate(prompt, temperature=temp, top_k=k, seed=args.seed))
            tag = "贪心" if temp == 0 else "T=%.1f, top-k=%s" % (temp, k or "不限")
            print("   %-18s %s%s" % (tag, PROMPT, text))


if __name__ == "__main__":
    main()
