# -*- coding: utf-8 -*-
"""
公共部分：人民日报语料的读取与清理、结果日志、fastText 临时目录。
word2vec_gensim.py 和 fasttext_classify.py 都会用到。
"""

import glob
import os
import re
import sys
import tempfile
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
# 语料默认放在仓库根目录：LLM-course-2026/1998-上半年-带音(1)/1998-上半年-带音/
DEFAULT_DATA = os.path.join(HERE, "..", "1998-上半年-带音(1)", "1998-上半年-带音")

LINE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-\d{3}\s+(.*)$")  # 日期-版面-文章号-段落号 正文
PINYIN = re.compile(r"\{[^}]*\}")                               # 拼音标注，如 这{zhe4}
FULL2HALF = {c: c - 0xFEE0 for c in list(range(0xFF10, 0xFF1A)) +
             list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}
PUNCT = set("，。、；：？！“”‘’（）《》—…·「」『』〓")


def corpus_files(data_dir, months):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))[:months]
    if not files:
        raise SystemExit("在 %s 下没有找到语料 txt，请用 --data 指定语料文件夹" % data_dir)
    return files


def iter_paragraphs(data_dir, months):
    """逐段返回 (日期, 版面, 文章号, 词列表)。已去掉行首编号、拼音标注和标点，全角数字字母转半角。"""
    for path in corpus_files(data_dir, months):
        with open(path, encoding="gbk", errors="ignore") as f:  # 原始语料是 GBK 编码
            for line in f:
                m = LINE.match(line.strip())
                if not m:
                    continue
                text = PINYIN.sub("", m.group(4)).translate(FULL2HALF)
                words = [sys.intern(w) for w in text.split() if w not in PUNCT]  # intern：相同的词共用一份内存
                if words:
                    yield m.group(1), m.group(2), m.group(3), words


def load_sentences(data_dir, months=6):
    """表示学习用：每个段落作为一个“句子”（词列表）。"""
    return [words for _, _, _, words in iter_paragraphs(data_dir, months)]


def load_articles(data_dir, max_words=200):
    """分类用：1998 年 1 月（9801）的文章，返回 [(版面号, 词列表)]，每篇取前 max_words 个词。"""
    arts = OrderedDict()
    for date, page, art, words in iter_paragraphs(data_dir, 1):
        arts.setdefault((date, page, art), []).extend(words)
    return [(key[1], words[:max_words]) for key, words in arts.items()]


def make_logger(filename):
    """返回一个 log 函数：同时打印到屏幕并写入本文件夹下的 filename（每次运行覆盖）。"""
    path = os.path.join(HERE, filename)
    f = open(path, "w", encoding="utf-8")

    def log(*args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        f.write(msg + "\n")
        f.flush()

    return log


def ft_workdir():
    """fastText 的训练/测试文件和预训练词向量存放的目录。

    fastText 的 C++ 代码在 Windows 上打不开含中文的路径，而本仓库的路径里有中文，
    所以这些中间文件放到系统临时目录（如 C:\\Users\\xxx\\AppData\\Local\\Temp\\llm_p137）。
    如果临时目录本身也含中文，就退回到当前盘符根目录下的 llm_p137_tmp。
    """
    d = os.path.join(tempfile.gettempdir(), "llm_p137")
    if not d.isascii():
        d = os.path.join(os.path.splitdrive(HERE)[0] + os.sep, "llm_p137_tmp")
    os.makedirs(d, exist_ok=True)
    return d


PRETRAINED_VEC = "word2vec_sg_100d.vec"  # word2vec_gensim.py 导出、fasttext_classify.py 使用的词向量文件名
