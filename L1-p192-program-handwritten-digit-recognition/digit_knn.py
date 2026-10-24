# -*- coding: utf-8 -*-
"""
L1 第192页 可选作业：把一个熟悉的项目重新做一次 —— 手写数字识别

只用 Python 标准库，不需要安装任何第三方包。PyCharm 里右键 Run 即可。

数据：digits.csv（1797 张 8x8 灰度手写数字，像素值 0~16）
     来源：UCI Optical Recognition of Handwritten Digits 数据集（scikit-learn 自带的 load_digits 副本），
     公开教学数据，可自由使用。
方法：最近质心（基线） vs. k 近邻（KNN，k=3）
评价：测试集准确率 + 混淆矩阵 + 条件变化/控制样本对测试
"""

import csv
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 8  # 图像是 8x8


# ---------------------------------------------------------------- 数据
def load_data(path=os.path.join(HERE, "digits.csv")):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            y.append(int(row[0]))
            X.append([int(v) for v in row[1:]])
    return X, y


def split(X, y, test_ratio=0.2, seed=0):
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return [X[i] for i in tr], [y[i] for i in tr], [X[i] for i in te], [y[i] for i in te]


# ---------------------------------------------------------------- 两个分类器
def dist2(a, b):
    return sum((p - q) ** 2 for p, q in zip(a, b))


class NearestCentroid:
    """基线：每个数字算一张“平均图”，新图离哪张平均图最近就判成哪个数字。"""

    def fit(self, X, y):
        sums, cnt = {}, Counter(y)
        for x, label in zip(X, y):
            s = sums.setdefault(label, [0.0] * len(x))
            for i, v in enumerate(x):
                s[i] += v
        self.centroids = {c: [v / cnt[c] for v in s] for c, s in sums.items()}
        return self

    def predict_one(self, x):
        return min(self.centroids, key=lambda c: dist2(x, self.centroids[c]))


class KNN:
    """k 近邻：找训练集里最像的 k 张图，投票决定。"""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_one(self, x):
        d = sorted((dist2(x, xi), yi) for xi, yi in zip(self.X, self.y))[:self.k]
        votes = Counter(label for _, label in d)
        return votes.most_common(1)[0][0]


def evaluate(model, X, y):
    pred = [model.predict_one(x) for x in X]
    acc = sum(p == t for p, t in zip(pred, y)) / len(y)
    return acc, pred


# ---------------------------------------------------------------- 图像变换（构造样本对用）
def rot180(x):
    return x[::-1]  # 8x8 展平后整体倒序 = 旋转 180 度


def scale(x, f):
    return [int(round(v * f)) for v in x]  # 笔画变淡/变浓


def shift_right(x, n=1):
    out = []
    for r in range(SIZE):
        row = x[r * SIZE:(r + 1) * SIZE]
        out += [0] * n + row[:SIZE - n]
    return out


def ascii_img(x):
    chars = " .:-=+*#%@"
    return ["".join(chars[min(9, v * 10 // 17)] * 2 for v in x[r * SIZE:(r + 1) * SIZE]) for r in range(SIZE)]


def show_pair(a, b, title_a, title_b):
    A, B = ascii_img(a), ascii_img(b)
    print("    %-18s    %s" % (title_a, title_b))
    for ra, rb in zip(A, B):
        print("    |%s|    |%s|" % (ra, rb))


# ---------------------------------------------------------------- 主程序
def main():
    X, y = load_data()
    Xtr, ytr, Xte, yte = split(X, y)
    print("== 数据：共 %d 张，训练 %d，测试 %d；各类别数量 %s" % (len(X), len(Xtr), len(Xte), dict(sorted(Counter(y).items()))))

    print("\n== 1. 测试集准确率")
    nc = NearestCentroid().fit(Xtr, ytr)
    acc_nc, _ = evaluate(nc, Xte, yte)
    knn = KNN(k=3).fit(Xtr, ytr)
    acc_knn, pred = evaluate(knn, Xte, yte)
    print("  最近质心（基线）: %.1f%%" % (acc_nc * 100))
    print("  KNN (k=3)      : %.1f%%" % (acc_knn * 100))

    print("\n== 2. KNN 混淆矩阵（行=真实，列=预测）")
    print("      " + " ".join("%3d" % c for c in range(10)))
    for t in range(10):
        row = [sum(1 for p, tt in zip(pred, yte) if tt == t and p == c) for c in range(10)]
        print("  %3d " % t + " ".join("%3s" % (v if v else ".") for v in row))
    errs = [(i, t, p) for i, (t, p) in enumerate(zip(yte, pred)) if t != p]
    if errs:
        i, t, p = errs[0]
        print("\n  一个错例（真实 %d，预测 %d）：" % (t, p))
        for line in ascii_img(Xte[i]):
            print("    |%s|" % line)

    print("\n== 3. 条件变化样本对 & 控制样本对")
    print("  只改一个条件，看预测是否按预期改变 / 保持不变（在全部测试样本上统计）")

    def pair_test(name, cond, transform, expect):
        """cond 选出原图；transform 变换；expect(原标签) 给出变换后应得的标签。"""
        items = [(x, t) for x, t in zip(Xte, yte) if cond(t)]
        ok = sum(knn.predict_one(transform(x)) == expect(t) for x, t in items)
        print("  %-34s 符合预期 %3d/%-3d = %5.1f%%" % (name, ok, len(items), 100.0 * ok / len(items)))
        return items

    print("\n  [关键条件变化：答案应该改变]")
    six = pair_test("6 旋转180° → 应判为 9", lambda t: t == 6, rot180, lambda t: 9)
    pair_test("9 旋转180° → 应判为 6", lambda t: t == 9, rot180, lambda t: 6)

    print("\n  [控制样本：答案应该保持不变]")
    zero = pair_test("0 旋转180° → 应仍为 0", lambda t: t == 0, rot180, lambda t: t)
    pair_test("8 旋转180° → 应仍为 8", lambda t: t == 8, rot180, lambda t: t)
    pair_test("所有数字 笔画变淡 ×0.75 → 不变", lambda t: True, lambda x: scale(x, 0.75), lambda t: t)
    pair_test("所有数字 右移1像素 → 不变", lambda t: True, shift_right, lambda t: t)
    pair_test("所有数字 右移2像素 → 不变", lambda t: True, lambda x: shift_right(x, 2), lambda t: t)

    print("\n  样本对示例：")
    x6 = six[0][0]
    show_pair(x6, rot180(x6), "原图 6 → 预测 %d" % knn.predict_one(x6), "旋转后 → 预测 %d" % knn.predict_one(rot180(x6)))
    x0 = zero[0][0]
    show_pair(x0, rot180(x0), "原图 0 → 预测 %d" % knn.predict_one(x0), "旋转后 → 预测 %d" % knn.predict_one(rot180(x0)))
    xs = Xte[0]
    show_pair(xs, shift_right(xs, 2), "原图 %d → 预测 %d" % (yte[0], knn.predict_one(xs)),
              "右移2 → 预测 %d" % knn.predict_one(shift_right(xs, 2)))


if __name__ == "__main__":
    main()
