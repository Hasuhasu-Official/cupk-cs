#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验三：关联规则挖掘（Apriori）——美国国会投票记录（House Votes 84）
- 数据集：UCI Congressional Voting Records (435 条记录，16 个投票特征，含缺失 ?)
- 任务：使用 Apriori 挖掘高置信度规则
- 参数：min_support = 0.30（30%），min_confidence = 0.90（90%）
- 输出：频繁项集 CSV、关联规则 CSV、（可选）统计图 PNG

复现性说明：
- Apriori 过程无随机性；只要数据一致、阈值一致，输出结果确定可复现。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import textwrap
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple


UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
)

# 16 个投票字段名（与 UCI names 文件一致的常见命名）
VOTE_FIELDS = [
    "handicapped-infants",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missile",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa",
]


@dataclass(frozen=True)
class Rule:
    antecedent: FrozenSet[str]
    consequent: FrozenSet[str]
    support_count: int
    support: float
    confidence: float
    lift: float


def download_if_needed(dst_path: str, url: str = UCI_DATA_URL) -> None:
    """
    若本地不存在数据文件，则从 UCI URL 下载。
    注意：若运行环境无网络，请手动下载后通过 --data_path 指定。
    """
    if os.path.exists(dst_path):
        return

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    print(f"[INFO] Local data not found. Downloading from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, dst_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download dataset. Please download manually and rerun with --data_path.\n"
            f"Target URL: {url}\nError: {e}"
        ) from e
    print(f"[INFO] Downloaded to: {dst_path}")


def read_house_votes84(data_path: str) -> List[Tuple[str, List[str]]]:
    """
    读取 house-votes-84.data
    每行格式：party, v1, v2, ... v16  其中 vote ∈ {y, n, ?}
    返回：[(party, [v1..v16]), ...]
    """
    records: List[Tuple[str, List[str]]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 17:
                raise ValueError(
                    f"Invalid line length at line {line_no}: expected 17 fields, got {len(parts)}"
                )
            party = parts[0]
            votes = parts[1:]
            records.append((party, votes))
    return records


def build_transactions(
    records: List[Tuple[str, List[str]]],
    include_party_item: bool = True,
    missing_policy: str = "ignore",
) -> Tuple[List[Set[str]], Dict[str, float]]:
    """
    将原始记录转换为交易集（transaction database）
    - 编码方式：每个“字段=取值”作为一个 item，例如 physician-fee-freeze=y
    - include_party_item=True 时，将 party 也作为 item（如 party=democrat）
      这有利于挖掘“投票组合 -> 党派”的规则
    - missing_policy:
        - ignore：遇到 ? 不加入该字段对应的 item（常用，减少噪声）
        - as_value：将 ? 作为一个取值（即 field=? 也作为 item）
    返回：
    - transactions: List[Set[item]]
    - stats: 一些统计信息（缺失率等）
    """
    if missing_policy not in {"ignore", "as_value"}:
        raise ValueError("missing_policy must be 'ignore' or 'as_value'")

    transactions: List[Set[str]] = []
    total_votes = 0
    missing_votes = 0

    for party, votes in records:
        t: Set[str] = set()

        if include_party_item:
            t.add(f"party={party}")

        for field, v in zip(VOTE_FIELDS, votes):
            total_votes += 1
            if v == "?":
                missing_votes += 1
                if missing_policy == "as_value":
                    t.add(f"{field}=?")
                # ignore 时不加入任何 item
            else:
                # 只加入 y/n
                t.add(f"{field}={v}")
        transactions.append(t)

    missing_rate = (missing_votes / total_votes) if total_votes else 0.0
    stats = {
        "n_transactions": float(len(transactions)),
        "total_votes": float(total_votes),
        "missing_votes": float(missing_votes),
        "missing_rate": missing_rate,
    }
    return transactions, stats


def support_count(
    transactions: List[Set[str]], itemset: FrozenSet[str]
) -> int:
    """计算给定 itemset 在 transactions 中出现（子集）的次数。"""
    cnt = 0
    for t in transactions:
        if itemset.issubset(t):
            cnt += 1
    return cnt


def apriori(
    transactions: List[Set[str]],
    min_support: float,
    max_len: int | None = None,
) -> Dict[FrozenSet[str], int]:
    """
    Apriori 主过程：返回所有频繁项集的支持度计数 support_count
    - min_support 为比例（如 0.30）
    - max_len 可限制最大项集大小（可选，默认不限制）
    """
    n = len(transactions)
    if n == 0:
        return {}

    min_sup_cnt = int(math.ceil(min_support * n))
    print(f"[INFO] n_transactions = {n}")
    print(f"[INFO] min_support = {min_support:.2f} => min_support_count = {min_sup_cnt}")

    # 1) 统计 1-itemset
    item_counter: Dict[str, int] = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counter[item] += 1

    freq: Dict[FrozenSet[str], int] = {}
    Lk: List[FrozenSet[str]] = []

    for item, cnt in item_counter.items():
        if cnt >= min_sup_cnt:
            fs = frozenset([item])
            freq[fs] = cnt
            Lk.append(fs)

    # 为了输出稳定（可复现），对 L1 排序
    Lk.sort(key=lambda s: (next(iter(s))))

    k = 2
    while Lk:
        if max_len is not None and k > max_len:
            break

        # 2) 由 L(k-1) 生成候选 Ck（join + prune）
        Ck = generate_candidates(Lk, k)
        if not Ck:
            break

        # 3) 计数
        Ck_count: Dict[FrozenSet[str], int] = defaultdict(int)
        for t in transactions:
            for c in Ck:
                if c.issubset(t):
                    Ck_count[c] += 1

        # 4) 剪枝：保留频繁项集 Lk
        Lk_next: List[FrozenSet[str]] = []
        for c, cnt in Ck_count.items():
            if cnt >= min_sup_cnt:
                freq[c] = cnt
                Lk_next.append(c)

        # 稳定排序，保证输出可复现
        Lk_next.sort(key=lambda s: (sorted(list(s))))
        Lk = Lk_next
        k += 1

    print(f"[INFO] total frequent itemsets = {len(freq)}")
    return freq


def generate_candidates(L_prev: List[FrozenSet[str]], k: int) -> List[FrozenSet[str]]:
    """
    由频繁 (k-1)-项集生成候选 k-项集：
    - join：两两合并，若 union 的大小为 k
    - prune：候选的任意 (k-1) 子集必须在 L_prev 中（Apriori 性质）
    """
    L_prev_set = set(L_prev)
    candidates: Set[FrozenSet[str]] = set()

    # join
    for i in range(len(L_prev)):
        for j in range(i + 1, len(L_prev)):
            union = L_prev[i] | L_prev[j]
            if len(union) == k:
                candidates.add(frozenset(union))

    # prune
    pruned: List[FrozenSet[str]] = []
    for c in candidates:
        all_subsets_frequent = True
        for subset in combinations(sorted(c), k - 1):
            if frozenset(subset) not in L_prev_set:
                all_subsets_frequent = False
                break
        if all_subsets_frequent:
            pruned.append(c)

    # 稳定排序
    pruned.sort(key=lambda s: (sorted(list(s))))
    return pruned


def generate_rules(
    freq_itemsets: Dict[FrozenSet[str], int],
    n_transactions: int,
    min_confidence: float,
) -> List[Rule]:
    """
    由频繁项集生成关联规则：
    对每个 |I|>=2 的频繁项集 I，枚举非空真子集 A 作为 antecedent，
    consequent = I \\ A
    confidence = sup(I) / sup(A)
    lift = confidence / sup(consequent)
    """
    rules: List[Rule] = []

    for itemset, sup_cnt in freq_itemsets.items():
        if len(itemset) < 2:
            continue

        items_sorted = sorted(itemset)
        # 枚举 antecedent 大小从 1 到 len-1
        for r in range(1, len(items_sorted)):
            for antecedent_tuple in combinations(items_sorted, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = frozenset(itemset - antecedent)

                sup_a = freq_itemsets.get(antecedent)
                sup_c = freq_itemsets.get(consequent)
                if sup_a is None or sup_c is None:
                    # 理论上频繁项集的子集都应频繁；这里做防御式处理
                    continue

                conf = sup_cnt / sup_a
                if conf + 1e-12 < min_confidence:
                    continue

                sup = sup_cnt / n_transactions
                sup_consequent = sup_c / n_transactions
                lift = conf / sup_consequent if sup_consequent > 0 else float("inf")

                rules.append(
                    Rule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support_count=sup_cnt,
                        support=sup,
                        confidence=conf,
                        lift=lift,
                    )
                )

    # 稳定排序：先按 confidence desc，再 lift desc，再 support desc，再规则字典序
    rules.sort(
        key=lambda r: (
            -r.confidence,
            -r.lift,
            -r.support,
            sorted(list(r.antecedent)),
            sorted(list(r.consequent)),
        )
    )
    return rules


def save_frequent_itemsets_csv(
    freq_itemsets: Dict[FrozenSet[str], int],
    n_transactions: int,
    out_path: str,
) -> None:
    rows = []
    for itemset, cnt in freq_itemsets.items():
        rows.append(
            (
                len(itemset),
                "{" + ", ".join(sorted(itemset)) + "}",
                cnt,
                cnt / n_transactions,
            )
        )
    rows.sort(key=lambda x: (-x[3], x[0], x[1]))  # support desc

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "itemset", "support_count", "support"])
        w.writerows(rows)


def save_rules_csv(rules: List[Rule], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["antecedent", "consequent", "support_count", "support", "confidence", "lift"]
        )
        for r in rules:
            w.writerow(
                [
                    "{" + ", ".join(sorted(r.antecedent)) + "}",
                    "{" + ", ".join(sorted(r.consequent)) + "}",
                    r.support_count,
                    f"{r.support:.6f}",
                    f"{r.confidence:.6f}",
                    f"{r.lift:.6f}",
                ]
            )


def maybe_make_plots(freq_itemsets: Dict[FrozenSet[str], int], rules: List[Rule], out_dir: str) -> None:
    """
    可选：生成统计图（若 matplotlib 可用）
    输出（默认保存到 out_dir）：
    1) frequent_itemsets_by_size.png         不同 k 的频繁项集数量
    2) freq_itemsets_support_hist.png        频繁项集支持度分布
    3) rules_confidence_hist.png             规则置信度分布（当前 rules 已经过 conf 阈值筛选）
    4) rules_lift_hist.png                   规则 lift 分布
    5) rules_support_vs_confidence.png       规则 support- confidence 散点图
    """
    import os
    from collections import defaultdict

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available. Skip plotting.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) 频繁项集数量 vs k ----------
    size_count = defaultdict(int)
    for itemset in freq_itemsets.keys():
        size_count[len(itemset)] += 1
    xs = sorted(size_count.keys())
    ys = [size_count[k] for k in xs]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Itemset size k")
    plt.ylabel("Number of frequent itemsets")
    plt.title("Frequent Itemsets Count by Size")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "frequent_itemsets_by_size.png"), dpi=200)
    plt.close()

    # ---------- 2) 频繁项集支持度分布 ----------
    # 这里的 support 是比例，需要知道事务总数 n。freq_itemsets 只有计数，因此不在此计算比例直方图。
    # 解决：从计数做相对分布（support_count / n）需要 n；如果你愿意传入 n，可以把 n 作为参数更规范。
    # 先用“支持计数”直方图替代（报告依然可用，注明为 support_count 分布）。
    counts = list(freq_itemsets.values())
    if counts:
        plt.figure()
        plt.hist(counts, bins=20)
        plt.xlabel("Support count")
        plt.ylabel("Number of frequent itemsets")
        plt.title("Support Count Distribution of Frequent Itemsets")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "freq_itemsets_supportcount_hist.png"), dpi=200)
        plt.close()

    # ---------- 3) 规则置信度分布 ----------
    confs = [r.confidence for r in rules]
    if confs:
        plt.figure()
        plt.hist(confs, bins=20)
        plt.xlabel("Confidence")
        plt.ylabel("Number of rules")
        plt.title("Confidence Distribution of Rules")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rules_confidence_hist.png"), dpi=200)
        plt.close()

    # ---------- 4) 规则 lift 分布 ----------
    lifts = [r.lift for r in rules if r.lift == r.lift]  # 排除 NaN
    if lifts:
        plt.figure()
        plt.hist(lifts, bins=20)
        plt.xlabel("Lift")
        plt.ylabel("Number of rules")
        plt.title("Lift Distribution of Rules")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rules_lift_hist.png"), dpi=200)
        plt.close()

    # ---------- 5) support vs confidence 散点图 ----------
    sups = [r.support for r in rules]
    if sups and confs:
        plt.figure()
        plt.scatter(sups, confs, s=10)
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Support vs Confidence of Rules")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rules_support_vs_confidence.png"), dpi=200)
        plt.close()



def print_top_itemsets(freq_itemsets: Dict[FrozenSet[str], int], n: int, n_transactions: int) -> None:
    rows = []
    for itemset, cnt in freq_itemsets.items():
        rows.append((cnt / n_transactions, cnt, itemset))
    rows.sort(key=lambda x: (-x[0], len(x[2]), sorted(list(x[2]))))

    print("\n[TOP] Frequent itemsets:")
    for i, (sup, cnt, itemset) in enumerate(rows[:n], start=1):
        items = ", ".join(sorted(itemset))
        print(f"  #{i:02d} support={sup:.4f} ({cnt})  {{{items}}}")


def print_top_rules(rules: List[Rule], n: int) -> None:
    print("\n[TOP] High-confidence rules:")
    for i, r in enumerate(rules[:n], start=1):
        a = ", ".join(sorted(r.antecedent))
        c = ", ".join(sorted(r.consequent))
        print(
            f"  #{i:02d} conf={r.confidence:.4f}  lift={r.lift:.4f}  "
            f"sup={r.support:.4f} ({r.support_count})\n"
            f"      {{{a}}}  =>  {{{c}}}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Apriori association rule mining on UCI House Votes 84 dataset.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/house-votes-84.data",
        help="Path to house-votes-84.data (default: data/house-votes-84.data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/exp3_apriori",
        help="Output directory (default: outputs/exp3_apriori).",
    )
    parser.add_argument(
        "--min_support",
        type=float,
        default=0.30,
        help="Minimum support ratio (default: 0.30).",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.90,
        help="Minimum confidence (default: 0.90).",
    )
    parser.add_argument(
        "--include_party",
        action="store_true",
        default=True,
        help="Include party as an item (default: True).",
    )
    parser.add_argument(
        "--no_party",
        action="store_true",
        help="Do not include party as an item (override --include_party).",
    )
    parser.add_argument(
        "--missing_policy",
        type=str,
        choices=["ignore", "as_value"],
        default="ignore",
        help="How to handle '?' missing votes: ignore or as_value (default: ignore).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Max itemset length (optional).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Print top-N itemsets/rules (default: 20).",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Generate simple plots (requires matplotlib).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    include_party_item = bool(args.include_party) and (not bool(args.no_party))

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 获取数据（若缺失则下载）
    try:
        download_if_needed(args.data_path, UCI_DATA_URL)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    # 2) 读取并构造 transactions
    records = read_house_votes84(args.data_path)
    transactions, stats = build_transactions(
        records=records,
        include_party_item=include_party_item,
        missing_policy=args.missing_policy,
    )

    n_transactions = len(transactions)
    print("\n[INFO] Dataset stats:")
    print(f"  records = {len(records)}")
    print(f"  transactions = {n_transactions}")
    print(f"  missing_rate (votes) = {stats['missing_rate']:.4f}")
    print(f"  include_party_item = {include_party_item}")
    print(f"  missing_policy = {args.missing_policy}")

    # 3) Apriori 频繁项集
    freq_itemsets = apriori(
        transactions=transactions,
        min_support=args.min_support,
        max_len=args.max_len,
    )

    # 4) 生成规则
    rules = generate_rules(
        freq_itemsets=freq_itemsets,
        n_transactions=n_transactions,
        min_confidence=args.min_confidence,
    )
    print(f"[INFO] total rules (conf >= {args.min_confidence:.2f}) = {len(rules)}")

    # 5) 保存结果
    itemsets_csv = os.path.join(args.output_dir, "frequent_itemsets.csv")
    rules_csv = os.path.join(args.output_dir, "association_rules.csv")
    save_frequent_itemsets_csv(freq_itemsets, n_transactions, itemsets_csv)
    save_rules_csv(rules, rules_csv)
    print(f"[INFO] Saved frequent itemsets to: {itemsets_csv}")
    print(f"[INFO] Saved rules to: {rules_csv}")

    # 6) 终端展示 Top-N
    print_top_itemsets(freq_itemsets, n=args.top_n, n_transactions=n_transactions)
    print_top_rules(rules, n=args.top_n)

    # 7) 可选作图（便于报告截图）
    if args.make_plots:
        maybe_make_plots(freq_itemsets, rules, args.output_dir)
        print(f"[INFO] Plots saved in: {args.output_dir}")

    print("\n[INFO] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
