#!/usr/bin/env python3
"""
按细粒度难度区间统计 Omni-MATH 结果：
- 难度区间：[1,2),[2,3),...,[9,10)，每个区间约 50 题
- 对每个方法，计算：
  * 每个区间中「未超 max token」比例（model_generation 非空）
  * 每个区间中「回答正确」比例（基于 gpt4o_evaluations.jsonl 的 TRUE）
  * 全部 450 题上的上述两个总体比例
同时画出柱状图：
- x 轴：难度区间 + Overall
- y 轴：比例
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Omni-Math.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(BASE_DIR, "imgs")


@dataclass
class SampleInfo:
    index: int
    difficulty: float


def load_dataset_indices() -> Dict[int, SampleInfo]:
    """
    读取 Omni-Math.jsonl，建立 index -> SampleInfo 映射。
    这里假设 jsonl 顺序与推理 / 评测文件一致（你的结果文件就是这样生成的）。
    """
    mapping: Dict[int, SampleInfo] = {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            diff = float(obj.get("difficulty", 0.0))
            mapping[idx] = SampleInfo(index=idx, difficulty=diff)
    return mapping


def parse_eval_file(eval_path: str) -> Dict[int, bool]:
    """
    解析 gpt4o_evaluations.jsonl，返回 index -> 是否正确(TRUE)。
    这里依赖 evaluate_gpt4o.py 的输出格式：
    - 每行:
        {
          "original_json": "<原始样本 json 字符串>",
          "gen": "## Equivalence Judgement\\nTRUE/ FALSE..."
        }
    顺序与原始样本一致，所以用 enumerate 的下标作为 index。
    """
    from aggregate_results import parse_report  # 复用已有解析逻辑

    correctness: Dict[int, bool] = {}
    with open(eval_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            info = parse_report(entry["gen"])
            if not info:
                continue
            is_correct = info.get("Equivalence Judgement", "") == "TRUE"
            correctness[idx] = is_correct
    return correctness


def parse_generation_file(gen_path: str) -> Dict[int, bool]:
    """
    解析 model_generations.jsonl，返回 index -> 是否未超 max token。
    约定：model_generation 字段为空字符串或缺失，表示超出 max token，被返回 None。
    """
    non_overflow: Dict[int, bool] = {}
    with open(gen_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            gen = obj.get("model_generation", "")
            ok = bool(gen and gen.strip())
            non_overflow[idx] = ok
    return non_overflow


def bin_difficulty(diff: float) -> str:
    """
    将原始 difficulty 映射到区间标签：
    [1,2), [2,3), ..., [9,10)，以及一个 'Overall'。
    """
    if diff < 1 or diff > 10:
        return "Other"
    # 右开区间：[k, k+1)
    for k in range(1, 10):
        if k <= diff < k + 1:
            return f"{k}-{k+1}"
    # diff == 10 这种极端情况，强行放到 9-10
    return "9-10"


def collect_stats_for_method(
    method_name: str,
    dataset_mapping: Dict[int, SampleInfo],
) -> Dict[str, Dict[str, float]]:
    """
    对单个方法收集统计：
    返回结构：
    {
      bin_label: {
        'non_overflow_ratio': ...,
        'non_overflow_count': ...,
        'non_overflow_total': ...,
        'accuracy_ratio': ...,
        'accuracy_count': ...,
        'accuracy_total': ...,
      },
      'Overall': {...}
    }
    """
    method_dir = os.path.join(RESULTS_DIR, method_name)
    if not os.path.isdir(method_dir):
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    subdirs = sorted(
        [d for d in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, d))],
        reverse=True,
    )
    if not subdirs:
        raise RuntimeError(f"No timestamp subdirs for method {method_name}")

    latest = subdirs[0]
    result_dir = os.path.join(method_dir, latest)
    gen_file = os.path.join(result_dir, "model_generations.jsonl")
    eval_file = os.path.join(result_dir, "gpt4o_evaluations.jsonl")

    non_overflow = parse_generation_file(gen_file)
    correctness = parse_eval_file(eval_file)

    # 统计器
    bin_non_overflow_total: Dict[str, int] = defaultdict(int)
    bin_non_overflow_count: Dict[str, int] = defaultdict(int)

    bin_acc_total: Dict[str, int] = defaultdict(int)
    bin_acc_count: Dict[str, int] = defaultdict(int)

    # 假定 index 范围由 dataset 决定，且 450 题
    for idx, info in dataset_mapping.items():
        diff_bin = bin_difficulty(info.difficulty)
        if diff_bin == "Other":
            continue

        # 未超 max token（只看是否有生成）
        ok = non_overflow.get(idx, False)
        bin_non_overflow_total[diff_bin] += 1
        if ok:
            bin_non_overflow_count[diff_bin] += 1

        # 正确率（只在有 GPT 评测结果的样本上统计）
        if idx in correctness:
            is_correct = correctness[idx]
            bin_acc_total[diff_bin] += 1
            if is_correct:
                bin_acc_count[diff_bin] += 1

    # 计算总体
    overall_non_overflow_total = sum(bin_non_overflow_total.values())
    overall_non_overflow_count = sum(bin_non_overflow_count.values())

    overall_acc_total = sum(bin_acc_total.values())
    overall_acc_count = sum(bin_acc_count.values())

    stats: Dict[str, Dict[str, float]] = {}

    for bin_label in sorted(bin_non_overflow_total.keys(), key=lambda x: (x == "Overall", x)):
        tot_no = bin_non_overflow_total[bin_label]
        cnt_no = bin_non_overflow_count[bin_label]

        tot_acc = bin_acc_total.get(bin_label, 0)
        cnt_acc = bin_acc_count.get(bin_label, 0)

        stats[bin_label] = {
            "non_overflow_ratio": cnt_no / tot_no if tot_no > 0 else 0.0,
            "non_overflow_count": cnt_no,
            "non_overflow_total": tot_no,
            "accuracy_ratio": cnt_acc / tot_acc if tot_acc > 0 else 0.0,
            "accuracy_count": cnt_acc,
            "accuracy_total": tot_acc,
        }

    # 添加 Overall
    stats["Overall"] = {
        "non_overflow_ratio": overall_non_overflow_count / overall_non_overflow_total
        if overall_non_overflow_total > 0
        else 0.0,
        "non_overflow_count": overall_non_overflow_count,
        "non_overflow_total": overall_non_overflow_total,
        "accuracy_ratio": overall_acc_count / overall_acc_total if overall_acc_total > 0 else 0.0,
        "accuracy_count": overall_acc_count,
        "accuracy_total": overall_acc_total,
    }

    return stats


def plot_method_stats(method_name: str, stats: Dict[str, Dict[str, float]]) -> None:
    """
    画出该方法在各难度区间及总体的比例柱状图：
    - 一张图里两组柱：未超 max token 比例 / 正确率。
    """
    os.makedirs(IMG_DIR, exist_ok=True)

    # 按区间顺序排：1-2,2-3,...,9-10,Overall
    bins = [f"{k}-{k+1}" for k in range(1, 10)] + ["Overall"]
    bins = [b for b in bins if b in stats]

    x = list(range(len(bins)))
    non_overflow_values = [stats[b]["non_overflow_ratio"] for b in bins]
    acc_values = [stats[b]["accuracy_ratio"] for b in bins]

    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], non_overflow_values, width=width, label="Non-overflow ratio")
    plt.bar([i + width / 2 for i in x], acc_values, width=width, label="Accuracy ratio")

    plt.xticks(x, bins, rotation=45)
    plt.ylim(0, 1.05)
    plt.ylabel("Ratio")
    plt.xlabel("Difficulty bin")
    plt.title(f"{method_name}: Non-overflow & Accuracy by difficulty")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(IMG_DIR, f"{method_name}_difficulty_bins.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {out_path}")


def main():
    dataset_mapping = load_dataset_indices()

    # 你当前有的三种方法
    methods = ["Qwen3-30B-baseline", "Qwen3-Feb3-400", "Qwen3-30B-with-prompt"]

    all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for m in methods:
        print(f"\n==== Method: {m} ====")
        stats = collect_stats_for_method(m, dataset_mapping)
        all_stats[m] = stats

        # 文本打印主要结果
        for bin_label in [f"{k}-{k+1}" for k in range(1, 10)] + ["Overall"]:
            if bin_label not in stats:
                continue
            s = stats[bin_label]
            print(
                f"{bin_label}: "
                f"non-overflow {s['non_overflow_count']}/{s['non_overflow_total']} "
                f"({s['non_overflow_ratio']:.3f}), "
                f"accuracy {s['accuracy_count']}/{s['accuracy_total']} "
                f"({s['accuracy_ratio']:.3f})"
            )

        # 画图
        plot_method_stats(m, stats)


if __name__ == "__main__":
    main()

