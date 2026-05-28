import os
import json
import argparse
import csv


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_dataset_items(data_root):
    """按数据文件条数统计题目数量：data/{data_root}.json"""
    data_path = os.path.join("data/test", f"{data_root}.json")
    data = load_json(data_path)
    if data is None:
        return 0
    if isinstance(data, list):
        return len(data)
    # 保险起见，如果不是 list，就尽量从常见字段里取长度
    for key in ["data", "examples"]:
        if key in data and isinstance(data[key], list):
            return len(data[key])
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name

    # 与 run_evaluation.sh 中保持一致
    datasets = ["gsm8k", "math", "svamp", "formula"]
    datasets_with_normal = ["gsm8k", "math"]
    datasets_mip_only = ["svamp", "formula"]

    # 保存每个 (dataset, version) 的统计
    stats = {}
    # 用于最终写入 CSV 的行
    csv_rows = []

    for dataset in datasets:
        # 判定有哪些 version
        if dataset in datasets_mip_only:
            versions = ["MiP"]
        else:
            versions = ["normal", "MiP"]

        # 题目总数（一个数据集在 normal / MiP 下用的其实是同一份题，但为了清晰，这里每个 version 单独记录）
        n_items = count_dataset_items(dataset)

        for version in versions:
            result_dir = os.path.join("results", dataset, version, model_name)
            analysis_path = os.path.join(result_dir, "analysis_info.json")
            info = load_json(analysis_path)
            if info is None:
                continue

            evaluation = info.get("evaluation_results", {})
            mean_answer_length = info.get("mean_answer_length", None)

            # 有的文件可能还没 eval 完成，跳过（完全没有 evaluation_results 才跳）
            if not evaluation:
                continue

            # 对于 MiP，我们关心 insufficient_condition；Normal 的话一般为 0
            insufficient_rate = evaluation.get("insufficient_condition", 0.0)
            correct_rate = evaluation.get("correct_answer", 0.0)

            stats[(dataset, version)] = {
                "n": n_items,
                "insufficient_rate": insufficient_rate,
                "correct_rate": correct_rate,
                "mean_answer_length": mean_answer_length,
            }

    print("\n==================== MiP 各数据集结果 ====================\n")
    for dataset in datasets:
        key = (dataset, "MiP")
        if key not in stats:
            continue
        s = stats[key]
        print(
            f"[MiP] {dataset}: "
            f"题目数 = {s['n']}, "
            f"弃答率(insufficient_condition) = {s['insufficient_rate']:.4f}, "
            f"mean_answer_length = {s['mean_answer_length']:.2f}"
        )
        csv_rows.append(
            {
                "level": "dataset",
                "dataset": dataset,
                "version": "MiP",
                "n": s["n"],
                "insufficient_rate": s["insufficient_rate"],
                "correct_rate": s["correct_rate"],
                "mean_answer_length": s["mean_answer_length"],
            }
        )

    print("\n==================== Normal (仅 gsm8k / math) 各数据集结果 ====================\n")
    for dataset in datasets_with_normal:
        key = (dataset, "normal")
        if key not in stats:
            continue
        s = stats[key]
        print(
            f"[Normal] {dataset}: "
            f"题目数 = {s['n']}, "
            f"accuracy(correct_answer) = {s['correct_rate']:.4f}, "
            f"mean_answer_length = {s['mean_answer_length']:.2f}"
        )
        csv_rows.append(
            {
                "level": "dataset",
                "dataset": dataset,
                "version": "normal",
                "n": s["n"],
                "insufficient_rate": s["insufficient_rate"],
                "correct_rate": s["correct_rate"],
                "mean_answer_length": s["mean_answer_length"],
            }
        )

    # 计算两大类的加权汇总
    def aggregate(category_keys):
        total_n = sum(stats[k]["n"] for k in category_keys if k in stats)
        if total_n == 0:
            return None
        abst = 0.0
        acc = 0.0
        mean_len = 0.0
        for k in category_keys:
            if k not in stats:
                continue
            s = stats[k]
            n = s["n"]
            abst += s["insufficient_rate"] * n
            acc += s["correct_rate"] * n
            if s["mean_answer_length"] is not None:
                mean_len += s["mean_answer_length"] * n
        abst /= total_n
        acc /= total_n
        mean_len /= total_n
        return {
            "n": total_n,
            "abst_rate": abst,
            "acc": acc,
            "mean_answer_length": mean_len,
        }

    mip_keys = [(d, "MiP") for d in datasets]
    normal_keys = [(d, "normal") for d in datasets_with_normal]

    mip_agg = aggregate(mip_keys)
    normal_agg = aggregate(normal_keys)

    print("\n==================== MiP / Normal 综合结果 ====================\n")
    if mip_agg is not None:
        print(
            "MiP 综合："
            f"总题目数 = {mip_agg['n']}, "
            f"弃答率 = {mip_agg['abst_rate']:.4f}, "
            f"accuracy = {mip_agg['acc']:.4f}, "
            f"mean_answer_length = {mip_agg['mean_answer_length']:.2f}"
        )
        csv_rows.append(
            {
                "level": "aggregate",
                "dataset": "ALL",
                "version": "MiP",
                "n": mip_agg["n"],
                "insufficient_rate": mip_agg["abst_rate"],
                "correct_rate": mip_agg["acc"],
                "mean_answer_length": mip_agg["mean_answer_length"],
            }
        )
    else:
        print("MiP 综合：无有效结果")

    if normal_agg is not None:
        print(
            "Normal 综合："
            f"总题目数 = {normal_agg['n']}, "
            f"弃答率 = {normal_agg['abst_rate']:.4f}, "
            f"accuracy = {normal_agg['acc']:.4f}, "
            f"mean_answer_length = {normal_agg['mean_answer_length']:.2f}"
        )
        csv_rows.append(
            {
                "level": "aggregate",
                "dataset": "ALL",
                "version": "normal",
                "n": normal_agg["n"],
                "insufficient_rate": normal_agg["abst_rate"],
                "correct_rate": normal_agg["acc"],
                "mean_answer_length": normal_agg["mean_answer_length"],
            }
        )
    else:
        print("Normal 综合：无有效结果")

    # ------------------------------------------------------------------
    # 将所有结果保存到 CSV：results/summary/{model_name}/summary.csv
    # ------------------------------------------------------------------
    if csv_rows:
        summary_dir = os.path.join("results", "summary", model_name)
        os.makedirs(summary_dir, exist_ok=True)
        csv_path = os.path.join(summary_dir, "summary.csv")
        fieldnames = [
            "level",
            "dataset",
            "version",
            "n",
            "insufficient_rate",
            "correct_rate",
            "mean_answer_length",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"\nSummary CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    main()

